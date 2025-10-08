import math
import torch
import collections
import gc
import copy
import logging
import numpy as np
import comfy
from comfy import sample as comfy_sample
from comfy import model_management as mm
from comfy.utils import ProgressBar
from contextlib import nullcontext


log = logging.getLogger(__name__)

# check for expected repository globals 
if "VAE_STRIDE" in globals():
    VAE_STRIDE = globals()["VAE_STRIDE"]
else:
    log.warning("VAE_STRIDE not found in module globals; using fallback default of (4, 8, 8).")
    VAE_STRIDE = (4, 8, 8)

if "scheduler_list" not in globals():
    log.warning("scheduler_list not found in module globals; UI dropdown may be incomplete.")
    # Fallback to comfy sampler scheduler list if the repo-specific list isn't present
    scheduler_list = comfy.samplers.KSampler.SCHEDULERS


class CustomVACEStyleTransfer:
    """
    Consumes pre-computed WANVIDIMAGE_EMBEDS for long-term motion guidance and
    uses a short-term, iterative FramePack context for temporal coherence.
    Generates the final video via chunked diffusion, ensuring memory safety for long videos.
    Assumes batch size of 1 for all operations.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vace_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Precomputed VACE embeddings from WanVideoVACEEncode."}),
                "initial_image": ("IMAGE", {"tooltip": "Optional style/first-frame reference (H,W,C) to seed short-term context."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (scheduler_list,),
                "chunk_size": ("INT", {"default": 16, "min": 4, "max": 512, "step": 4, "tooltip": "Pixel frames per chunk (must be a multiple of VAE stride)."}),
                "context_size": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "Short-term context in PIXEL frames for FramePack."}),
                "overlap": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "Pixel overlap between chunks for blending."}),
                "low_vram": ("BOOLEAN", {"default": False, "tooltip": "Aggressively offload models between chunks to save VRAM."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Passed to the initial_image encoder if used."}),
            },
            "optional": {
                "seq_max_warn": ("INT", {"default": 8192, "min": 1024, "max": 65536}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "WanVideoWrapper"

    def _normalize_vace_context(self, vace_embeds):
        """Robustly normalizes vace_embeds['vace_context'] into a list of per-latent-frame CPU tensors."""
        out_list = []
        vctx = vace_embeds.get("vace_context")
        if vctx is None: return out_list
        items_to_process = vctx if isinstance(vctx, list) else [vctx]
        for item in items_to_process:
            t = torch.as_tensor(item).detach()
            if t.ndim == 3: # (C, H, W) -> (C, 1, H, W)
                out_list.append(t.unsqueeze(1).cpu().contiguous())
            elif t.ndim == 4: # Assume (C, T, H, W)
                for i in range(t.shape[1]):
                    out_list.append(t[:, i:i+1].cpu().contiguous())
        
        # Final validation of shapes
        for i, t in enumerate(out_list):
            if t.ndim != 4 or t.shape[1] != 1:
                raise RuntimeError(f"Normalized vace_context[{i}] has unexpected shape {tuple(t.shape)}; expected (C,1,H,W).")
        return out_list

    def _prepare_initial_framepack(self, vae, initial_image, device, dtype, H_lat, W_lat, tiled_vae):
        """Builds a single-frame latent from initial_image using a robust VAE encode call."""
        C_lat = getattr(vae, "latent_channels", 4)
        fallback_zeros = torch.zeros((1, C_lat, 1, H_lat, W_lat), device=device, dtype=dtype)
        if initial_image is None: return fallback_zeros

        try:
            img = initial_image.clone().to(torch.float32)
            if img.ndim == 3: img = img.unsqueeze(0)
            if img.shape[-1] == 4: img = img[..., :3]
            fmin, fmax = float(img.min()), float(img.max())
            if fmax > 1.5: img = img / 255.0
            img = (img * 2.0) - 1.0
            img_chw = img.permute(0, 3, 1, 2).to(device=device, dtype=dtype)

            with torch.no_grad():
                # Robust VAE encode with fallbacks
                try:
                    lat = vae.encode([img_chw], device=device, tiled=tiled_vae)[0]
                except TypeError:
                    try:
                        lat = vae.encode([img_chw])[0]
                    except Exception:
                        lat = vae.encode(img_chw)
            
            if lat is None:
                log.warning("VAE returned None for initial image; falling back to zeros.")
                # free caches proactively
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return fallback_zeros
            if lat.ndim == 4: lat = lat.unsqueeze(2)
            if lat.ndim != 5:
                raise RuntimeError(f"Unexpected latent shape from VAE for initial image: {lat.shape}")
            return lat.detach()
        except Exception as e:
            log.warning(f"Failed to encode initial_image for FramePack context: {e}. Falling back to zeros.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return fallback_zeros

    def _build_framepack_cond(self, svd_model, framepack_context, chunk_px_len, vace_embeds, device, dtype):
        """Builds the short-term conditioning using the model's conditioner/embedder."""
        try:
            conditioner = svd_model.model.conditioner.embedder
            target_shape = vace_embeds.get("target_shape")
            if target_shape:
                pixels_h, pixels_w = int(target_shape[2] * VAE_STRIDE[1]), int(target_shape[3] * VAE_STRIDE[2])
            else:
                pixels_h, pixels_w = framepack_context.shape[3] * VAE_STRIDE[1], framepack_context.shape[4] * VAE_STRIDE[2]
            
            value_and_mask = conditioner.standard_embedder.get_value(
                chunk_px_len, pixels_h, pixels_w, vace_embeds.get("fps", 6), 
                vace_embeds.get("motion_bucket_id", 127), device
            )
            # Ensure value_and_mask is on the right device/dtype if it's a tensor/tuple
            try:
                if isinstance(value_and_mask, tuple):
                    value_and_mask = tuple(v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v for v in value_and_mask)
                elif isinstance(value_and_mask, torch.Tensor):
                    value_and_mask = value_and_mask.to(device=device, dtype=dtype)
            except Exception:
                pass

            return conditioner(framepack_context.to(dtype=dtype, device=device), value_and_mask)
        except Exception as e:
            if not getattr(self, "_logged_framepack_failure", False):
                log.info(f"FramePack conditioning not available or failed: {e}. Continuing without short-term context.")
                self._logged_framepack_failure = True
            return None

    def generate(self, model, vace_embeds, initial_image=None, seed=0, steps=25, cfg=7.0,
                 sampler_name="k_euler_ancestral", scheduler=None, chunk_size=16,
                 context_size=4, overlap=4, low_vram=False, tiled_vae=False, **kwargs):

        if not isinstance(vace_embeds, dict): raise ValueError("vace_embeds must be a dict.")
        if overlap >= chunk_size: raise ValueError("Overlap must be smaller than chunk_size.")
        stride = int(VAE_STRIDE[0]);
        if chunk_size % stride != 0: raise ValueError(f"chunk_size must be multiple of {stride}.")
        
        # Global seeding for reproducibility of non-sampler operations
        seed32 = int(seed) & 0xFFFFFFFF
        torch.manual_seed(seed32)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed32)
        np.random.seed(seed32)

        vace_context_list = self._normalize_vace_context(vace_embeds)
        total_lat_frames = len(vace_context_list);
        if total_lat_frames == 0: raise ValueError("Empty vace_context.")

        num_frames = int(vace_embeds.get("num_frames", total_lat_frames * stride))
        expected_lat_frames = min(math.ceil(num_frames / stride), total_lat_frames)

        svd_model = model.clone();
        vae = svd_model.first_stage_model
        device, offload_device = mm.get_torch_device(), mm.unet_offload_device()
        v_dtype = getattr(vae, "dtype", torch.float32)

        C_lat, H_lat, W_lat = vace_context_list[0].shape[0], vace_context_list[0].shape[2], vace_context_list[0].shape[3]
        lat_overlap = max(0, overlap // stride)
        
        # Ensure model and VAE are on the correct device and in eval mode before the loop
        try:
            svd_model.to(device)
            vae.to(device)
            try:
                svd_model.model.to(device)
            except Exception:
                pass
            try:
                svd_model.first_stage_model.to(device)
            except Exception:
                pass
            svd_model.model.eval(); vae.eval()
        except Exception as e:
            log.warning(f"Could not move model to device or set eval mode: {e}")

        # Autocast/mixed-precision support if available in mm; fallback to nullcontext
        autocast_cm = nullcontext()
        try:
            autocast_fn = getattr(mm, "autocast_autoclear", None)
            if callable(autocast_fn):
                try:
                    autocast_cm = autocast_fn(device)
                except Exception:
                    autocast_cm = nullcontext()
        except Exception:
            autocast_cm = nullcontext()

        generated_chunks = collections.deque()
        frames_done_px = 0
        pbar = ProgressBar(num_frames or 1)
        
        framepack_context = self._prepare_initial_framepack(vae, initial_image, device, v_dtype, H_lat, W_lat, tiled_vae)

        with autocast_cm:
            with torch.no_grad():
                while frames_done_px < num_frames:
                    px_start, px_end = frames_done_px, min(frames_done_px + chunk_size, num_frames)
                    lat_start, lat_end = px_start // stride, math.ceil(px_end / stride)
                    this_lat_len = lat_end - lat_start
                    if this_lat_len <= 0: break

                    vace_slice = [t.to(device=device, dtype=v_dtype) for t in vace_context_list[lat_start:lat_end]]

                    # Construct a minimal positive conditioning dictionary
                    chunk_vace = {
                        "vace_context": vace_slice,
                        "vace_scale": vace_embeds.get("vace_scale", 1.0),
                    }
                    
                    cond_from_framepack = self._build_framepack_cond(svd_model, framepack_context, px_end - px_start, vace_embeds, device, v_dtype)
                    if cond_from_framepack is not None:
                        # Add conditioning under both keys for maximum compatibility
                        chunk_vace["cond"] = cond_from_framepack
                        chunk_vace["framepack_context"] = cond_from_framepack

                    gen = torch.Generator(device=device).manual_seed((int(seed) + frames_done_px) & 0xFFFFFFFF)
                    noise = torch.randn((1, C_lat, this_lat_len, H_lat, W_lat), device=device, dtype=v_dtype, generator=gen)

                    sampled = comfy_sample.sample(
                        model=svd_model, noise=noise, steps=int(steps), cfg=float(cfg),
                        sampler_name=sampler_name, scheduler=scheduler or "unipc", # Handle None scheduler
                        positive=chunk_vace, negative=None, latent_image=noise, denoise=1.0,
                    )

                    if sampled is None: raise RuntimeError("Sampler returned None.")
                    if sampled.dim() == 4: sampled = sampled.unsqueeze(2)

                    if generated_chunks and lat_overlap > 0:
                        prev = generated_chunks[-1].to(device, dtype=v_dtype)
                        eff_overlap = min(lat_overlap, prev.shape[2], sampled.shape[2])
                        if eff_overlap > 0:
                            tail = prev[:, :, -eff_overlap:, :, :]
                            head = sampled[:, :, :eff_overlap, :, :]
                            alpha = torch.linspace(0.0, 1.0, eff_overlap, device=device, dtype=v_dtype).view(1, 1, eff_overlap, 1, 1)
                            blended = (1.0 - alpha) * tail + alpha * head
                            sampled[:, :, :eff_overlap, :, :] = blended
                    
                    # Detach and move the finished chunk to CPU to keep GPU memory low
                    generated_chunks.append(sampled.detach().cpu().contiguous())

                    needed_lat_for_context = max(1, math.ceil(context_size / stride))
                    framepack_context = sampled[:, :, -needed_lat_for_context:, :, :].detach().clone().to(device=device, dtype=v_dtype)

                    frames_done_px += this_lat_len * stride
                    pbar.update(px_end - px_start)

                    if low_vram:
                        # Offload to reduce VRAM pressure then ensure reload even if offload raises
                        try:
                            try:
                                svd_model.model.to(offload_device)
                            except Exception:
                                pass
                            try:
                                vae.to(offload_device)
                            except Exception:
                                pass
                            mm.soft_empty_cache(); gc.collect()
                        finally:
                            # Best-effort reload to working device for next chunk
                            try:
                                svd_model.model.to(device)
                            except Exception:
                                pass
                            try:
                                vae.to(device)
                            except Exception:
                                pass

        if not generated_chunks: raise RuntimeError("No chunks were generated.")
        
        # Conditional final concatenation to respect low_vram mode
        if low_vram:
            # Keep final latents on CPU to avoid a large GPU allocation spike.
            final_latents = torch.cat(list(generated_chunks), dim=2).contiguous()  # stays on CPU
        else:
            final_latents = torch.cat([c.to(device) for c in generated_chunks], dim=2).contiguous()
            
        final_latents = final_latents[:, :, :expected_lat_frames, :, :]
        return ({"samples": final_latents},)

# NODE_CLASS_MAPPINGS["CustomVACEStyleTransfer"] = CustomVACEStyleTransfer
# NODE_DISPLAY_NAME_MAPPINGS["CustomVACEStyleTransfer"] = "Custom VACE Style Transfer Sampler"
