
# import math
# import copy
# import torch
# import torch.nn.functional as F
# import gc
# from typing import Optional, Dict
# from comfy import model_management as mm
# from comfy.utils import ProgressBar
 import logging

log = logging.getLogger(__name__)

# Import VACE encoder + stride globals 
try:
    from .nodes import WanVideoVACEEncode, VAE_STRIDE
except ImportError:
    VAE_STRIDE = (4, 8, 8)
    from comfy.model_management import device, offload_device


class CustomVACEStyleTransfer:
    """
    Custom VACE encoder node for long-video motion + style transfer.

    Purpose:
        Builds WANVIDIMAGE_EMBEDS dict (motion + appearance conditioning)
        from a reference video (motion) and style image/video (appearance).
        Handles extended videos (1 mins) via chunked encoding.

    Inputs:
        - ref_video: motion source (T,H,W,C)
        - style_image: appearance source (1,H,W,C) or (H,W,C)
        - chunk_size: frames per encode batch (16–64 typical)
        - pad_mode: how to align frame count (repeat_last | zeros | noise | truncate)
        - motion_weight, style_weight: scale appearance/motion influence
        - low_vram/tiled_vae: performance options
        - seed: ensures deterministic padding
        - use_framepack: enables tiled style for multi-view conditioning

    Output:
        WANVIDIMAGE_EMBEDS dict compatible with WanVideoSampler / KSampler.

    VRAM Tip:
        For 1-min @512x512, use chunk_size=16 + low_vram=True (~3.5GB peak).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE",),
                "ref_video": ("IMAGE", {"tooltip": "Reference video (T,H,W,C) providing motion."}),
                "style_image": ("IMAGE", {"tooltip": "Style image or single-frame video for appearance."}),
                "chunk_size": ("INT", {"default": 16, "min": 1, "max": 1024, "tooltip": "Frames per encode chunk."}),
                "pad_mode": (["repeat_last", "zeros", "noise", "truncate"], {"default": "repeat_last"}),
                "noise_scale": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "tooltip": "Noise intensity for padding."}),
                "motion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "style_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "low_vram": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ref_masks": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "RNG seed for noise padding."}),
                "use_framepack": ("BOOLEAN", {"default": False, "tooltip": "Enable framepack-style tiled style input."}),
                "framepack_cols": ("INT", {"default": 4, "min": 1, "max": 16}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    # Helpers
    def _to_tensor(self, x):
        return torch.from_numpy(x) if isinstance(x, (list, tuple)) else x

    def _norm_frames(self, frames, device, dtype):
        if frames.ndim != 4:
            raise ValueError("ref_video must be 4D (T,H,W,C)")
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        frames = frames.to(dtype=dtype, device=device)
        fmin, fmax = float(frames.min()), float(frames.max())
        if fmax > 1.0 and fmin >= 0.0:
            frames = frames / 255.0
        elif fmin < 0.0:
            frames = (frames + 1.0) / 2.0
        return frames.clamp(0.0, 1.0)

    def _norm_single_img(self, img, device, dtype):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.to(dtype=dtype, device=device)
        fmin, fmax = float(img.min()), float(img.max())
        if fmax > 1.0 and fmin >= 0.0:
            img = img / 255.0
        elif fmin < 0.0:
            img = (img + 1.0) / 2.0
        return img.clamp(0.0, 1.0)

    def _resize_style(self, img, H, W, device, dtype):
        if img.shape[1:3] != (H, W):
            img = F.interpolate(img.permute(0,3,1,2).to(device=device, dtype=dtype),
                                size=(H, W), mode='bilinear', align_corners=False).permute(0,2,3,1)
        return img.to(device=device, dtype=dtype)

    def _resize_mask(self, masks, H, W, device, dtype):
        if masks is None:
            return None
        if masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks.permute(0,3,1,2)
        elif masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = F.interpolate(masks.to(dtype=dtype, device=device), size=(H, W), mode="nearest")
        return masks.squeeze(1)

    def _pad_or_truncate_to_stride(self, frames, stride, pad_mode, noise_scale, seed, device, dtype):
        T = frames.shape[0]
        rem = T % stride
        if rem == 0:
            return frames, T
        if pad_mode == "truncate":
            return frames[:T - rem], T - rem
        pad_len = stride - rem
        if pad_mode == "repeat_last":
            pad = frames[-1:].repeat(pad_len, 1, 1, 1)
        elif pad_mode == "zeros":
            pad = torch.zeros((pad_len, frames.shape[1], frames.shape[2], 3), device=device, dtype=dtype)
        else:  # noise
            gen = torch.Generator(device=device).manual_seed(int(seed & 0xFFFFFFFF))
            pad = (0.5 + torch.randn((pad_len, frames.shape[1], frames.shape[2], 3),
                                     generator=gen, device=device, dtype=dtype) * noise_scale).clamp(0.0, 1.0)
        return torch.cat([frames, pad], dim=0), frames.shape[0] + pad_len

    def _pack_style_image(self, style_image, cols):
        cols = max(1, int(cols))
        rows = cols
        tiles = style_image.repeat(rows * cols, 1, 1, 1)
        rows_cat = [torch.cat(list(tiles[r*cols:(r+1)*cols]), dim=1) for r in range(rows)]
        return torch.cat(rows_cat, dim=0).unsqueeze(0)

    def process(self, vae, ref_video, style_image, chunk_size, pad_mode="repeat_last",
                noise_scale=0.05, motion_weight=1.0, style_weight=1.0,
                tiled_vae=False, low_vram=False, ref_masks=None, strength=1.0,
                vace_start_percent=0.0, vace_end_percent=1.0, prev_vace_embeds=None,
                seed=0, use_framepack=False, framepack_cols=4):

        if vae is None or ref_video is None:
            raise ValueError("vae and ref_video are required inputs.")

        vae_device = getattr(vae, "device", mm.get_torch_device())
        v_dtype = getattr(vae, "dtype", torch.float32)
        stride = int(VAE_STRIDE[0])

        ref_video = self._norm_frames(ref_video, vae_device, v_dtype) * motion_weight
        style_image = self._norm_single_img(style_image, vae_device, v_dtype) * style_weight
        ref_video, num_frames = self._pad_or_truncate_to_stride(ref_video, stride, pad_mode, noise_scale, seed, vae_device, v_dtype)
        H, W = ref_video.shape[1:3]
        ref_masks = self._resize_mask(ref_masks, H, W, vae_device, v_dtype)
        if use_framepack:
            style_image = self._pack_style_image(style_image, framepack_cols)

        if num_frames > 4096:
            log.warning(f"[CustomVACEStyleTransfer] Very long video ({num_frames} frames) may take long to process.")

        vace_embeds = copy.deepcopy(prev_vace_embeds) if prev_vace_embeds else {
            "vace_context": [],
            "vace_scale": 1.0,
            "has_ref": True,
            "target_shape": None,
            "vace_start_percent": float(vace_start_percent),
            "vace_end_percent": float(vace_end_percent),
            "num_frames": 0,
            "vace_seq_len": 0,
            "additional_vace_inputs": []
        }

        enc = WanVideoVACEEncode()
        encoded = 0
        pbar = ProgressBar(num_frames or 1)

        try:
            while encoded < num_frames:
                this_chunk = min(chunk_size, num_frames - encoded)
                chunk_seed = (seed + encoded) & 0xFFFFFFFF
                chunk_frames = ref_video[encoded:encoded + this_chunk]
                mask_chunk = ref_masks[encoded:encoded + this_chunk] if ref_masks is not None else None
                style_resized = self._resize_style(style_image, H, W, vae_device, v_dtype)

                chunk_out = enc.process(vae, W, H, this_chunk, float(strength),
                                        float(vace_start_percent), float(vace_end_percent),
                                        input_frames=chunk_frames, ref_images=style_resized,
                                        input_masks=mask_chunk, prev_vace_embeds=None,
                                        tiled_vae=tiled_vae)

                if not (isinstance(chunk_out, (list, tuple)) and len(chunk_out) > 0):
                    raise ValueError("WanVideoVACEEncode.process returned invalid output.")
                chunk_vace = chunk_out[0]
                chunk_context = chunk_vace.get("vace_context")

                # Extend list if encoder outputs list, else append tensor
                if isinstance(chunk_context, list):
                    vace_embeds["vace_context"].extend([ctx.detach().cpu() for ctx in chunk_context])
                elif isinstance(chunk_context, torch.Tensor):
                    vace_embeds["vace_context"].append(chunk_context.detach().cpu())

                vace_embeds["additional_vace_inputs"].append(chunk_vace)
                if vace_embeds["target_shape"] is None:
                    vace_embeds["target_shape"] = tuple(chunk_vace.get("target_shape", ()))
                encoded += this_chunk
                try:
                    pbar.update(this_chunk)
                except Exception:
                    pass

                if low_vram:
                    vae.to(offload_device)
                    mm.soft_empty_cache()
                    gc.collect()
                    if encoded < num_frames:
                        vae.to(vae_device)
        finally:
            vae.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        # Final metadata aggregation
        vace_embeds["num_frames"] += num_frames
        lat_h, lat_w = H // VAE_STRIDE[1], W // VAE_STRIDE[2]
        lat_t = (num_frames - 1) // VAE_STRIDE[0] + 1
        vace_embeds["target_shape"] = (16, lat_t, lat_h, lat_w)
        vace_embeds["vace_seq_len"] = len(vace_embeds["vace_context"])
        vace_embeds["has_ref"] = True

        if vace_embeds["vace_seq_len"] > 4096:
            log.warning(f"[CustomVACEStyleTransfer] Warning: seq_len={vace_embeds['vace_seq_len']} may exceed model context.")

        log.info(f"[CustomVACEStyleTransfer] Completed {vace_embeds['num_frames']} frames; seq_len={vace_embeds['vace_seq_len']}.")
        return (vace_embeds,)


