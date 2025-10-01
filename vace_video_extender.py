# Add required imports at top of nodes.py if not already present 
# import comfy
# import comfy.sample
# import comfy.utils
# import collections
# try:
#     import lora
# except Exception:
#     lora = None

class VACEVideoExtender:
    """
    Generates long videos using latent chaining and blending with a low VRAM mode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "initial_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (scheduler_list,),
                "total_frames": ("INT", {"default": 42, "min": 1, "max": 8192}),
                "chunk_size": ("INT", {"default": 14, "min": 1, "max": 200}),
                "context_size": ("INT", {"default": 3, "min": 1, "max": 64}),
                "overlap": ("INT", {"default": 4, "min": 0, "max": 64}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                "low_vram": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "tooltip": ("STRING", {"forceInput": True, "default": "Long video generator using latent chaining and blending."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_long_video"
    CATEGORY = "WanVideoWrapper"

    def _encode_image(self, vae, image, device, dtype):
        # Expect IMAGE type: (B,H,W,C)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.permute(0, 3, 1, 2).to(device, dtype=dtype)  # B,C,H,W
        samples = vae.encode(image, device=device, dtype=dtype,
                             tiled=False, encode_batch_size=1)
        # output: (C,T,H,W) convention
        if samples.dim() == 4:  # B,C,H,W
            samples = samples.permute(1, 0, 2, 3)
        return samples

    def _decode_latents(self, vae, latents, device, dtype):
        decoded = vae.decode(latents.to(device, dtype=dtype),
                             device=device, dtype=dtype,
                             tiled=False, half=False)
        if decoded.dim() == 4:  # B,C,H,W
            decoded = decoded.permute(1, 0, 2, 3)
        return decoded

    def _generate_chunk(self, svd_model, context_latents, chunk_size, seed, steps,
                        cfg, sampler_name, scheduler, motion_bucket_id, fps, device, dtype):
        value_and_mask = svd_model.model.conditioner.embedder.standard_embedder.get_value(
            chunk_size, context_latents.shape[2], context_latents.shape[3], fps, motion_bucket_id, device
        )
        cond = svd_model.model.conditioner.embedder(context_latents.to(device, dtype=dtype), value_and_mask)

        # Noise: (C,T,H,W)
        C, _, H, W = context_latents.shape
        gen = torch.Generator(device=device).manual_seed(int(seed & 0xFFFFFFFFFFFFFFFF))
        noise = torch.randn((C, chunk_size, H, W), device=device, dtype=dtype, generator=gen)

        samples = comfy.sample.sample(svd_model, noise, steps, cfg, sampler_name,
                                      scheduler, cond, None, noise, denoise=1.0)
        if samples.dim() == 4:  # B,C,H,W
            samples = samples.permute(1, 0, 2, 3)
        return samples

    def generate_long_video(self, model, initial_image, seed, steps, cfg, sampler_name,
                            scheduler, total_frames, chunk_size, context_size, overlap,
                            motion_bucket_id, fps, low_vram, tooltip=None):

        svd_model = model.clone()
        vae = svd_model.first_stage_model
        device = svd_model.device
        dtype = getattr(vae, "dtype", torch.float32)

        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size}).")

        from comfy.utils import ProgressBar
        pbar = ProgressBar(total_frames)

        initial_latent = self._encode_image(vae, initial_image, device, dtype)
        frames_generated = initial_latent.shape[1]
        all_chunks = collections.deque([initial_latent.cpu() if low_vram else initial_latent])

        while frames_generated < total_frames:
            if len(all_chunks) > 1 and context_size > all_chunks[-1].shape[1]:
                context = torch.cat(list(all_chunks)[-2:], dim=1)
            else:
                context = all_chunks[-1]
            context_latents = context[:, -context_size:, :, :]

            chunk_seed = (seed + frames_generated) & 0xFFFFFFFFFFFFFFFF
            new_chunk = self._generate_chunk(svd_model, context_latents, chunk_size,
                                             chunk_seed, steps, cfg, sampler_name,
                                             scheduler, motion_bucket_id, fps, device, dtype)

            if overlap > 0 and len(all_chunks) > 0:
                last = all_chunks[-1].to(device, dtype=dtype)
                effective = min(overlap, last.shape[1], new_chunk.shape[1])
                blend_a = last[:, -effective:, :, :]
                blend_b = new_chunk[:, :effective, :, :]
                alpha = torch.linspace(0, 1, effective, device=device, dtype=dtype).view(1, effective, 1, 1)
                blended = (1 - alpha) * blend_a + alpha * blend_b
                all_chunks[-1] = last[:, :-effective, :, :].cpu() if low_vram else last[:, :-effective, :, :]
                processed = torch.cat([blended, new_chunk[:, effective:, :, :]], dim=1)
            else:
                processed = new_chunk

            all_chunks.append(processed.cpu() if low_vram else processed)
            prev = frames_generated
            frames_generated = sum(c.shape[1] for c in all_chunks)
            pbar.update(frames_generated - prev)

        if low_vram:
            decoded_list = []
            for c in all_chunks:
                dec = self._decode_latents(vae, c, device, dtype)
                decoded_list.append(dec.cpu())
                torch.cuda.empty_cache()
            decoded = torch.cat(decoded_list, dim=1)
        else:
            all_cat = torch.cat([c.to(device, dtype=dtype) for c in all_chunks], dim=1)
            decoded = self._decode_latents(vae, all_cat, device, dtype)

        decoded = decoded.clamp(0, 1).cpu()
        out = decoded.permute(1, 2, 3, 0)[:total_frames]
        return (out,)
