class WanVideoVACEExtender:
    """
    VACE extender — append extra VACE chunks to an existing WANVIDIMAGE_EMBEDS,
    or build the base VACE embeds from (vae + input_frames) and then append.

    Key points:
    - chunk_size operates on PIXEL frames. VAE_STRIDE[0] (temporal stride, usually 4)
      groups pixel -> latent frames; chunk_size MUST be a multiple of VAE_STRIDE[0].
    - This node intentionally rejects overlap != 0 to avoid latent blending pitfalls.
      Use WanVideoContextOptions for windowed context/overlap where appropriate.
    - If extend_frames == 0 and vace_embeds is provided, this node passes it through.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "extend_frames": ("INT", {"default": 16, "min": 0, "max": 8192, "tooltip": "Number of pixel frames to append."}),
                "chunk_size": ("INT", {"default": 16, "min": 1, "max": 1024, "tooltip": "Frames per encode chunk (must be multiple of VAE_STRIDE[0])."}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Overlap (pixel frames). Not supported for latent blending; must be 0."}),
                "strategy": (["repeat_last", "zeros", "noise"], {"default": "repeat_last", "tooltip": "How to fabricate pixel frames for extension."}),
                "noise_scale": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Noise strength for strategy='noise'."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Base RNG seed for deterministic chunk noise."}),
                "apply_global_seed": ("BOOLEAN", {"default": False, "tooltip": "If True, set torch.manual_seed(seed) at start for full-run reproducibility."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding (for large images)."}),
                "low_vram": ("BOOLEAN", {"default": False, "tooltip": "Aggressive offload between chunks to minimize peak VRAM."}),
                "truncate_base": ("BOOLEAN", {"default": False, "tooltip": "Truncate base frames to stride multiple if True; else pad with strategy."}),
            },
            "optional": {
                "vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "vae": ("WANVAE",),
                "input_frames": ("IMAGE",),
                "input_masks": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "mask_pad_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "Value to pad short masks with (0=masked, 1=unmasked)."}),
                "seq_max_warn": ("INT", {"default": 4096, "min": 1024, "max": 16384, "step": 1, "tooltip": "Warn if total seq_len exceeds this (model limit)."}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "extend"
    CATEGORY = "WanVideoWrapper"

    # Helpers
    def _normalize_input_frames(self, frames: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Accept and normalize frames to (T, H, W, 3) float32 in [0,1].
        Accepts None -> returns None.
        Handles common inputs:
          - uint8 [0,255]
          - float in [-1,1]
          - float in [0,1]
        """
        if frames is None:
            return None
        if not isinstance(frames, torch.Tensor):
            raise TypeError("input_frames must be a torch.Tensor with shape (T,H,W,C).")
        if frames.ndim != 4:
            raise ValueError("input_frames must be 4D tensor (T, H, W, C).")
        # Reduce RGBA -> RGB silently
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        frames = frames.to(torch.float32)
        fmin = float(frames.min())
        fmax = float(frames.max())
        # Improved heuristics
        if fmax > 1.0:
            if fmin >= 0.0:
                frames /= 255.0  # Assume uint8
            else:
                raise ValueError("Input range invalid: min<0 and max>1")
        elif fmin < 0.0:
            frames = (frames + 1.0) / 2.0  # Assume [-1,1]
        # Clamp always
        frames = frames.clamp(0.0, 1.0)
        return frames

    def _resize_frames(self, frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resize frames from (T, H_src, W_src, C) to (T, target_h, target_w, C) using bilinear interp.
        """
        if frames.shape[1:3] == (target_h, target_w):
            return frames
        log.info(f"Resizing frames from ({frames.shape[1]}, {frames.shape[2]}) -> ({target_h}, {target_w})")
        # Permute to (T, C, H_src, W_src)
        frames_chw = frames.permute(0, 3, 1, 2)
        resized = F.interpolate(frames_chw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1).contiguous()

    def _resize_mask(self, mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resize mask from (T, H_src, W_src) or (T, H_src, W_src, 1) to (T, target_h, target_w).
        Uses nearest interpolation.
        """
        orig_ndim = mask.ndim
        if mask.shape[-2:] == (target_h, target_w):
            if orig_ndim == 3:
                return mask
            else:
                return mask[..., 0]
        log.info(f"Resizing masks from ({mask.shape[-2]}, {mask.shape[-1]}) -> ({target_h}, {target_w})")
        # Ensure (T, 1, H, W) for interpolate
        if orig_ndim == 3:
            mask4d = mask.unsqueeze(-1)  # (T, H, W, 1)
        else:
            mask4d = mask  # Assume (T, H, W, 1)
        # Permute to (T, 1, H, W)
        mask_interp = F.interpolate(mask4d.permute(0, 3, 1, 2), size=(target_h, target_w), mode='nearest').permute(0, 2, 3, 1)
        # Return as (T, H, W)
        return mask_interp.squeeze(-1)

    def _make_chunk_pixels(self, strategy: str, chunk_len: int, input_frames: Optional[torch.Tensor], height: int, width: int, noise_scale: float, seed: int) -> torch.Tensor:
        """
        Build pixel frames for a chunk on CPU, dtype float32, shape (chunk_len, H, W, 3), values in [0,1].
        """
        if strategy == "zeros":
            return torch.zeros((chunk_len, height, width, 3), dtype=torch.float32, device="cpu")

        if strategy == "repeat_last" and input_frames is not None:
            last = input_frames.clone()[-1]
            if last.shape[-1] == 4:
                last = last[..., :3]
            last = last.to(torch.float32).clamp(0.0, 1.0)
            out = last.unsqueeze(0).repeat(chunk_len, 1, 1, 1).contiguous()
            return out.to("cpu")

        # noise:
        gen = torch.Generator(device="cpu")
        derived = int(seed & 0xFFFFFFFF)
        gen.manual_seed(derived)
        base = torch.rand((chunk_len, height, width, 3), generator=gen, dtype=torch.float32, device="cpu")
        g = torch.randn((chunk_len, height, width, 3), generator=gen, dtype=torch.float32, device="cpu") * float(noise_scale)
        noise = base + g
        noise = torch.clamp(noise, 0.0, 1.0)
        return noise

    def _slice_mask_for_chunk(self, input_masks: Optional[torch.Tensor], start_frame: int, chunk_len: int, pad_value: float = 1.0) -> Optional[torch.Tensor]:
        """
        Slice or pad masks for the chunk. Accepts None.
        Expected input_masks shape: (T, H, W) or (T, H, W, 1)
        Returns mask chunk of shape (chunk_len, H, W) or None.
        """
        if input_masks is None:
            return None
        if not isinstance(input_masks, torch.Tensor):
            raise TypeError("input_masks must be a torch.Tensor.")
        total_available = input_masks.shape[0]
        end_frame = start_frame + chunk_len
        if total_available < end_frame:
            # pad with pad_value to avoid silent failures
            pad_len = end_frame - total_available
            log.warning(f"WanVideoVACEExtender: mask too short ({total_available} < {end_frame}); padding {pad_len} frames with {pad_value}.")
            if input_masks.ndim == 4 and input_masks.shape[-1] == 1:
                pad = torch.full((pad_len, input_masks.shape[1], input_masks.shape[2], 1), pad_value, dtype=input_masks.dtype, device=input_masks.device)
                input_masks = torch.cat([input_masks, pad], dim=0)
                mask = input_masks[start_frame:end_frame, ..., 0]
            else:
                pad = torch.full((pad_len, input_masks.shape[1], input_masks.shape[2]), pad_value, dtype=input_masks.dtype, device=input_masks.device)
                input_masks = torch.cat([input_masks, pad], dim=0)
                mask = input_masks[start_frame:end_frame]
        else:
            if input_masks.ndim == 4 and input_masks.shape[-1] == 1:
                mask = input_masks[start_frame:end_frame, ..., 0]
            else:
                mask = input_masks[start_frame:end_frame]
        return mask

    def _pad_base_masks(self, input_masks: Optional[torch.Tensor], num_frames: int, pad_value: float = 1.0) -> Optional[torch.Tensor]:
        """
        Pad base masks to num_frames with pad_value if too short.
        """
        if input_masks is None:
            return None
        if input_masks.shape[0] < num_frames:
            log.warning(f"WanVideoVACEExtender: Padding base masks from {input_masks.shape[0]} to {num_frames}")
            pad_len = num_frames - input_masks.shape[0]
            if input_masks.ndim == 4 and input_masks.shape[-1] == 1:
                pad = torch.full((pad_len, *input_masks.shape[1:-1], 1), pad_value, dtype=input_masks.dtype, device=input_masks.device)
                input_masks = torch.cat([input_masks, pad], dim=0)[:, ..., 0]
            else:
                pad = torch.full((pad_len, *input_masks.shape[1:]), pad_value, dtype=input_masks.dtype, device=input_masks.device)
                input_masks = torch.cat([input_masks, pad], dim=0)
        return input_masks

    # Main function
    def extend(self,
               extend_frames: int,
               chunk_size: int,
               overlap: int,
               strategy: str,
               noise_scale: float,
               seed: int,
               apply_global_seed: bool,
               tiled_vae: bool,
               low_vram: bool,
               truncate_base: bool,
               vace_embeds=None,
               vae=None,
               input_frames=None,
               input_masks=None,
               strength: float = 1.0,
               vace_start_percent: float = 0.0,
               vace_end_percent: float = 1.0,
               mask_pad_value: float = 1.0,
               seq_max_warn: int = 4096):
        """
        extend_frames: number of pixel frames to append
        chunk_size: frames per encode chunk (must be multiple of VAE_STRIDE[0])
        overlap: must be 0 (latent blending not supported by this node)
        """

        # Basic validation
        extend_frames = int(extend_frames)
        chunk_size = int(chunk_size)
        overlap = int(overlap)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be >= 1")
        if extend_frames < 0:
            raise ValueError("extend_frames must be >= 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap != 0:
            raise ValueError("overlap != 0 is not supported. Use WanVideoContextOptions for latent-aware context overlap/windowing.")

        # Enforce pixel -> latent stride alignment
        stride = int(VAE_STRIDE[0])
        if chunk_size % stride != 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of VAE_STRIDE[0] ({stride}).")

        # Pass-through shortcut
        if extend_frames == 0 and vace_embeds is not None:
            log.info("WanVideoVACEExtender: extend_frames==0 and vace_embeds provided — passing through.")
            return (vace_embeds,)

        # If extension required, ensure VAE exists
        if extend_frames > 0 and vae is None:
            raise ValueError("VAE is required to create additional VACE chunks (pass your WANVAE).")

        # Apply optional global seed for full-run determinism (still derive per-chunk seeds)
        base_seed = int(seed & 0xFFFFFFFF)
        if apply_global_seed:
            torch.manual_seed(base_seed)
            log.info(f"WanVideoVACEExtender: applied global torch.manual_seed({base_seed})")

        # Normalize inputs
        input_frames = self._normalize_input_frames(input_frames) if input_frames is not None else None
        if input_masks is not None and not isinstance(input_masks, torch.Tensor):
            raise TypeError("input_masks must be a torch.Tensor if provided.")

        # Build base vace_embeds if not provided
        if vace_embeds is None:
            if input_frames is None:
                raise ValueError("When vace_embeds is not provided, input_frames must be provided to build base embeds.")
            num_frames = int(input_frames.shape[0])
            # ensure base frames length aligned to stride
            remainder = num_frames % stride
            if remainder != 0:
                if truncate_base:
                    new_num = (num_frames // stride) * stride
                    if new_num == 0:
                        raise ValueError(f"input_frames length ({num_frames}) too small for VAE stride {stride}. Provide more frames.")
                    log.warning(f"WanVideoVACEExtender: truncating input_frames from {num_frames} -> {new_num} frames to align with stride {stride}.")
                    input_frames = input_frames[:new_num]
                    num_frames = new_num
                else:
                    pad_len = stride - remainder
                    log.info(f"WanVideoVACEExtender: padding base input_frames by {pad_len} frames with strategy '{strategy}' to align with stride {stride}.")
                    pad_frames = self._make_chunk_pixels(strategy, pad_len, input_frames, input_frames.shape[1], input_frames.shape[2], noise_scale, base_seed)
                    input_frames = torch.cat([input_frames, pad_frames], dim=0)
                    num_frames += pad_len

            # Pad masks for base if needed
            input_masks = self._pad_base_masks(input_masks, num_frames, mask_pad_value)

            H = int(input_frames.shape[1])
            W = int(input_frames.shape[2])
            enc = WanVideoVACEEncode()
            log.info("WanVideoVACEExtender: building base vace_embeds from input_frames")
            vace_out = enc.process(vae, W, H, num_frames, float(strength), float(vace_start_percent), float(vace_end_percent),
                                   input_frames=input_frames, ref_images=None, input_masks=input_masks, prev_vace_embeds=None, tiled_vae=tiled_vae)
            vace_embeds = vace_out[0]

        if not isinstance(vace_embeds, dict):
            raise ValueError("vace_embeds must be a dict (WANVIDIMAGE_EMBEDS).")

        # Target pixel resolution from target_shape in base embeds
        target_shape = vace_embeds.get("target_shape")
        if target_shape is None:
            raise ValueError("vace_embeds missing 'target_shape'; cannot infer width/height for encoding.")
        lat_h = int(target_shape[2])
        lat_w = int(target_shape[3])
        H = lat_h * VAE_STRIDE[1]
        W = lat_w * VAE_STRIDE[2]

        if input_frames is not None:
            input_frames = self._resize_frames(input_frames, H, W)
        if input_masks is not None:
            input_masks = self._resize_mask(input_masks, H, W)

        if "additional_vace_inputs" not in vace_embeds or not isinstance(vace_embeds["additional_vace_inputs"], list):
            vace_embeds["additional_vace_inputs"] = []

        enc = WanVideoVACEEncode()
        total_to_make = int(extend_frames)
        made = 0
        pbar = ProgressBar(total_to_make if total_to_make > 0 else 1)
        input_for_repeat = input_frames.clone() if input_frames is not None else None

        if vae is not None:
            vae = vae.to(device)
            v_dtype = getattr(vae, "dtype", torch.float32)
        else:
            v_dtype = torch.float32

        try:
            log.info(f"WanVideoVACEExtender: starting creation of {total_to_make} frames (chunk_size={chunk_size}, strategy={strategy})")
            while made < total_to_make:
                this_chunk = min(chunk_size, total_to_make - made)

                chunk_seed = hash((base_seed, made)) & 0xFFFFFFFF
                chunk_pixels = self._make_chunk_pixels(strategy, this_chunk, input_for_repeat, H, W, noise_scale, chunk_seed)

                if not low_vram:
                    chunk_pixels = chunk_pixels.to(device, non_blocking=True)

                mask_chunk = self._slice_mask_for_chunk(input_masks, made, this_chunk, mask_pad_value) if input_masks is not None else None
                if mask_chunk is not None and not low_vram:
                    mask_chunk = mask_chunk.to(device, non_blocking=True)

                # Encode the chunk using VACE encoder
                try:
                    new_vace_tuple = enc.process(vae, W, H, this_chunk, float(strength), float(vace_start_percent), float(vace_end_percent),
                                                 input_frames=chunk_pixels, ref_images=None, input_masks=mask_chunk, prev_vace_embeds=None, tiled_vae=tiled_vae)
                except Exception as e:
                    log.error(f"WanVideoVACEExtender: VACE encoding failed at offset {made}: {e}")
                    raise

                if not (isinstance(new_vace_tuple, (list, tuple)) and len(new_vace_tuple) > 0):
                    raise ValueError("WanVideoVACEEncode.process returned unexpected structure.")
                new_vace = new_vace_tuple[0]
                if not isinstance(new_vace, dict):
                    raise TypeError(f"WanVideoVACEExtender: expected dict from encoder, got {type(new_vace)}")

                # Validate shape compatibility: check target_shape equality
                new_target = new_vace.get("target_shape")
                if new_target is None:
                    raise ValueError("Encoded vace chunk missing 'target_shape'.")
                if tuple(new_target) != tuple(vace_embeds.get("target_shape")):
                    raise ValueError(f"Encoded chunk target_shape {new_target} != base target_shape {vace_embeds.get('target_shape')}")

                vace_embeds["additional_vace_inputs"].append(new_vace)

                made += this_chunk
                pbar.update(this_chunk)

                if low_vram and vae is not None:
                    try:
                        vae.to(offload_device)
                    except Exception:
                        pass
                    mm.soft_empty_cache()
                    gc.collect()
                    if made < total_to_make:
                        vae.to(device)
                del chunk_pixels  # Explicit del for GC
                if mask_chunk is not None:
                    del mask_chunk

        finally:
            # Ensure VAE offloaded and caches freed
            if vae is not None:
                try:
                    vae.to(offload_device)
                except Exception:
                    pass
            mm.soft_empty_cache()
            gc.collect()

        # Update metadata
        previous_num = int(vace_embeds.get("num_frames", 0))
        vace_embeds["num_frames"] = previous_num + total_to_make

        base_seq = vace_embeds.get("vace_seq_len", 0)
        add_seq = sum(c.get("vace_seq_len", 0) for c in vace_embeds["additional_vace_inputs"])
        total_seq = base_seq + add_seq
        if total_seq > seq_max_warn:
            log.warning(f"WanVideoVACEExtender: Total seq_len {total_seq} may exceed model limits (~{seq_max_warn}). Consider shorter chunks or context options.")

        log.info(f"WanVideoVACEExtender: appended {total_to_make} frames — new num_frames={vace_embeds['num_frames']}, total_seq_len={total_seq}")

        return (vace_embeds,)