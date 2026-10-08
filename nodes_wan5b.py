"""
ComfyUI nodes for Wan 2.2 TI2V-5B keyframe control.

Designed specifically for the Wan 2.2 5B model (48-channel Wan2.2 VAE).
Includes a soft end-frame landing to reduce the common "last frames corrupted" artifact.
"""

import torch
import nodes
import comfy.utils
import comfy.model_management
import comfy.latent_formats


def _ensure_4n1(length: int) -> int:
    """Force length onto the 4n+1 grid required by Wan temporal VAE."""
    if length < 1:
        return 1
    return ((length - 1) // 4) * 4 + 1


def _soft_end_mask(num_latent_frames: int, soft_frames: int, end_strength: float, device) -> torch.Tensor:
    """
    Build a temporal mask for the end keyframe.

    mask == 0  -> fully locked to the keyframe latent
    mask == 1  -> free generation (noise)

    Instead of hard-locking only the very last latent slot (which causes
    the classic dirty last frames), we ramp the lock strength over the
    last `soft_frames` latent slots so the model can approach the end
    frame smoothly.
    """
    mask = torch.ones([1, 1, num_latent_frames, 1, 1], device=device)

    if soft_frames <= 0 or end_strength <= 0:
        return mask

    soft_frames = min(soft_frames, num_latent_frames)

    # Linear ramp from free -> locked over the last soft_frames slots
    # At the final slot the mask is (1 - end_strength)
    for i in range(soft_frames):
        t = (i + 1) / soft_frames  # 0..1 toward the end
        lock = end_strength * t
        idx = num_latent_frames - soft_frames + i
        mask[:, :, idx, :, :] = 1.0 - lock

    return mask


class Wan5B_FirstLastFrameToLatent:
    """
    First + Last (optional Middle) frame conditioning for Wan 2.2 TI2V-5B.

    Key improvement over previous 5B FLF nodes:
    - Soft end-frame landing (ramp) instead of hard lock on the last latent
      frames. This greatly reduces the common "last few frames are corrupted"
      / noisy / flashing artifact.
    - Independent strength controls for start / middle / end.
    - Correct 48-channel Wan2.2 latent handling.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": nodes.MAX_RESOLUTION,
                    "step": 4,
                    "tooltip": "Total frames. Must be 4n+1 (81, 121, …). Value is auto-snapped if needed."
                }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "start_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly the start frame is locked (1.0 = hard lock)."
                }),
                "end_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly the end frame is locked. Slightly < 1.0 often looks cleaner."
                }),
                "soft_end_frames": ("INT", {
                    "default": 3, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of latent frames over which the end keyframe is ramped in. "
                               "Higher = smoother approach, less risk of last-frame corruption."
                }),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "middle_strength": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Strength of the optional middle keyframe."
                }),
                "middle_position": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "Where the middle keyframe sits in the timeline (0=start, 1=end)."
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "Wan5B/keyframe"
    DESCRIPTION = (
        "First/Last/Middle frame conditioning for Wan 2.2 TI2V-5B with soft end-frame landing "
        "to reduce last-frame corruption."
    )

    def encode(
        self,
        vae,
        width,
        height,
        length,
        batch_size,
        start_strength=1.0,
        end_strength=0.85,
        soft_end_frames=3,
        start_image=None,
        end_image=None,
        middle_image=None,
        middle_strength=0.75,
        middle_position=0.5,
    ):
        length = _ensure_4n1(length)
        device = comfy.model_management.intermediate_device()

        # Wan2.2 5B: 48 channels, spatial /16, temporal ((L-1)//4)+1
        num_latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [1, 48, num_latent_t, height // 16, width // 16],
            device=device,
        )

        # mask = 1 → free generation, mask = 0 → locked to keyframe
        mask = torch.ones(
            [1, 1, num_latent_t, latent.shape[-2], latent.shape[-1]],
            device=device,
        )

        def _prep_image(img):
            # Resize to target resolution, keep channel-last for vae.encode
            return comfy.utils.common_upscale(
                img[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        # ---- Start frame ----
        if start_image is not None and start_strength > 0:
            start_image = _prep_image(start_image)
            start_lat = vae.encode(start_image)  # [1, 48, t_start, h, w]
            t_s = start_lat.shape[2]
            latent[:, :, :t_s] = start_lat
            # Hard-ish lock on the first latent frame(s)
            mask[:, :, :t_s] = 1.0 - start_strength

        # ---- Middle frame (optional) ----
        if middle_image is not None and middle_strength > 0:
            middle_image = _prep_image(middle_image)
            mid_lat = vae.encode(middle_image)
            t_m = mid_lat.shape[2]
            # Place around the requested relative position
            center = int(round(middle_position * (num_latent_t - 1)))
            start_idx = max(0, min(center - t_m // 2, num_latent_t - t_m))
            latent[:, :, start_idx : start_idx + t_m] = mid_lat
            mask[:, :, start_idx : start_idx + t_m] = 1.0 - middle_strength

        # ---- End frame with soft landing ----
        if end_image is not None and end_strength > 0:
            end_image = _prep_image(end_image)
            end_lat = vae.encode(end_image)
            t_e = end_lat.shape[2]

            # Soft temporal mask for the end region
            soft_mask = _soft_end_mask(
                num_latent_frames=num_latent_t,
                soft_frames=soft_end_frames,
                end_strength=end_strength,
                device=device,
            )
            # Expand spatial dims
            soft_mask = soft_mask.expand(-1, -1, -1, latent.shape[-2], latent.shape[-1])

            # Write the end latent into the final slots
            latent[:, :, -t_e:] = end_lat

            # Combine: where soft_mask is lower, we lock more strongly
            # Keep any stronger lock that may already exist from middle frame
            mask[:, :, -t_e:] = torch.minimum(mask[:, :, -t_e:], soft_mask[:, :, -t_e:])

        # Apply Wan22 latent format scaling only on the free (noisy) regions
        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        processed = latent_format.process_out(latent)
        # Locked regions keep the raw encoded latent; free regions get the scaled version
        final_latent = processed * mask + latent * (1.0 - mask)

        out_latent["samples"] = final_latent.repeat((batch_size,) + (1,) * (final_latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size,) + (1,) * (mask.ndim - 1))
        return (out_latent,)


class Wan5B_FirstLastFrameToLatent_Tiled:
    """Same as Wan5B_FirstLastFrameToLatent but uses tiled VAE encode to save VRAM."""

    @classmethod
    def INPUT_TYPES(s):
        base = Wan5B_FirstLastFrameToLatent.INPUT_TYPES()
        base["required"].update({
            "tile_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
            "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
            "temporal_size": ("INT", {
                "default": 64, "min": 8, "max": 4096, "step": 4,
                "tooltip": "Frames encoded at once in the temporal dimension."
            }),
            "temporal_overlap": ("INT", {
                "default": 8, "min": 4, "max": 4096, "step": 4,
                "tooltip": "Temporal overlap for tiling."
            }),
        })
        return base

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "Wan5B/keyframe"
    DESCRIPTION = "Tiled-VAE version of Wan5B First/Last Frame (lower VRAM)."

    def encode(
        self,
        vae,
        width,
        height,
        length,
        batch_size,
        start_strength=1.0,
        end_strength=0.85,
        soft_end_frames=3,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        start_image=None,
        end_image=None,
        middle_image=None,
        middle_strength=0.75,
        middle_position=0.5,
    ):
        length = _ensure_4n1(length)
        device = comfy.model_management.intermediate_device()

        num_latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [1, 48, num_latent_t, height // 16, width // 16],
            device=device,
        )
        mask = torch.ones(
            [1, 1, num_latent_t, latent.shape[-2], latent.shape[-1]],
            device=device,
        )

        def _prep_image(img):
            return comfy.utils.common_upscale(
                img[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        def _tiled_encode(img):
            return vae.encode_tiled(
                img,
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )

        if start_image is not None and start_strength > 0:
            start_image = _prep_image(start_image)
            start_lat = _tiled_encode(start_image)
            t_s = start_lat.shape[2]
            latent[:, :, :t_s] = start_lat
            mask[:, :, :t_s] = 1.0 - start_strength

        if middle_image is not None and middle_strength > 0:
            middle_image = _prep_image(middle_image)
            mid_lat = _tiled_encode(middle_image)
            t_m = mid_lat.shape[2]
            center = int(round(middle_position * (num_latent_t - 1)))
            start_idx = max(0, min(center - t_m // 2, num_latent_t - t_m))
            latent[:, :, start_idx : start_idx + t_m] = mid_lat
            mask[:, :, start_idx : start_idx + t_m] = 1.0 - middle_strength

        if end_image is not None and end_strength > 0:
            end_image = _prep_image(end_image)
            end_lat = _tiled_encode(end_image)
            t_e = end_lat.shape[2]

            soft_mask = _soft_end_mask(
                num_latent_frames=num_latent_t,
                soft_frames=soft_end_frames,
                end_strength=end_strength,
                device=device,
            ).expand(-1, -1, -1, latent.shape[-2], latent.shape[-1])

            latent[:, :, -t_e:] = end_lat
            mask[:, :, -t_e:] = torch.minimum(mask[:, :, -t_e:], soft_mask[:, :, -t_e:])

        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        processed = latent_format.process_out(latent)
        final_latent = processed * mask + latent * (1.0 - mask)

        out_latent["samples"] = final_latent.repeat((batch_size,) + (1,) * (final_latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size,) + (1,) * (mask.ndim - 1))
        return (out_latent,)


NODE_CLASS_MAPPINGS = {
    "Wan5B_FirstLastFrameToLatent": Wan5B_FirstLastFrameToLatent,
    "Wan5B_FirstLastFrameToLatent_Tiled": Wan5B_FirstLastFrameToLatent_Tiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan5B_FirstLastFrameToLatent": "Wan5B First/Last Frame to Latent",
    "Wan5B_FirstLastFrameToLatent_Tiled": "Wan5B First/Last Frame to Latent (Tiled VAE)",
}
