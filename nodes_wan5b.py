"""
ComfyUI nodes for Wan 2.2 TI2V-5B keyframe control (v2).

Changes vs v1:
  1. Removed the process_out blend. The sampler applies process_latent_in itself,
     so we hand it the raw VAE latent (same as core Wan22ImageToVideoLatent).
  2. Middle / end keyframes are encoded as a static 5-frame clip and the LAST
     latent slot is taken, so they match the "4 pixel frames per slot" semantics
     of the causal VAE. Only the start frame uses single-frame (first-slot) encoding.
  3. End padding: the end keyframe is placed `end_padding_latents` slots before
     the true end. Free slots after it are trimmed after decode, which avoids the
     badly decoded final latent slots.
  4. Optional ramp: neighbouring slots before a keyframe are filled with the
     keyframe latent at decreasing lock strength (a real soft landing).
  5. Release node: run pass 1 with locks, then release middle/end locks for a
     second, lower-denoise pass so motion flows through the keyframes.
  6. Single encode path for all keyframes (tiled optional), Lanczos/area resize,
     alpha dropped, width/height snapped to multiples of 32.

Mask convention (ComfyUI noise_mask): 1.0 = free / denoised, 0.0 = locked.
"""

import torch
import nodes
import comfy.utils
import comfy.model_management


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ensure_4n1(length: int) -> int:
    if length < 1:
        return 1
    return ((length - 1) // 4) * 4 + 1


def _snap32(v: int) -> int:
    return max(32, (v // 32) * 32)


def _prep_image(img, width, height):
    """First frame only, RGB only, resized with a method suited to the direction."""
    img = img[:1, :, :, :3]
    src_h, src_w = img.shape[1], img.shape[2]
    method = "area" if (src_w > width or src_h > height) else "lanczos"
    out = comfy.utils.common_upscale(
        img.movedim(-1, 1), width, height, method, "center"
    ).movedim(1, -1)
    return out.clamp(0.0, 1.0)


def _encode(vae, img, tiled, tile_size, overlap, temporal_size, temporal_overlap):
    if tiled:
        return vae.encode_tiled(
            img,
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=overlap,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
    return vae.encode(img)


def _encode_first_slot(vae, img, **kw):
    """Single frame -> first-slot latent [1, 48, 1, h, w]. Start frame only."""
    return _encode(vae, img, **kw)[:, :, :1]


def _encode_body_slot(vae, img, **kw):
    """
    Encode a static 5-frame clip and keep the last latent slot, which is what a
    normal (4-pixel-frame) slot looks like to the model.
    """
    clip = img.repeat(5, 1, 1, 1)
    return _encode(vae, clip, **kw)[:, :, -1:]


def _place(latent, mask, filled, idx, lat, strength, ramp, min_idx=1):
    """Write keyframe latent at idx with lock (1-strength) and optional pre-ramp."""
    lat = lat.to(latent)
    latent[:, :, idx] = lat[:, :, 0]
    mask[:, :, idx].clamp_(max=1.0 - strength)
    filled[idx] = True

    for r in range(1, ramp + 1):
        j = idx - r
        if j < min_idx or filled[j]:
            break
        lock = strength * (1.0 - r / (ramp + 1))
        latent[:, :, j] = lat[:, :, 0]
        mask[:, :, j].clamp_(max=1.0 - lock)
        filled[j] = True


# --------------------------------------------------------------------------- #
# main node
# --------------------------------------------------------------------------- #
class Wan5B_KeyframeToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 1280, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 704, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 121, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4,
                    "tooltip": "Frames you want in the FINAL video (4n+1). Padding is added on top and trimmed later."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "middle_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Lower = looser pass-through. 0.3-0.6 is a good range."}),
                "end_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_padding_latents": ("INT", {
                    "default": 2, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Free latent slots after the end keyframe (4 frames each). "
                               "Trimmed afterwards with 'Wan5B Trim Frames'. 2-3 fixes dirty last frames."}),
                "ramp_frames": ("INT", {
                    "default": 0, "min": 0, "max": 6, "step": 1,
                    "tooltip": "Experimental. Slots BEFORE each middle/end keyframe filled with that "
                               "keyframe at decreasing lock. 0 = off."}),
                "tiled": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_position": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "Position on the (unpadded) timeline, 0=start, 1=end."}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "output_frames")
    FUNCTION = "encode"
    CATEGORY = "Wan5B/keyframe"
    DESCRIPTION = (
        "Start/middle/end keyframes for Wan 2.2 TI2V-5B with correct slot encoding, "
        "end padding and optional ramps. Feed output_frames to Wan5B Trim Frames after decode."
    )

    def encode(
        self, vae, width, height, length, batch_size,
        start_strength, middle_strength, end_strength,
        end_padding_latents, ramp_frames, tiled,
        start_image=None, middle_image=None, end_image=None,
        middle_position=0.5,
        tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
    ):
        width, height = _snap32(width), _snap32(height)
        length = _ensure_4n1(length)

        base_t = ((length - 1) // 4) + 1                 # slots for the final video
        pad = end_padding_latents if end_image is not None else 0
        total_t = base_t + pad                            # slots actually sampled
        end_idx = total_t - 1 - pad                       # == base_t - 1

        device = comfy.model_management.intermediate_device()
        lh, lw = height // 16, width // 16
        latent = torch.zeros([1, 48, total_t, lh, lw], device=device)
        mask = torch.ones([1, 1, total_t, lh, lw], device=device)
        filled = [False] * total_t

        enc_kw = dict(tiled=tiled, tile_size=tile_size, overlap=overlap,
                      temporal_size=temporal_size, temporal_overlap=temporal_overlap)
        meta = {"start": [], "middle": [], "end": []}

        # ---- start (slot 0, first-frame semantics) ----
        if start_image is not None and start_strength > 0:
            img = _prep_image(start_image, width, height)
            lat = _encode_first_slot(vae, img, **enc_kw).to(latent)
            latent[:, :, 0] = lat[:, :, 0]
            mask[:, :, 0] = 1.0 - start_strength
            filled[0] = True
            meta["start"].append(0)

        # ---- end (body-slot semantics, placed before the padding) ----
        if end_image is not None and end_strength > 0 and end_idx >= 1:
            img = _prep_image(end_image, width, height)
            lat = _encode_body_slot(vae, img, **enc_kw)
            _place(latent, mask, filled, end_idx, lat, end_strength, ramp_frames)
            meta["end"].append(end_idx)

        # ---- middle (body-slot semantics) ----
        if middle_image is not None and middle_strength > 0 and base_t > 3:
            idx = int(round(middle_position * (base_t - 1)))
            idx = max(1, min(idx, base_t - 2))
            if filled[idx]:                               # avoid colliding with another keyframe
                idx = idx - 1 if not filled[idx - 1] and idx - 1 >= 1 else idx + 1
            if 1 <= idx < total_t and not filled[idx]:
                img = _prep_image(middle_image, width, height)
                lat = _encode_body_slot(vae, img, **enc_kw)
                _place(latent, mask, filled, idx, lat, middle_strength, ramp_frames)
                meta["middle"].append(idx)

        rep = (batch_size,) + (1,) * 4
        out = {
            "samples": latent.repeat(rep),        # raw VAE latent; sampler scales it itself
            "noise_mask": mask.repeat(rep),
            "wan5b_keyframes": meta,              # survives KSampler (latent.copy())
        }
        return (out, length)


# --------------------------------------------------------------------------- #
# release locks for a second pass
# --------------------------------------------------------------------------- #
class Wan5B_ReleaseKeyframeMask:
    """
    Two-pass workflow:
      pass 1: locked keyframes, normal denoise  -> establishes structure
      pass 2: this node, then a lower-denoise pass (e.g. denoise 0.35-0.55)
              -> middle/end become free so motion flows through them.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "release_middle": ("BOOLEAN", {"default": True}),
                "release_end": ("BOOLEAN", {"default": True}),
                "release_start": ("BOOLEAN", {"default": False}),
                "release_amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "1.0 = fully free. Lower keeps a partial lock."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "release"
    CATEGORY = "Wan5B/keyframe"

    def release(self, latent, release_middle, release_end, release_start, release_amount):
        out = latent.copy()
        mask = out.get("noise_mask", None)
        meta = out.get("wan5b_keyframes", None)
        if mask is None or meta is None:
            return (out,)

        mask = mask.clone()
        idxs = []
        if release_start:
            idxs += meta.get("start", [])
        if release_middle:
            idxs += meta.get("middle", [])
        if release_end:
            idxs += meta.get("end", [])

        # also release ramp slots sitting before released keyframes
        for i in idxs:
            j = i
            while j >= 1 and j >= i - 8:
                cur = mask[:, :, j]
                mask[:, :, j] = cur + (1.0 - cur) * release_amount
                j -= 1
                if j < 1 or mask[:, :, j].min() >= 1.0:
                    break

        out["noise_mask"] = mask
        return (out,)


# --------------------------------------------------------------------------- #
# trim padding after decode
# --------------------------------------------------------------------------- #
class Wan5B_TrimFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_frames": ("INT", {"default": 121, "min": 1, "max": 100000, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim"
    CATEGORY = "Wan5B/keyframe"

    def trim(self, images, output_frames):
        return (images[:output_frames],)


NODE_CLASS_MAPPINGS = {
    "Wan5B_KeyframeToLatent": Wan5B_KeyframeToLatent,
    "Wan5B_ReleaseKeyframeMask": Wan5B_ReleaseKeyframeMask,
    "Wan5B_TrimFrames": Wan5B_TrimFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan5B_KeyframeToLatent": "Wan5B Keyframes to Latent",
    "Wan5B_ReleaseKeyframeMask": "Wan5B Release Keyframe Mask",
    "Wan5B_TrimFrames": "Wan5B Trim Frames",
}
