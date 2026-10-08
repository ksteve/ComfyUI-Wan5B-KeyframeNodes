# ComfyUI-Wan5B-KeyframeNodes

Custom ComfyUI nodes for **Wan 2.2 TI2V-5B** with proper first / last / middle keyframe support and a **soft end-frame landing** that reduces the common "last frames corrupted" artifact.

## Why this exists

The Wan 2.2 **5B** model uses a different VAE (48-channel, 4×16×16) than the 14B models. Most existing First-Last-Frame nodes were written for the old VAE or hard-lock the final latent frames, which frequently produces noisy / glitchy last frames.

These nodes are built specifically for the 5B latent layout and replace the hard lock with a controllable soft ramp.

## Nodes

### Wan5B First/Last Frame to Latent
Main node.

| Input | Description |
|-------|-------------|
| `vae` | **Must** be the Wan 2.2 VAE |
| `start_image` / `end_image` | Keyframes (optional independently) |
| `middle_image` | Optional third anchor |
| `start_strength` | 0–1 lock strength for the start (default 1.0) |
| `end_strength` | 0–1 lock strength for the end (default **0.85**) |
| `soft_end_frames` | How many latent frames the end keyframe ramps over (default **3**) |
| `middle_strength` / `middle_position` | Control for the optional middle keyframe |
| `length` | Total pixel frames (auto-snapped to 4n+1) |

**Recommended starting values for clean endings:**
- `end_strength` = 0.80 – 0.90
- `soft_end_frames` = 2 – 4

### Wan5B First/Last Frame to Latent (Tiled VAE)
Same logic, but uses `vae.encode_tiled()` to lower peak VRAM. Useful on 8–12 GB cards.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ksteve/ComfyUI-Wan5B-KeyframeNodes
```

Restart ComfyUI. The nodes appear under **Wan5B/keyframe**.

## Usage tips

1. Connect the **Wan 2.2 VAE** (not the 2.1 / 14B VAE).
2. Feed the output `latent` into your normal 5B sampler pipeline (native or Kijai wrapper).
3. If the ending still looks slightly soft, raise `end_strength` toward 1.0 or lower `soft_end_frames`.
4. If you still see corruption, increase `soft_end_frames` to 4–6.
5. Keep start and end images in a similar composition / lighting for best motion.

## License

GPL-3.0 (same family as ComfyUI core nodes this builds upon).

## Credits

- Inspired by ComfyUI native `WanFirstLastFrameToVideo` and `Wan22ImageToVideoLatent`
- Builds on the approach in [stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent](https://github.com/stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent)
- Soft-landing fix designed to address the long-standing last-frame corruption reports on 5B
