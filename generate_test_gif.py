#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image

size_dir = "generated/increasing_size"
frames_dir = "generated/increasing_frames"
os.makedirs(size_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

def generate_single_frame_gif(width, height, filename):
    """
    Generates a single-frame noise image (RGB), then saves it as a GIF
    with a single global color map (no local colormap).
    """
    # Create an RGB noise image
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img_rgb = Image.fromarray(noise, mode='RGB')

    # We'll let Pillow generate a single global palette when saving:
    #   - "optimize=False" ensures it does not create local color maps
    #   - "palette=Image.ADAPTIVE" for a global adaptive palette
    img_rgb.save(
        filename,
        format='GIF',
        save_all=False,   # single-frame
        optimize=False,
        palette=Image.ADAPTIVE,
        loop=0
    )

def generate_multi_frame_gif(num_frames, width, height, filename):
    """
    Generates a multi-frame noise GIF with a single global color map,
    ensuring no local colormaps appear in each frame.
    """
    frames_rgb = []
    for _ in range(num_frames):
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frames_rgb.append(Image.fromarray(noise, mode='RGB'))

    # We'll let Pillow do one single global palette at save time.
    # For multi-frame:
    #   - frames_rgb[0].save(..., save_all=True, append_images=frames_rgb[1:], ...)
    #   - "optimize=False" prevents local colormaps
    #   - "palette=Image.ADAPTIVE" => global adaptive palette
    frames_rgb[0].save(
        filename,
        format='GIF',
        save_all=True,
        append_images=frames_rgb[1:],
        loop=0,
        duration=200,
        optimize=False,
        palette=Image.ADAPTIVE,
        disposal=2
    )

print("Generating GIFs with increasing image sizes (single-frame) ...")
for i in range(1, 11):
    w = h = i * 100  # 100x100, 200x200, ... 1000x1000
    out_gif = os.path.join(size_dir, f"noise_size_{w}x{h}.gif")
    generate_single_frame_gif(w, h, out_gif)
    print(f"Saved {out_gif}")

print("\nGenerating animated GIFs with increasing frame counts ...")
for num in range(1, 11):
    w, h = 300, 300  # fixed resolution for all frames
    out_gif = os.path.join(frames_dir, f"noise_frames_{num}.gif")
    generate_multi_frame_gif(num, w, h, out_gif)
    print(f"Saved {out_gif}")

print("\nGeneration complete.")