#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image

# Directories for output images
size_dir = "generated/increasing_size"
frames_dir = "generated/increasing_frames"
os.makedirs(size_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# -------------------------------
# Set 1: Increasing Image Sizes
# -------------------------------
print("Generating GIFs with increasing image sizes (noise images)...")
for i in range(1, 11):
    # Increase size by 100 pixels each step (100x100, 200x200, ..., 1000x1000)
    width = height = i * 100
    # Create random noise image (RGB)
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise, 'RGB')
    # Convert to a palette-based image for GIF format
    img = img.convert('P', palette=Image.ADAPTIVE)
    filename = os.path.join(size_dir, f"noise_size_{width}x{height}.gif")
    img.save(filename)
    print(f"Saved {filename}")

# -----------------------------------------
# Set 2: Increasing Number of Frames
# -----------------------------------------
print("\nGenerating animated GIFs with increasing number of frames (noise images)...")
for num_frames in range(1, 11):
    frames = []
    # Fixed size for all frames
    width, height = 300, 300
    for f in range(num_frames):
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frame = Image.fromarray(noise, 'RGB')
        # Convert to palette-based image for GIFs
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        frames.append(frame)
    filename = os.path.join(frames_dir, f"noise_frames_{num_frames}.gif")
    # Save frames as an animated GIF; duration is in milliseconds, loop=0 means infinite loop.
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"Saved {filename}")

print("\nTest GIF generation complete.")
