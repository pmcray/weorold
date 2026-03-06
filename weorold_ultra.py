import cv2
import numpy as np
import os
import subprocess

# --- Configuration ---
TEXTURE_PATH = 'wp5_shaded_clouds.png'
OUTPUT_ANIMATION_PATH = 'ultra_globe_rotation.mp4'
BACKGROUND_COLOR = (0, 0, 0)

def create_rotating_globe(texture_path, output_path, frames=60, fps=30):
    print(f"Loading high-res texture from {texture_path}...")
    img = cv2.imread(texture_path)
    if img is None:
        raise FileNotFoundError(f"Texture not found at {texture_path}")
    
    # Ensure texture is in float for better math
    img = img.astype(np.float32)
    h, w, _ = img.shape
    
    # SSAA (Super-Sampled Anti-Aliasing): Render at 2x resolution then downscale
    # This completely eliminates aliasing/blockiness at the edges and in high-freq details.
    render_size = 1024 # Final size
    ssaa_size = render_size * 2
    print(f"Rendering high-fidelity globe (SSAA 2x) at {ssaa_size}px...")

    # 1. Coordinate Grid
    y, x = np.indices((ssaa_size, ssaa_size), dtype=np.float32)
    
    # 2. Camera & Projection (Perspective)
    # distance from camera to center of sphere
    d = 2.5 
    # field of view scale
    fov_scale = 1.2
    
    # Normalize grid to [-1.2, 1.2] roughly
    cx, cy = ssaa_size / 2.0, ssaa_size / 2.0
    nx = (x - cx) / (ssaa_size / 2.0) * fov_scale
    ny = (y - cy) / (ssaa_size / 2.0) * fov_scale
    
    # Solve for ray-sphere intersection
    # Ray origin: (0, 0, d). Ray direction: (nx, ny, -1)
    # Equation: (nx*t)^2 + (ny*t)^2 + (d-t)^2 = 1
    # (nx^2 + ny^2 + 1)t^2 - 2dt + (d^2 - 1) = 0
    a = nx**2 + ny**2 + 1
    b = -2 * d
    c = d**2 - 1
    
    discriminant = b**2 - 4 * a * c
    mask = discriminant >= 0
    
    t = np.zeros_like(nx)
    t[mask] = (-b - np.sqrt(discriminant[mask])) / (2 * a[mask])
    
    # Interaction points on the sphere
    px = nx * t
    py = ny * t
    pz = d - t
    
    # Unit vectors from center (0,0,0) to intersection points
    # (Already unit length because we intersected with r=1)
    # Now we have our spherical unit vectors.
    
    # Latitude (phi) - [-pi/2, pi/2]
    phi = np.arcsin(np.clip(py, -1, 1))
    
    # Longitude (theta)
    theta = np.zeros_like(nx)
    theta[mask] = np.arctan2(px[mask], pz[mask])
    
    # Base mappings
    v_map = ((0.5 - phi / np.pi) * (h - 1)).astype(np.float32)
    u_base = (theta / (2 * np.pi)) * (w - 1)

    # Use H.264 codec for web/notebook compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (render_size, render_size))

    if not out.isOpened():
        print("Warning: 'avc1' (H.264) codec failed. Falling back to 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (render_size, render_size))

    print("Generating ultra-smooth perspective animation...")
    for i in range(frames):
        if i % 10 == 0:
            print(f"Rendering frame {i}/{frames}...")
            
        # Rotation shift
        u_shift = (i / frames) * (w - 1)
        u_map = ((u_base + u_shift) % (w - 1)).astype(np.float32)
        
        # SSAA high-res frame
        # CUBIC is much sharper than LINEAR
        ssaa_frame = cv2.remap(img, u_map, v_map, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        # Apply black space
        ssaa_frame[~mask] = 0
        
        # Downsample to final size (this is the SSAA magic)
        final_frame = cv2.resize(ssaa_frame, (render_size, render_size), interpolation=cv2.INTER_AREA)
        
        out.write(final_frame.astype(np.uint8))

    out.release()
    print(f"Animation saved to {output_path}")

    # Re-encode for browser compatibility using ffmpeg if available
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    if os.path.exists(output_path):
        os.rename(output_path, temp_path)
        print("Re-encoding video for browser compatibility (H.264)...")
        try:
            # Use ffmpeg to convert to H.264 (libx264) with yuv420p pixel format
            # and faststart for streaming/notebook playback.
            subprocess.run([
                'ffmpeg', '-y', '-v', 'error',
                '-i', temp_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ], check=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Successfully re-encoded to H.264: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: ffmpeg re-encoding failed ({e}). Keeping original file.")
            if os.path.exists(temp_path):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)

import sys

def main():
    if len(sys.argv) > 1:
        texture_path = sys.argv[1]
    else:
        texture_path = TEXTURE_PATH

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = OUTPUT_ANIMATION_PATH

    create_rotating_globe(texture_path, output_path, frames=60)

if __name__ == "__main__":
    main()
