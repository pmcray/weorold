import cv2
import numpy as np
from wp2_heightmap_synthesis import generate_noise_map

def generate_random_landmask(h, w, land_threshold=0.5, scale=200.0, octaves=8):
    """Generates a random landmask using fractal noise."""
    print(f"Generating random landmask ({h}x{w})...")
    noise = generate_noise_map((h, w), scale=scale, octaves=octaves)
    
    # Threshold to create land/sea
    mask = (noise > land_threshold).astype(np.uint8) * 255
    
    # Optional: Clean up small islands or holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove small dots
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill small holes
    
    return mask

if __name__ == "__main__":
    h, w = 1024, 2048
    # Test with different seeds (randomized by noise function)
    mask = generate_random_landmask(h, w, land_threshold=0.55)
    cv2.imwrite('wp6_random_mask.png', mask)
    print("Random mask saved to wp6_random_mask.png")
