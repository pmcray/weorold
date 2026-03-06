import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label, regionprops

def generate_noise_map(shape, scale=100.0, octaves=8, persistence=0.5, lacunarity=2.2):
    """
    Generates a horizontally seamless fractal noise map with smoother transitions.
    """
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        # Calculate grid size for this octave
        w_grid = int(max(4, shape[1] * frequency / scale))
        h_grid = int(max(4, shape[0] * frequency / scale))
        
        # Generate random grid
        grid = np.random.normal(0, 1, (h_grid, w_grid)).astype(np.float32)
        
        # Pad for seamless horizontal wrapping
        # We use a larger pad to ensure cubic interpolation has enough context
        pad = 3
        grid_padded = np.pad(grid, ((pad, pad), (pad, pad)), mode='wrap')
        
        # Resize to full resolution
        # Using INTER_CUBIC for all layers to ensure smoothness
        layer = cv2.resize(grid_padded, 
                           (int(shape[1] * (w_grid + 2*pad) / w_grid), 
                            int(shape[0] * (h_grid + 2*pad) / h_grid)), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Extract the middle part correctly
        start_y = int(pad * shape[0] / h_grid)
        start_x = int(pad * shape[1] / w_grid)
        noise += layer[start_y:start_y + shape[0], start_x:start_x + shape[1]] * amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
        
    return (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1

def periodic_edt(mask):
    """Computes distance transform with horizontal wrapping."""
    # Triple the width to handle wrapping
    mask_tripled = np.tile(mask, (1, 3))
    dist_tripled = distance_transform_edt(mask_tripled)
    # Extract the middle part
    w = mask.shape[1]
    return dist_tripled[:, w:2*w]

def clean_binary_mask(binary_map, min_area=500):
    """
    Advanced cleaning of binary mask to remove names, legends, and noise.
    """
    print("Performing advanced mask cleaning...")
    
    # 1. Morphological opening to remove thin lines (text, ticks) while keeping large landmasses
    # A 3x3 or 5x5 kernel is usually enough to kill text but keep islands
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    
    # 2. Fill holes inside landmasses (where text might have been carved out)
    # We find contours and fill them
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(filled, [cnt], -1, 255, -1)
            
    # 3. Position-based Legend Removal
    # Most legends are in the corners, especially bottom-right
    h, w = filled.shape
    # Clear the bottom-right 25% area if components there are isolated
    legend_zone_h = int(h * 0.7)
    legend_zone_w = int(w * 0.7)
    
    label_img = label(filled)
    regions = regionprops(label_img)
    
    output_mask = filled.copy()
    for region in regions:
        r_min, c_min, r_max, c_max = region.bbox
        # If the component is entirely within the bottom-right zone and not huge, remove it
        if r_min > legend_zone_h and c_min > legend_zone_w:
            if region.area < (h * w * 0.05): # Less than 5% of total area
                # Check if it's "too geometric" (like a circle or scale bar)
                is_geometric = region.solidity > 0.9 or region.extent > 0.8
                if is_geometric or region.area < 2000:
                    for coord in region.coords:
                        output_mask[coord[0], coord[1]] = 0

    return output_mask

def process_sketch_to_fractal_mask(image_path, upscale_factor=4, noise_strength=15.0, min_land_area=800):
    """
    Loads a sketch, converts it to a high-resolution fractal landmask.
    """
    print(f"Loading sketch from {image_path}...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to HSV for better color-based separation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Define "Ocean" color (Light Blue)
    # H: 90-110, S: 10-100, V: 150-255
    lower_ocean = np.array([80, 5, 150])
    upper_ocean = np.array([120, 150, 255])
    
    ocean_mask = cv2.inRange(hsv, lower_ocean, upper_ocean)
    
    # Land is everything that is NOT ocean and NOT pure white/black (text is black)
    # Actually, let's just use the inverse of the ocean mask as the base
    binary = cv2.bitwise_not(ocean_mask)
    
    # Clean the mask before upscaling
    cleaned_binary = clean_binary_mask(binary, min_area=min_land_area)
    
    # Upscale
    new_shape = (img_bgr.shape[0] * upscale_factor, img_bgr.shape[1] * upscale_factor)
    print(f"Upscaling to {new_shape[1]}x{new_shape[0]}...")
    binary_up = cv2.resize(cleaned_binary, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 1. Compute Signed Distance Field (SDF)
    # periodic_edt computes distance to the nearest boundary with horizontal wrapping
    dist_land = periodic_edt(binary_up > 0)
    dist_ocean = periodic_edt(binary_up == 0)
    sdf = dist_land - dist_ocean # Positive inside land, negative in ocean
    
    # 2. Generate Fractal Noise
    print("Generating fractal noise layer...")
    # Scale determines the 'frequency' of the coastline jaggedness
    noise = generate_noise_map(new_shape, scale=20.0, octaves=4)
    
    # 3. Displace the SDF with noise
    # We add noise specifically near the coast (where SDF is near 0)
    # The noise_strength determines how jagged the coastline becomes
    displaced_sdf = sdf + noise * noise_strength
    
    # 4. Final Mask
    fractal_mask = (displaced_sdf > 0).astype(np.uint8) * 255
    
    # Output the result
    output_path = 'wp1_fractal_mask.png'
    cv2.imwrite(output_path, fractal_mask)
    print(f"Fractal landmask saved to {output_path}")
    
    # Save a comparison debug image (original vs fractal)
    # Downsample fractal for comparison
    comparison_fractal = cv2.resize(fractal_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    comparison = np.hstack((binary, comparison_fractal))
    cv2.imwrite('wp1_comparison.png', comparison)
    print("Comparison saved to wp1_comparison.png")

if __name__ == "__main__":
    process_sketch_to_fractal_mask('Motoki_Aspsp_uk_Fig02_c_Islands.jpg', upscale_factor=3, noise_strength=20.0)
