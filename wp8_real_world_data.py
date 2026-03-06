import cv2
import numpy as np
from wp2_heightmap_synthesis import generate_noise_map, periodic_edt

def create_earth_mask(h, w):
    """Creates a coarse mask for Earth's continents."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # Americas
    cv2.fillPoly(mask, [(np.array([[0.1, 0.2], [0.2, 0.15], [0.3, 0.4], [0.25, 0.8], [0.15, 0.9], [0.1, 0.7]]) * [w, h]).astype(np.int32)], 255)
    # Eurasia + Africa
    cv2.fillPoly(mask, [(np.array([[0.4, 0.1], [0.6, 0.05], [0.9, 0.1], [0.95, 0.3], [0.8, 0.5], [0.7, 0.4], [0.5, 0.4], [0.4, 0.2]]) * [w, h]).astype(np.int32)], 255)
    cv2.fillPoly(mask, [(np.array([[0.45, 0.4], [0.6, 0.4], [0.55, 0.7], [0.4, 0.6], [0.45, 0.45]]) * [w, h]).astype(np.int32)], 255)
    # Australia
    cv2.fillPoly(mask, [(np.array([[0.75, 0.6], [0.85, 0.65], [0.8, 0.8], [0.7, 0.75]]) * [w, h]).astype(np.int32)], 255)
    # Antarctica
    cv2.rectangle(mask, (0, int(0.9 * h)), (w, h), 255, -1)
    
    # Smooth the mask to make it look more organic
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def create_mars_mask(h, w):
    """Creates a coarse mask for Mars' features (Hellas, Argyre, Northern Basin)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # Northern Basin (lower elevation, so mask is "ocean")
    # We want a "Green Mars" where the northern basin is a sea.
    # Our heightmap logic makes masked areas land, so for Mars we'll mask the southern highlands.
    cv2.rectangle(mask, (0, int(0.4 * h)), (w, h), 255, -1)
    # Hellas Basin (hole in southern highlands)
    cv2.circle(mask, (int(0.7 * w), int(0.7 * h)), int(0.1 * h), 0, -1)
    # Argyre Basin (hole in southern highlands)
    cv2.circle(mask, (int(0.3 * w), int(0.7 * h)), int(0.05 * h), 0, -1)
    
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def synthesize_real_world_map(name, mask_func, sea_level_offset=0.0):
    print(f"Synthesizing {name}...")
    h, w = 1024, 2048
    mask = mask_func(h, w)
    
    is_land = mask > 127
    
    # 1. Base Elevation
    dist_land = periodic_edt(is_land).astype(np.float32)
    max_dist = dist_land.max()
    base_elevation = (dist_land / max_dist) ** 0.5
    
    # 2. Add Noise
    noise_land = generate_noise_map((h, w), scale=150.0, octaves=8)
    heightmap_land = 0.3 * base_elevation + 0.7 * (base_elevation * noise_land)
    heightmap_land = heightmap_land ** 1.5 
    
    # 3. Ocean Depth
    dist_ocean = periodic_edt(~is_land).astype(np.float32)
    max_ocean_dist = dist_ocean.max()
    base_depth = (dist_ocean / max_ocean_dist) ** 0.7 
    noise_ocean = generate_noise_map((h, w), scale=300.0, octaves=8)
    heightmap_ocean = -(0.2 * base_depth + 0.8 * (base_depth * noise_ocean))
    
    # 4. Combine
    final_h = np.zeros((h, w), dtype=np.float32)
    final_h[is_land] = heightmap_land[is_land]
    final_h[~is_land] = heightmap_ocean[~is_land]
    
    # 5. Apply Sea Level Offset (for higher/lower sea levels)
    final_h -= sea_level_offset
    
    # 6. Normalize to 16-bit
    h_min, h_max = final_h.min(), final_h.max()
    h_norm = (final_h - h_min) / (h_max - h_min)
    h_16bit = (h_norm * 65535).astype(np.uint16)
    
    # 7. Update mask based on new sea level
    new_mask = (final_h > 0).astype(np.uint8) * 255
    
    output_height = f'{name.lower()}_height_map.png'
    output_mask = f'{name.lower()}_mask.png'
    cv2.imwrite(output_height, h_16bit)
    cv2.imwrite(output_mask, new_mask)
    
    print(f"Saved {output_height} and {output_mask}")

if __name__ == "__main__":
    # Simulate "Drowned Earth" (Sea level up by 0.1)
    synthesize_real_world_map("Earth", create_earth_mask, sea_level_offset=0.1)
    # Simulate "Green Mars" (Sea level set at current average depth of basins)
    synthesize_real_world_map("Mars", create_mars_mask, sea_level_offset=-0.2)
