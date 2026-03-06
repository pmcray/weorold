import cv2
import numpy as np

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
        pad = 3
        grid_padded = np.pad(grid, ((pad, pad), (pad, pad)), mode='wrap')
        
        # Resize to full resolution with cubic interpolation for smoothness
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
        
    return (noise - noise.min()) / (noise.max() - noise.min())

def color_lerp_multi(val, stops, colors):
    """Linearly interpolates colors based on value stops for arrays of values."""
    val = np.asarray(val)
    out = np.zeros((val.shape[0], 3), dtype=np.float32)
    for i in range(3): # R, G, B
        out[:, i] = np.interp(val, stops, [c[i] for c in colors])
    return out

def create_surface_texture(heightmap_path, mask_path='wp1_fractal_mask.png', output_prefix='wp3', temperature=0.5, moisture=0.5):
    print(f"Loading heightmap from {heightmap_path} and mask from {mask_path} (Temp: {temperature}, Moisture: {moisture})...")
    h16 = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")
    is_land = mask > 127
    h_norm = h16.astype(np.float32) / 65535.0
    height, width = h_norm.shape
    del h16

    # Normalize land and ocean separately
    if np.any(is_land):
        land_vals = h_norm[is_land]
        h_land_norm = (land_vals - land_vals.min()) / (max(1e-6, land_vals.max() - land_vals.min()))
    else:
        h_land_norm = np.array([], dtype=np.float32)

    if np.any(~is_land):
        ocean_vals = h_norm[~is_land]
        h_ocean_norm = (ocean_vals.max() - ocean_vals) / (max(1e-6, ocean_vals.max() - ocean_vals.min()))
    else:
        h_ocean_norm = np.array([], dtype=np.float32)
    
    # --- RELIEF MAP ---
    print("Generating Relief Map...")
    relief_land_stops = [0.0, 0.2, 0.5, 0.8, 1.0]
    relief_land_colors = [[34, 139, 34], [154, 205, 50], [222, 184, 135], [139, 69, 19], [255, 250, 250]]
    relief_ocean_stops = [0.0, 1.0]
    relief_ocean_colors = [[50, 100, 200], [10, 30, 80]]

    relief_img = np.zeros((height, width, 3), dtype=np.uint8)
    if np.any(is_land):
        relief_img[is_land] = color_lerp_multi(h_land_norm, relief_land_stops, relief_land_colors)
    if np.any(~is_land):
        relief_img[~is_land] = color_lerp_multi(h_ocean_norm, relief_ocean_stops, relief_ocean_colors)
    
    # --- BIOME MAP ---
    print("Generating Biome Map...")
    biome_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if np.any(is_land):
        # Calculate latitude factor only for land pixels to save memory
        row_indices = np.where(is_land)[0]
        lat_factor_land = np.abs((row_indices.astype(np.float32) / (height - 1)) - 0.5) * 2
        
        # 1. Base logic: Forest/Mountain base
        # Shift forest to mountain transition based on moisture
        forest_threshold = 0.4 + (moisture * 0.4)
        land_colors = color_lerp_multi(h_land_norm, [0, forest_threshold, 1.0], [[34, 139, 34], [100, 100, 100], [255, 255, 255]])
        
        # 2. Polar Ice Caps (Linear blend)
        # Shift ice cap size based on temperature
        ice_threshold = 0.6 + (temperature * 0.3)
        ice_mask = lat_factor_land > ice_threshold
        if np.any(ice_mask):
            alpha = (lat_factor_land[ice_mask] - ice_threshold) / (1.0 - ice_threshold)
            ice_color = [240, 248, 255]
            for i in range(3):
                land_colors[ice_mask, i] = (1 - alpha) * land_colors[ice_mask, i] + alpha * ice_color[i]
        
        # 3. Equatorial Deserts (Low lat, Low height)
        # Desert size inversely related to moisture
        desert_intensity = (1.5 - moisture)
        desert_weight = np.clip((1.0 - lat_factor_land) * (1.0 - h_land_norm) - (0.2 + moisture * 0.3), 0, 1) * desert_intensity
        desert_weight = np.clip(desert_weight, 0, 1)
        desert_color = [210, 180, 140]
        for i in range(3):
            land_colors[:, i] = (1 - desert_weight) * land_colors[:, i] + desert_weight * desert_color[i]

        biome_img[is_land] = land_colors
    
    if np.any(~is_land):
        biome_img[~is_land] = relief_img[~is_land]
    
    # --- Detail Noise ---
    print("Generating detail noise...")
    # Global noise for base texture
    noise_layer = generate_noise_map((height, width), scale=20.0, octaves=6)
    
    # Secondary high-detail fractal noise for ocean depth
    ocean_noise = generate_noise_map((height, width), scale=50.0, octaves=8, persistence=0.6)
    
    # Blend noise layers
    detail_noise = (0.8 + 0.4 * noise_layer)
    if np.any(~is_land):
        detail_noise[~is_land] *= (0.9 + 0.2 * ocean_noise[~is_land])
    detail_noise = detail_noise[:, :, np.newaxis]
    
    final_relief = np.clip(relief_img * detail_noise, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_relief_map.png', cv2.cvtColor(final_relief, cv2.COLOR_RGB2BGR))
    
    final_biome = np.clip(biome_img * detail_noise, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_biome_map.png', cv2.cvtColor(final_biome, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_prefix}_surface_texture.png', cv2.cvtColor(final_biome, cv2.COLOR_RGB2BGR))
    print(f"Saved outputs with prefix {output_prefix}.")

if __name__ == "__main__":
    create_surface_texture('wp2_height_map.png')
