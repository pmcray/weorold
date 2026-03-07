import cv2
import numpy as np
import scipy.ndimage as ndimage

def generate_noise_map(shape, scale=100.0, octaves=8, persistence=0.5, lacunarity=2.2):
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        w_grid = int(max(4, shape[1] * frequency / scale))
        h_grid = int(max(4, shape[0] * frequency / scale))
        
        grid = np.random.normal(0, 1, (h_grid, w_grid)).astype(np.float32)
        pad = 3
        grid_padded = np.pad(grid, ((pad, pad), (pad, pad)), mode='wrap')
        
        layer = cv2.resize(grid_padded, 
                           (int(shape[1] * (w_grid + 2*pad) / w_grid), 
                            int(shape[0] * (h_grid + 2*pad) / h_grid)), 
                           interpolation=cv2.INTER_CUBIC)
        
        start_y = int(pad * shape[0] / h_grid)
        start_x = int(pad * shape[1] / w_grid)
        noise += layer[start_y:start_y + shape[0], start_x:start_x + shape[1]] * amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
        
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

def simulate_climate(height_norm, is_land, global_temp=0.5, global_moisture=0.5):
    height, width = height_norm.shape
    
    # 1. Temperature: based on latitude and elevation
    y_indices = np.linspace(0, 1, height)[:, None]
    lat_factor = 1.0 - np.abs(y_indices - 0.5) * 2  # 1 at equator, 0 at poles
    
    base_temp = lat_factor * (0.5 + global_temp * 0.5)
    # Elevation cooling
    temp_map = base_temp - (height_norm * 0.6)
    temp_map = np.clip(temp_map, 0, 1)
    
    # 2. Moisture & Prevailing Winds (Rain Shadows)
    # Simplified wind: West to East in temperate zones, East to West in tropics
    wind_x = np.where((lat_factor > 0.3) & (lat_factor < 0.7), 1.0, -1.0)
    
    moisture_map = np.full((height, width), global_moisture, dtype=np.float32)
    moisture_map[~is_land] = 1.0 # Oceans are full moisture
    
    # Simple Rain Shadow simulation (sweep horizontally)
    print("Simulating rain shadows...")
    for y in range(height):
        current_moisture = 1.0
        # Determine sweep direction based on wind
        xs = range(width) if wind_x[y, 0] > 0 else range(width-1, -1, -1)
        for x in xs:
            if not is_land[y, x]:
                current_moisture = 1.0 # recharge over ocean
            else:
                # lose moisture based on elevation change (orographic effect)
                elev = height_norm[y, x]
                if x > 0 and x < width-1:
                    prev_elev = height_norm[y, x-1] if wind_x[y, 0] > 0 else height_norm[y, x+1]
                    if elev > prev_elev:
                        # Dropping rain on windward side
                        moisture_map[y, x] = min(1.0, current_moisture + 0.2)
                        current_moisture *= 0.8 # Lose moisture
                    else:
                        # Leeward side (rain shadow)
                        moisture_map[y, x] = current_moisture
                        current_moisture *= 0.99 # Slow evaporation
                else:
                    moisture_map[y, x] = current_moisture
                    
    # Smooth the maps to remove artifacts
    temp_map = ndimage.gaussian_filter(temp_map, sigma=2)
    moisture_map = ndimage.gaussian_filter(moisture_map, sigma=2)
    
    return temp_map, moisture_map

def whittaker_biome_color(t, m):
    # Colors (R, G, B)
    TROPICAL_RAINFOREST = [34, 139, 34]
    TROPICAL_SEASONAL_FOREST = [107, 142, 35]
    SAVANNA = [189, 183, 107]
    SUBTROPICAL_DESERT = [210, 180, 140]
    TEMPERATE_RAINFOREST = [46, 139, 87]
    TEMPERATE_DECIDUOUS_FOREST = [0, 100, 0]
    WOODLAND = [143, 188, 143]
    TEMPERATE_DESERT = [244, 164, 96]
    TAIGA = [85, 107, 47]
    TUNDRA = [176, 224, 230]
    ICE = [240, 248, 255]

    # Map T [0..1] and M [0..1]
    if t < 0.2:
        return ICE if m < 0.5 else TUNDRA
    elif t < 0.4:
        return TUNDRA if m < 0.3 else TAIGA
    elif t < 0.7:
        if m < 0.2: return TEMPERATE_DESERT
        if m < 0.4: return WOODLAND
        if m < 0.7: return TEMPERATE_DECIDUOUS_FOREST
        return TEMPERATE_RAINFOREST
    else:
        if m < 0.2: return SUBTROPICAL_DESERT
        if m < 0.4: return SAVANNA
        if m < 0.7: return TROPICAL_SEASONAL_FOREST
        return TROPICAL_RAINFOREST

def color_lerp_multi(val, stops, colors):
    val = np.asarray(val)
    out = np.zeros((val.shape[0], 3), dtype=np.float32)
    for i in range(3):
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

    print("Simulating Climate...")
    temp_map, moisture_map = simulate_climate(h_norm, is_land, global_temp=temperature, global_moisture=moisture)

    print("Generating Biome Map...")
    biome_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Vectorized Whittaker Biome Application
    for y in range(height):
        for x in range(width):
            if is_land[y, x]:
                biome_img[y, x] = whittaker_biome_color(temp_map[y, x], moisture_map[y, x])
    
    relief_ocean_stops = [0.0, 1.0]
    relief_ocean_colors = [[50, 100, 200], [10, 30, 80]]
    if np.any(~is_land):
        ocean_vals = h_norm[~is_land]
        h_ocean_norm = (ocean_vals.max() - ocean_vals) / (max(1e-6, ocean_vals.max() - ocean_vals.min()))
        biome_img[~is_land] = color_lerp_multi(h_ocean_norm, relief_ocean_stops, relief_ocean_colors)

    print("Generating detail noise...")
    noise_layer = generate_noise_map((height, width), scale=20.0, octaves=6)
    ocean_noise = generate_noise_map((height, width), scale=50.0, octaves=8, persistence=0.6)
    
    detail_noise = (0.8 + 0.4 * noise_layer)
    if np.any(~is_land):
        detail_noise[~is_land] *= (0.9 + 0.2 * ocean_noise[~is_land])
    detail_noise = detail_noise[:, :, np.newaxis]
    
    final_biome = np.clip(biome_img * detail_noise, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_biome_map.png', cv2.cvtColor(final_biome, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_prefix}_surface_texture.png', cv2.cvtColor(final_biome, cv2.COLOR_RGB2BGR))
    
    # Export temp and moisture maps for other systems (like vegetation)
    cv2.imwrite(f'{output_prefix}_temperature_map.png', (temp_map * 255).astype(np.uint8))
    cv2.imwrite(f'{output_prefix}_moisture_map.png', (moisture_map * 255).astype(np.uint8))
    
    print(f"Saved outputs with prefix {output_prefix}.")

if __name__ == "__main__":
    create_surface_texture('wp2_height_map.png')
