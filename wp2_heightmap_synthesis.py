import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

def generate_noise_map(shape, scale=100.0, octaves=8, persistence=0.5, lacunarity=2.2):
    """
    Generates a horizontally seamless fractal noise map.
    """
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        # Calculate resolution for this octave
        w_low = int(max(2, shape[1] * frequency / scale))
        h_low = int(max(2, shape[0] * frequency / scale))
        
        # Generate low-res random noise
        # Using a normal distribution for better tonal variety
        low_res_noise = np.random.normal(0, 1, (h_low, w_low)).astype(np.float32)
        
        # Horizontal wrap padding (4 columns for cubic interpolation safety)
        pad = 4
        low_res_noise_padded = np.pad(low_res_noise, ((0, 0), (pad, pad)), mode='wrap')
        
        # Resize to maintain density
        target_w_padded = int(shape[1] * (w_low + 2 * pad) / w_low)
        # Use bilinear for speed and smoothness in higher frequency layers
        interp = cv2.INTER_CUBIC if (h_low < 100) else cv2.INTER_LINEAR
        layer_padded = cv2.resize(low_res_noise_padded, (target_w_padded, shape[0]), interpolation=interp)
        
        # Crop the central part
        start_x = int(pad * shape[1] / w_low)
        layer = layer_padded[:, start_x:start_x + shape[1]]
        
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    # Normalize to [0, 1]
    n_min, n_max = noise.min(), noise.max()
    return (noise - n_min) / (n_max - n_min)

def periodic_edt(mask):
    """Computes distance transform with horizontal wrapping."""
    # Triple the width to handle wrapping
    mask_tripled = np.tile(mask, (1, 3))
    dist_tripled = distance_transform_edt(mask_tripled)
    # Extract the middle part
    w = mask.shape[1]
    return dist_tripled[:, w:2*w]

def synthesize_heightmap(mask_path, output_path='wp2_height_map.png'):
    print(f"Loading mask from {mask_path}...")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    is_land = mask > 127
    h, w = mask.shape

    # 1. Base Land Elevation (Distance from Coast)
    print("Calculating base land elevation...")
    dist_land = periodic_edt(is_land).astype(np.float32)
    # Normalize by max distance in each continent (roughly)
    # For a simpler approach, normalize by global max or local max
    max_dist = dist_land.max()
    base_elevation = (dist_land / max_dist) ** 0.5 # Square root for faster rise near coast
    
    # 2. Fractal Noise (Land)
    print("Generating land noise...")
    # Large scale for mountains, small scale for hills
    noise_land = generate_noise_map((h, w), scale=150.0, octaves=8)
    # Boost mountains in the center of landmasses
    heightmap_land = 0.3 * base_elevation + 0.7 * (base_elevation * noise_land)
    # Apply exponential peaking for sharper mountains
    heightmap_land = heightmap_land ** 1.5 
    
    # 3. Base Ocean Depth (Distance from Coast)
    print("Calculating ocean depth...")
    dist_ocean = periodic_edt(~is_land).astype(np.float32)
    max_ocean_dist = dist_ocean.max()
    base_depth = (dist_ocean / max_ocean_dist) ** 0.7 
    
    # 4. Fractal Noise (Ocean)
    print("Generating ocean noise...")
    # Increase octaves for smoother, deeper fractal look
    noise_ocean = generate_noise_map((h, w), scale=300.0, octaves=8)
    heightmap_ocean = -(0.2 * base_depth + 0.8 * (base_depth * noise_ocean))
    
    # 5. Combine and Normalize
    # Map from [-1.0, 1.0] where 0.0 is sea level
    final_h = np.zeros((h, w), dtype=np.float32)
    final_h[is_land] = heightmap_land[is_land]
    final_h[~is_land] = heightmap_ocean[~is_land]
    
    # Normalize to 0-65535 (16-bit grayscale)
    # Sea level will be around 0.5 (32768)
    h_min, h_max = final_h.min(), final_h.max()
    h_norm = (final_h - h_min) / (h_max - h_min)
    h_16bit = (h_norm * 65535).astype(np.uint16)
    
    cv2.imwrite(output_path, h_16bit)
    print(f"Heightmap saved to {output_path}")

    # Generate a debug preview (8-bit color map)
    # We'll use a terrain-like colormap for the preview
    h_8bit = (h_norm * 255).astype(np.uint8)
    # 0 is deep ocean, 128 is sea level, 255 is peak
    preview = cv2.applyColorMap(h_8bit, cv2.COLORMAP_JET)
    cv2.imwrite('wp2_height_preview.png', preview)
    print("Preview saved to wp2_height_preview.png")

if __name__ == "__main__":
    synthesize_heightmap('wp1_fractal_mask.png')
