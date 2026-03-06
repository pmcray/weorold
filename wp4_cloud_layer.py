import cv2
import numpy as np

def generate_noise_map(shape, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Generates a horizontally seamless fractal noise map.
    """
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        w_low = int(max(1, shape[1] * frequency / scale))
        h_low = int(max(1, shape[0] * frequency / scale))
        
        low_res_noise = np.random.randn(h_low, w_low).astype(np.float32)
        
        # Horizontal wrap padding (4 columns for cubic interpolation safety)
        pad = 4
        low_res_noise_padded = np.pad(low_res_noise, ((0, 0), (pad, pad)), mode='wrap')
        
        # Resize to maintain density
        target_w_padded = int(shape[1] * (w_low + 2 * pad) / w_low)
        layer_padded = cv2.resize(low_res_noise_padded, (target_w_padded, shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Crop the central part
        start_x = int(pad * shape[1] / w_low)
        layer = layer_padded[:, start_x:start_x + shape[1]]
        
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    # Normalize to [0, 1]
    n_min, n_max = noise.min(), noise.max()
    return (noise - n_min) / (n_max - n_min)

def create_cloud_layer(shape, output_path='wp4_cloud_map.png', density=0.5):
    height, width = shape
    print(f"Generating clouds {width}x{height} (Density: {density})...")
    
    # 1. Base Fractal Noise
    print("Generating base noise...")
    noise = generate_noise_map((height, width), scale=200.0, octaves=6)
    
    # 2. Domain Warping (Wind Effect)
    # We use two more noise maps to displace the original noise
    print("Applying domain warping (wind swirls)...")
    warp_x = generate_noise_map((height, width), scale=300.0, octaves=4) * 100 - 50
    warp_y = generate_noise_map((height, width), scale=300.0, octaves=4) * 100 - 50
    
    # Create coordinate grids
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply warping
    map_x = (cols + warp_x).astype(np.float32)
    map_y = (rows + warp_y).astype(np.float32)
    warped_noise = cv2.remap(noise, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    # 3. Latitudinal Influence (More clouds at equator and 60 N/S)
    lat = np.linspace(0, 1, height).reshape(-1, 1)
    # Earth has 3 main cloud bands: Equator (0.5), and mid-latitudes (0.2 and 0.8 roughly)
    lat_influence = np.sin(np.pi * lat * 3) ** 2 # peaks at 1/6, 3/6 (0.5), 5/6
    lat_grid = np.tile(lat_influence, (1, width))
    
    clouds = warped_noise * (0.4 + 0.6 * lat_grid)
    
    # 4. Thresholding and Softening
    # Clouds should be binary-ish but soft
    # Map noise to alpha based on density
    # If density is 0.0, threshold is high (1.0). If 1.0, threshold is low (0.0)
    threshold = 1.0 - (density * 0.8) # From 1.0 to 0.2
    alpha = np.clip((clouds - threshold) * 5.0, 0, 1) # Sharpen the edges
    alpha = (alpha * 255).astype(np.uint8)
    
    # Final image is white (255, 255, 255) with the alpha mask
    cloud_img = np.ones((height, width, 4), dtype=np.uint8) * 255
    cloud_img[:, :, 3] = alpha
    
    cv2.imwrite(output_path, cloud_img)
    print(f"Cloud map saved to {output_path}")

if __name__ == "__main__":
    # Get shape from existing heightmap
    h16 = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)
    if h16 is not None:
        create_cloud_layer(h16.shape)
    else:
        create_cloud_layer((1024, 2048))
