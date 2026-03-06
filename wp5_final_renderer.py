import cv2
import numpy as np

def generate_normal_map(h16, strength=10.0):
    print("Generating normal map...")
    # Convert heightmap to float
    h = h16.astype(np.float32) / 65535.0
    
    # Pad horizontally for horizontal wrapping (1px is enough for Sobel 3x3)
    h_padded = cv2.copyMakeBorder(h, 0, 0, 1, 1, cv2.BORDER_WRAP)
    
    # Calculate gradients on padded image
    dzdx_p = cv2.Sobel(h_padded, cv2.CV_32F, 1, 0, ksize=3)
    dzdy_p = cv2.Sobel(h_padded, cv2.CV_32F, 0, 1, ksize=3)
    
    # Crop back to original size
    dzdx = dzdx_p[:, 1:-1]
    dzdy = dzdy_p[:, 1:-1]
    
    # Normal vector components
    nx = -dzdx * strength
    ny = -dzdy * strength
    nz = 1.0
    
    # Normalize
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm
    
    # Map to RGB [0, 255]
    # Standard normal map format: X->R, Y->G, Z->B
    # Shift [-1, 1] to [0, 255]
    normal_map = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((nx + 1) * 0.5 * 255).astype(np.uint8)
    normal_map[:, :, 1] = ((ny + 1) * 0.5 * 255).astype(np.uint8)
    normal_map[:, :, 2] = ((nz + 1) * 0.5 * 255).astype(np.uint8)
    
    return normal_map, (nx, ny, nz)

def apply_shading(texture, normal_vecs, light_dir=(1.0, 1.0, 0.5), mask_path=None):
    print("Applying Phong shading...")
    nx, ny, nz = normal_vecs
    
    # Normalize light direction
    light_dir = np.array(light_dir)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # 1. Diffuse (Lambertian)
    dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    dot_clamped = np.clip(dot, 0.2, 1.0)
    
    # 2. Specular (Phong) for water
    specular = np.zeros_like(dot)
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            is_ocean = mask < 127
            # Reflection vector: R = 2*(N.L)*N - L
            rx = 2 * dot * nx - light_dir[0]
            ry = 2 * dot * ny - light_dir[1]
            rz = 2 * dot * nz - light_dir[2]
            
            # View vector (assume camera is at +Z)
            vx, vy, vz = 0, 0, 1.0
            
            # Specular: (R.V)^n
            rdotv = rx * vx + ry * vy + rz * vz
            specular[is_ocean] = np.clip(rdotv[is_ocean], 0, 1) ** 32 # Sharp specular
            
    # Combine
    shaded = texture.astype(np.float32) * dot_clamped[:, :, np.newaxis]
    # Add specular white highlights
    shaded += (specular[:, :, np.newaxis] * 255 * 0.5) 
    
    return np.clip(shaded, 0, 255).astype(np.uint8)

def render_final_maps(texture_path='wp3_surface_texture.png', heightmap_path='wp2_height_map.png', cloud_path='wp4_cloud_map.png', mask_path='wp1_fractal_mask.png', output_prefix='wp5'):
    # 1. Load Data
    print(f"Loading data for final render from {texture_path} and {heightmap_path}...")
    texture_bgr = cv2.imread(texture_path)
    if texture_bgr is None:
        raise FileNotFoundError(f"Texture not found at {texture_path}")
    texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
    
    h16 = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    if h16 is None:
        raise FileNotFoundError(f"Heightmap not found at {heightmap_path}")
    
    clouds = cv2.imread(cloud_path, cv2.IMREAD_UNCHANGED) # RGBA

    # 2. Shading
    normal_map, normal_vecs = generate_normal_map(h16, strength=10.0)
    shaded_terrain = apply_shading(texture, normal_vecs, light_dir=(1.0, 1.0, 1.0), mask_path=mask_path)
    
    cv2.imwrite(f'{output_prefix}_normal_map.png', cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_prefix}_shaded_terrain.png', cv2.cvtColor(shaded_terrain, cv2.COLOR_RGB2BGR))

    # 3. Combine with Clouds (Optional)
    if clouds is not None:
        print("Blending clouds...")
        if clouds.shape[0:2] != texture.shape[0:2]:
            print(f"Resizing clouds from {clouds.shape[0:2]} to {texture.shape[0:2]}...")
            clouds = cv2.resize(clouds, (texture.shape[1], texture.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        alpha_c = (clouds[:, :, 3] / 255.0)[:, :, np.newaxis]
        cloud_rgb = cv2.cvtColor(clouds[:, :, 0:3], cv2.COLOR_BGR2RGB)
        final_with_clouds = (1 - alpha_c) * shaded_terrain + alpha_c * cloud_rgb
        cv2.imwrite(f'{output_prefix}_shaded_clouds.png', cv2.cvtColor(final_with_clouds.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"Final 2D map with clouds saved to {output_prefix}_shaded_clouds.png")
    
    print(f"All 2D maps rendered with prefix {output_prefix}.")

if __name__ == "__main__":
    render_final_maps()
