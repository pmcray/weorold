import cv2
import numpy as np
from scipy.spatial import Voronoi

def generate_tectonic_heightmap(height, width, num_plates=15, blur_sigma=20):
    print(f"Simulating {num_plates} tectonic plates...")
    
    # 1. Generate Plate Centers and Voronoi
    points = np.column_stack([
        np.random.randint(0, height, num_plates),
        np.random.randint(0, width, num_plates)
    ])
    
    # Simple plate assignment via distance (Voronoi-like)
    y, x = np.indices((height, width))
    plate_map = np.zeros((height, width), dtype=np.int32)
    
    # For speed on 2048x1024, we use a smaller grid for plate assignment and upscale
    # or just brute force the nearest point for each pixel using a KDTree or vectorized distance
    # Vectorized distance for few points is fine
    min_dist = np.full((height, width), np.inf)
    for i, p in enumerate(points):
        # Account for horizontal wrapping in distance calculation
        dx = np.abs(x - p[1])
        dx = np.minimum(dx, width - dx)
        dy = y - p[0]
        dist_sq = dx**2 + dy**2
        mask = dist_sq < min_dist
        min_dist[mask] = dist_sq[mask]
        plate_map[mask] = i

    # 2. Assign Velocities to Plates
    plate_velocities = np.random.uniform(-1, 1, (num_plates, 2)) # (dy, dx)
    
    # 3. Calculate Convergence/Divergence at boundaries
    # We look at the gradient of the plate_map to find boundaries
    print("Calculating plate boundaries and stress...")
    
    # Shift maps to find neighbors
    shift_r = np.roll(plate_map, 1, axis=1)
    shift_l = np.roll(plate_map, -1, axis=1)
    shift_d = np.roll(plate_map, 1, axis=0)
    shift_u = np.roll(plate_map, -1, axis=0)
    
    # Stress map
    tectonic_stress = np.zeros((height, width), dtype=np.float32)
    
    def calculate_boundary_stress(p1_indices, p2_indices, normal_vec):
        # normal_vec is the direction from p1 to p2
        v1 = plate_velocities[p1_indices]
        v2 = plate_velocities[p2_indices]
        # Relative velocity
        v_rel = v1 - v2
        # Projection onto normal: positive = convergence, negative = divergence
        return v_rel[:, 0] * normal_vec[0] + v_rel[:, 1] * normal_vec[1]

    # Horizontal boundaries (X-axis)
    mask_h = plate_map != shift_r
    tectonic_stress[mask_h] += calculate_boundary_stress(plate_map[mask_h], shift_r[mask_h], (0, 1))
    
    # Vertical boundaries (Y-axis)
    mask_v = plate_map != shift_d
    tectonic_stress[mask_v] += calculate_boundary_stress(plate_map[mask_v], shift_d[mask_v], (1, 0))

    # 4. Convert Stress to Elevation
    # Positive stress -> Mountains (folded crust)
    # Negative stress -> Trenches/Rifts
    print("Building tectonic elevation layers...")
    
    # Separate convergent and divergent
    mountains = np.maximum(0, tectonic_stress)
    trenches = np.minimum(0, tectonic_stress)
    
    # Spread the stress to create ranges (blur)
    # Mountains are wider, trenches are sharper
    mountains_smoothed = cv2.GaussianBlur(mountains, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    trenches_smoothed = cv2.GaussianBlur(trenches, (0, 0), sigmaX=blur_sigma/2, sigmaY=blur_sigma/2)
    
    # Normalize
    if mountains_smoothed.max() > 0:
        mountains_smoothed /= mountains_smoothed.max()
    if trenches_smoothed.min() < 0:
        trenches_smoothed /= np.abs(trenches_smoothed.min())
        
    # Combine: 0.7 base + 0.3 tectonic influence
    tectonic_h = (mountains_smoothed * 0.8) + (trenches_smoothed * 0.5)
    
    # Scale to 16-bit range for blending later
    # We return a [-1, 1] normalized float map
    return tectonic_h

def apply_tectonics_to_heightmap(base_h16, tectonic_influence=0.4):
    h, w = base_h16.shape
    t_map = generate_tectonic_heightmap(h, w)
    
    # Convert base to float [0, 1]
    h_float = base_h16.astype(np.float32) / 65535.0
    
    # Add tectonic features
    # Mountains add height, Trenches subtract
    h_final = h_float + (t_map * tectonic_influence)
    
    # Re-normalize to [0, 1] and back to 16-bit
    h_final = (h_final - h_final.min()) / (h_final.max() - h_final.min())
    return (h_final * 65535).astype(np.uint16)

if __name__ == "__main__":
    # Test
    test_h = np.zeros((512, 1024), dtype=np.uint16)
    result = apply_tectonics_to_heightmap(test_h)
    cv2.imwrite('debug_tectonic_map.png', (result // 256).astype(np.uint8))
