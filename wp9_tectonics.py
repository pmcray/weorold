import cv2
import numpy as np

def generate_tectonic_evolution(base_h16, num_plates=15, time_steps=5, tectonic_influence=0.4, blur_sigma=20):
    print(f"Simulating {time_steps} epochs of tectonic evolution with {num_plates} plates...")
    
    # Convert base to float [0, 1]
    h_float = base_h16.astype(np.float32) / 65535.0
    height, width = h_float.shape
    
    # 1. Generate Plate Centers and Initial Voronoi
    points = np.column_stack([
        np.random.randint(0, height, num_plates),
        np.random.randint(0, width, num_plates)
    ])
    
    y, x = np.indices((height, width))
    plate_map = np.zeros((height, width), dtype=np.int32)
    
    min_dist = np.full((height, width), np.inf)
    for i, p in enumerate(points):
        dx = np.abs(x - p[1])
        dx = np.minimum(dx, width - dx)
        dy = y - p[0]
        dist_sq = dx**2 + dy**2
        mask = dist_sq < min_dist
        min_dist[mask] = dist_sq[mask]
        plate_map[mask] = i

    # 2. Assign Velocities to Plates
    plate_velocities = np.random.uniform(-2, 2, (num_plates, 2)) # (dy, dx)
    
    # Evolutionary loop
    for step in range(time_steps):
        print(f"  Epoch {step+1}/{time_steps}...")
        
        # Calculate Convergence/Divergence at boundaries
        shift_r = np.roll(plate_map, 1, axis=1)
        shift_d = np.roll(plate_map, 1, axis=0)
        
        tectonic_stress = np.zeros((height, width), dtype=np.float32)
        
        def calculate_stress(p1_idx, p2_idx, normal):
            v_rel = plate_velocities[p1_idx] - plate_velocities[p2_idx]
            return v_rel[:, 0] * normal[0] + v_rel[:, 1] * normal[1]

        mask_h = plate_map != shift_r
        tectonic_stress[mask_h] += calculate_stress(plate_map[mask_h], shift_r[mask_h], (0, 1))
        
        mask_v = plate_map != shift_d
        tectonic_stress[mask_v] += calculate_stress(plate_map[mask_v], shift_d[mask_v], (1, 0))

        # Separate convergent and divergent
        mountains = np.maximum(0, tectonic_stress)
        trenches = np.minimum(0, tectonic_stress)
        
        mountains_smoothed = cv2.GaussianBlur(mountains, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        trenches_smoothed = cv2.GaussianBlur(trenches, (0, 0), sigmaX=blur_sigma/2, sigmaY=blur_sigma/2)
        
        if mountains_smoothed.max() > 0:
            mountains_smoothed /= mountains_smoothed.max()
        if trenches_smoothed.min() < 0:
            trenches_smoothed /= np.abs(trenches_smoothed.min())
            
        epoch_h = (mountains_smoothed * 0.8) + (trenches_smoothed * 0.5)
        
        # Apply stress to heightmap incrementally
        h_float += epoch_h * (tectonic_influence / time_steps)
        
        # Erode mountains slightly each epoch (Sedimentation/Deltas preview)
        erosion_mask = h_float > 0.6
        h_float[erosion_mask] *= 0.98 # Wind/thermal erosion over epochs
        
        # Move plate centers
        points[:, 0] = (points[:, 0] + plate_velocities[:, 0]).astype(int) % height
        points[:, 1] = (points[:, 1] + plate_velocities[:, 1]).astype(int) % width
        
        # We don't recalculate the full voronoi every step to save time, 
        # but in a true sim we would. Here we approximate by blurring the boundary stress over time.
        blur_sigma = max(5, blur_sigma - 2)

    # Re-normalize to [0, 1] and back to 16-bit
    h_float = (h_float - h_float.min()) / (h_float.max() - h_float.min() + 1e-8)
    return (h_float * 65535).astype(np.uint16)

def apply_tectonics_to_heightmap(base_h16, tectonic_influence=0.4):
    return generate_tectonic_evolution(base_h16, time_steps=5, tectonic_influence=tectonic_influence)

if __name__ == "__main__":
    test_h = np.zeros((512, 1024), dtype=np.uint16)
    result = apply_tectonics_to_heightmap(test_h)
    cv2.imwrite('debug_tectonic_map.png', (result // 256).astype(np.uint8))
