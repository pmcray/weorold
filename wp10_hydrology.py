import cv2
import numpy as np

def simulate_hydrology(height_map_16, mask_path='wp1_fractal_mask.png', rain_intensity=1.0, iterations=100):
    print("Simulating hydrological flow (rivers and lakes)...")
    
    h_float = height_map_16.astype(np.float32) / 65535.0
    height, width = h_float.shape
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros_like(h_float, dtype=np.uint8) + 255 # Assume all land
    is_land = mask > 127
    
    # 1. Flow Accumulation using Steepest Descent
    # We use a vectorized approach: each cell passes its "water" to its lowest neighbor.
    flow_map = np.full((height, width), rain_intensity, dtype=np.float32)
    # Only land gets rain for river formation
    flow_map[~is_land] = 0
    
    # Pad for neighbors
    h_padded = cv2.copyMakeBorder(h_float, 1, 1, 1, 1, cv2.BORDER_WRAP)
    
    print("Calculating drainage directions...")
    # Find lowest neighbor for each pixel
    # Neighbors: 
    # [(-1,-1), (-1,0), (-1,1),
    #  ( 0,-1),         ( 0,1),
    #  ( 1,-1), ( 1,0), ( 1,1)]
    
    # We do a few iterations of flow. In each iteration, water moves one step.
    # To get long rivers, we need many iterations or a recursive sort.
    # For a simple procedural effect, we'll do 50-100 steps.
    
    # Precompute neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # To speed up, we find the index of the lowest neighbor once
    neighbor_heights = []
    for dy, dx in offsets:
        # roll is slow in a loop, use slicing if possible, but roll handles BORDER_WRAP easily
        neighbor_heights.append(np.roll(np.roll(h_float, -dy, axis=0), -dx, axis=1))
        
    neighbor_heights = np.stack(neighbor_heights)
    lowest_neighbor_idx = np.argmin(neighbor_heights, axis=0)
    lowest_height = np.min(neighbor_heights, axis=0)
    
    # A cell only flows if the lowest neighbor is actually lower
    flow_dest_mask = lowest_height < h_float
    
    print(f"Propagating flow for {iterations} steps...")
    current_flow = flow_map.copy()
    accumulated_flow = np.zeros_like(flow_map)
    
    for i in range(iterations):
        # In each step, water moves to the precomputed destination
        next_flow = np.zeros_like(current_flow)
        
        for idx, (dy, dx) in enumerate(offsets):
            # Pixels that flow to neighbor 'idx'
            src_mask = (lowest_neighbor_idx == idx) & flow_dest_mask
            if np.any(src_mask):
                # Move water from current_flow[src_mask] to its neighbor
                # This is tricky to vectorize perfectly without loops over offsets
                # We can 'roll' the flow map back to its destination
                flow_to_move = current_flow * src_mask
                next_flow += np.roll(np.roll(flow_to_move, dy, axis=0), dx, axis=1)
        
        accumulated_flow += current_flow
        current_flow = next_flow
        
    # 2. Identify Rivers (High accumulated flow)
    # Log scale flow for better visualization
    river_intensity = np.log1p(accumulated_flow)
    if river_intensity.max() > 0:
        river_intensity /= river_intensity.max()
    
    # 3. Identify Sinks/Lakes
    # Sinks are where lowest_neighbor >= current_height
    sinks = ~flow_dest_mask & is_land
    # Simple lake: blur the sinks and add water depth
    lake_map = cv2.GaussianBlur(sinks.astype(np.float32), (0,0), sigmaX=3, sigmaY=3)
    
    return river_intensity, lake_map

def apply_hydrology_to_texture(texture_rgb, river_map, lake_map, river_color=[50, 100, 200], lake_color=[40, 80, 180]):
    print("Adding rivers and lakes to texture...")
    out = texture_rgb.astype(np.float32)
    
    # 1. Add Rivers
    # Only show rivers above a certain threshold
    river_threshold = 0.4
    river_mask = np.clip((river_map - river_threshold) / (1.0 - river_threshold), 0, 1)
    # Make rivers a bit wider for visibility
    river_mask = cv2.dilate(river_mask, np.ones((2,2), np.uint8))
    
    for i in range(3): # R, G, B
        out[:, :, i] = (1 - river_mask) * out[:, :, i] + river_mask * river_color[i]
        
    # 2. Add Lakes
    lake_mask = np.clip(lake_map * 2.0, 0, 1)
    for i in range(3):
        out[:, :, i] = (1 - lake_mask) * out[:, :, i] + lake_mask * lake_color[i]
        
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Test
    h = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)
    if h is not None:
        r, l = simulate_hydrology(h)
        cv2.imwrite('debug_rivers.png', (r * 255).astype(np.uint8))
