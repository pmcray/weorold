import cv2
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not found. wp11_erosion_gpu will not function. Install with 'pip install cupy-cuda12x' (or your CUDA version).")

def simulate_hydraulic_erosion_gpu(height_map_16, num_particles=1000000, max_lifetime=30, inertia=0.1, 
                                  sediment_capacity_factor=4, min_sediment_capacity=0.01, 
                                  erode_speed=0.3, deposit_speed=0.3, evaporate_speed=0.01, gravity=4):
    """
    Massively parallel hydraulic erosion using CuPy.
    Processes all particles simultaneously on the GPU.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for GPU erosion. Please use the CPU version or install CuPy.")

    print(f"Simulating GPU hydraulic erosion with {num_particles} particles...")
    
    # 1. Transfer data to GPU
    h_gpu = cp.asarray(height_map_16.astype(np.float32) / 65535.0)
    height, width = h_gpu.shape
    
    # 2. Initialize particles on GPU
    # State: pos_x, pos_y, dir_x, dir_y, speed, water, sediment, active_mask
    pos_x = cp.random.uniform(0, width - 1, num_particles)
    pos_y = cp.random.uniform(0, height - 1, num_particles)
    dir_x = cp.zeros(num_particles)
    dir_y = cp.zeros(num_particles)
    speed = cp.ones(num_particles)
    water = cp.ones(num_particles)
    sediment = cp.zeros(num_particles)
    active = cp.ones(num_particles, dtype=bool)

    # Pre-calculated constants
    inertia_c = float(inertia)
    inv_inertia_c = 1.0 - inertia_c
    
    for i in range(max_lifetime):
        # Only process active particles
        if not cp.any(active):
            break
            
        # Bilinear setup
        node_x = pos_x.astype(cp.int32)
        node_y = pos_y.astype(cp.int32)
        px = pos_x - node_x
        py = pos_y - node_y
        
        # Wrapping coordinates
        x0 = node_x % width
        x1 = (node_x + 1) % width
        y0 = cp.clip(node_y, 0, height - 1)
        y1 = cp.clip(node_y + 1, 0, height - 1)
        
        # Sample heights
        h00 = h_gpu[y0, x0]
        h10 = h_gpu[y0, x1]
        h01 = h_gpu[y1, x0]
        h11 = h_gpu[y1, x1]
        
        # Gradient & Height
        gx = (h10 - h00) * (1 - py) + (h11 - h01) * py
        gy = (h01 - h00) * (1 - px) + (h11 - h10) * px
        h_old = h00 * (1 - px) * (1 - py) + h10 * px * (1 - py) + h01 * (1 - px) * py + h11 * px * py
        
        # Update direction
        dir_x = (dir_x * inertia_c - gx * inv_inertia_c)
        dir_y = (dir_y * inertia_c - gy * inv_inertia_c)
        
        # Normalize
        mag = cp.sqrt(dir_x**2 + dir_y**2)
        mask_mag = (mag != 0) & active
        dir_x[mask_mag] /= mag[mask_mag]
        dir_y[mask_mag] /= mag[mask_mag]
        
        # Update position
        pos_x[active] += dir_x[active]
        pos_y[active] += dir_y[active]
        
        # Bounds check
        active &= (pos_y >= 0) & (pos_y < height - 1)
        if not cp.any(active): break

        # New Height
        nx0 = pos_x.astype(cp.int32) % width
        nx1 = (pos_x.astype(cp.int32) + 1) % width
        ny0 = cp.clip(pos_y.astype(cp.int32), 0, height - 1)
        ny1 = cp.clip(pos_y.astype(cp.int32) + 1, 0, height - 1)
        npx = pos_x - pos_x.astype(cp.int32)
        npy = pos_y - pos_y.astype(cp.int32)
        
        h_new = h_gpu[ny0, nx0] * (1 - npx) * (1 - npy) + \
                h_gpu[ny0, nx1] * npx * (1 - npy) + \
                h_gpu[ny1, nx0] * (1 - npx) * npy + \
                h_gpu[ny1, nx1] * npx * npy
        
        delta_h = h_new - h_old
        
        # Capacity logic
        capacity = cp.maximum(-delta_h, min_sediment_capacity) * speed * water * sediment_capacity_factor
        
        # 3. Apply Erosion / Deposition (This uses atomic adds on GPU)
        deposit_mask = (sediment > capacity) | (delta_h > 0)
        
        # Deposition
        to_deposit = cp.where(delta_h > 0, cp.minimum(delta_h, sediment), (sediment - capacity) * deposit_speed)
        d_amount = to_deposit * active * deposit_mask
        
        # Use cp.add.at for atomic updates (equivalent to scatter_add)
        cp.add.at(h_gpu, (y0, x0), d_amount * (1 - px) * (1 - py))
        cp.add.at(h_gpu, (y0, x1), d_amount * px * (1 - py))
        cp.add.at(h_gpu, (y1, x0), d_amount * (1 - px) * py)
        cp.add.at(h_gpu, (y1, x1), d_amount * px * py)
        sediment[deposit_mask] -= to_deposit[deposit_mask]
        
        # Erosion
        erode_mask = active & (~deposit_mask)
        to_erode = cp.minimum((capacity - sediment) * erode_speed, -delta_h)
        e_amount = to_erode * erode_mask
        
        cp.add.at(h_gpu, (y0, x0), -e_amount * (1 - px) * (1 - py))
        cp.add.at(h_gpu, (y1, x1), -e_amount * px * py)
        cp.add.at(h_gpu, (y0, x1), -e_amount * px * (1 - py))
        cp.add.at(h_gpu, (y1, x0), -e_amount * (1 - px) * py)
        sediment[erode_mask] += to_erode[erode_mask]
        
        # Update speed and water
        speed = cp.sqrt(cp.maximum(0, speed**2 + delta_h * gravity))
        water *= (1 - evaporate_speed)
        active &= (speed > 0)

    # 4. Final normalization and return to CPU
    h_final = (h_gpu - h_gpu.min()) / (h_gpu.max() - h_gpu.min())
    return (cp.asnumpy(h_final) * 65535).astype(np.uint16)

if __name__ == "__main__":
    # Setup test
    h16 = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)
    if h16 is not None:
        eroded = simulate_hydraulic_erosion_gpu(h16, num_particles=1000000)
        cv2.imwrite('debug_eroded_map_gpu.png', (eroded // 256).astype(np.uint8))
