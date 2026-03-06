import cv2
import numpy as np
import random

def simulate_hydraulic_erosion(height_map_16, num_particles=50000, max_lifetime=30, inertia=0.1, 
                              sediment_capacity_factor=4, min_sediment_capacity=0.01, 
                              erode_speed=0.3, deposit_speed=0.3, evaporate_speed=0.01, gravity=4):
    
    print(f"Simulating hydraulic erosion with {num_particles} particles...")
    
    # Work in float [0, 1] for precision
    h_float = height_map_16.astype(np.float32) / 65535.0
    height, width = h_float.shape
    
    # Pre-calculate gradients for efficiency (Sobel)
    # We'll update the heightmap but recalculating gradients every particle is too slow.
    # We'll recalculate gradients in batches or just use local sampling.
    # Local sampling (4-pixel) is better for particle movement.
    
    def get_height_and_gradient(x, y):
        # Bilinear interpolation
        x_i = int(x)
        y_i = int(y)
        u = x - x_i
        v = y - y_i
        
        # Wrapping coordinates
        x0 = x_i % width
        x1 = (x_i + 1) % width
        y0 = np.clip(y_i, 0, height - 1)
        y1 = np.clip(y_i + 1, 0, height - 1)
        
        h00 = h_float[y0, x0]
        h10 = h_float[y0, x1]
        h01 = h_float[y1, x0]
        h11 = h_float[y1, x1]
        
        # Gradient
        gx = (h10 - h00) * (1 - v) + (h11 - h01) * v
        gy = (h01 - h00) * (1 - u) + (h11 - h10) * u
        
        # Height
        h = h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) + h01 * (1 - u) * v + h11 * u * v
        
        return h, gx, gy

    # Particle Simulation Loop
    for p in range(num_particles):
        if p % 10000 == 0 and p > 0:
            print(f"  Processed {p} particles...")
            
        # Random start position
        posX = random.uniform(0, width - 1)
        posY = random.uniform(0, height - 1)
        dirX = 0
        dirY = 0
        speed = 1
        water = 1
        sediment = 0
        
        for i in range(max_lifetime):
            nodeX = int(posX)
            nodeY = int(posY)
            
            # Offset within cell
            px = posX - nodeX
            py = posY - nodeY
            
            # Get height and gradient at current position
            h_old, gx, gy = get_height_and_gradient(posX, posY)
            
            # Update direction and position
            dirX = (dirX * inertia - gx * (1 - inertia))
            dirY = (dirY * inertia - gy * (1 - inertia))
            
            # Normalize direction
            mag = np.sqrt(dirX**2 + dirY**2)
            if mag != 0:
                dirX /= mag
                dirY /= mag
            
            posX += dirX
            posY += dirY
            
            # Stop if out of bounds (Y only, X wraps)
            if posY < 0 or posY >= height - 1:
                break
            
            # New height
            h_new, _, _ = get_height_and_gradient(posX, posY)
            delta_h = h_new - h_old
            
            # Calculate sediment capacity
            capacity = max(-delta_h, min_sediment_capacity) * speed * water * sediment_capacity_factor
            
            if sediment > capacity or delta_h > 0:
                # Deposit
                to_deposit = (sediment - capacity) * deposit_speed if delta_h < 0 else min(delta_h, sediment)
                sediment -= to_deposit
                # Distribute deposit to 4 neighbors
                x0, y0 = nodeX % width, int(np.clip(nodeY, 0, height - 1))
                x1, y1 = (nodeX + 1) % width, int(np.clip(nodeY + 1, 0, height - 1))
                h_float[y0, x0] += to_deposit * (1 - px) * (1 - py)
                h_float[y0, x1] += to_deposit * px * (1 - py)
                h_float[y1, x0] += to_deposit * (1 - px) * py
                h_float[y1, x1] += to_deposit * px * py
            else:
                # Erode
                to_erode = min((capacity - sediment) * erode_speed, -delta_h)
                sediment += to_erode
                # Distribute erosion to 4 neighbors
                x0, y0 = nodeX % width, int(np.clip(nodeY, 0, height - 1))
                x1, y1 = (nodeX + 1) % width, int(np.clip(nodeY + 1, 0, height - 1))
                h_float[y0, x0] -= to_erode * (1 - px) * (1 - py)
                h_float[y0, x1] -= to_erode * px * (1 - py)
                h_float[y1, x0] -= to_erode * (1 - px) * py
                h_float[y1, x1] -= to_erode * px * py
                
            # Update speed and water
            speed = np.sqrt(max(0, speed**2 + delta_h * gravity))
            water *= (1 - evaporate_speed)
            
            if speed == 0:
                break
                
    # Re-normalize and return
    h_final = (h_float - h_float.min()) / (h_float.max() - h_float.min())
    return (h_final * 65535).astype(np.uint16)

if __name__ == "__main__":
    # Test on existing heightmap
    h16 = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)
    if h16 is not None:
        eroded = simulate_hydraulic_erosion(h16, num_particles=100000)
        cv2.imwrite('debug_eroded_map.png', (eroded // 256).astype(np.uint8))
