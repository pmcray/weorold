import cv2
import numpy as np
import os
from IPython.display import Image, Video, display

# Import all work packages
import wp1_fractal_coastline as wp1
import wp2_heightmap_synthesis as wp2
import wp3_biome_texturing as wp3
import wp4_cloud_layer as wp4
import wp5_final_renderer as wp5
import wp9_tectonics as wp9
import wp10_hydrology as wp10
import wp11_erosion as wp11
import weorold_ultra as globe

def run_full_pipeline(input_sketch='Motoki_Aspsp_uk_Fig02_c_Islands.jpg', output_name='aethelgard'):
    print(f"=== Starting Full Pipeline for {output_name} ===")
    
    # 1. WP1: Coastline Fractalization
    print("\n[Step 1] Fractalizing coastline...")
    wp1.process_sketch_to_fractal_mask(input_sketch, upscale_factor=4, noise_strength=20.0)
    # Output: wp1_fractal_mask.png
    
    # 2. WP2: Base Heightmap Synthesis
    print("\n[Step 2] Synthesizing base heightmap...")
    wp2.synthesize_heightmap('wp1_fractal_mask.png')
    # Output: wp2_height_map.png
    
    # 3. WP9: Tectonic Influence
    print("\n[Step 3] Applying tectonic features...")
    base_h16 = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)
    tectonic_h16 = wp9.apply_tectonics_to_heightmap(base_h16, tectonic_influence=0.3)
    cv2.imwrite(f'{output_name}_height_map_tectonic.png', tectonic_h16)
    
    # 4. WP11: Hydraulic Erosion
    print("\n[Step 4] Simulating hydraulic erosion...")
    # Using a moderate number of particles for performance in this pipeline
    eroded_h16 = wp11.simulate_hydraulic_erosion(tectonic_h16, num_particles=150000)
    cv2.imwrite(f'{output_name}_height_map.png', eroded_h16)
    
    # 5. WP10: Hydrology (Rivers & Lakes)
    print("\n[Step 5] Simulating hydrology...")
    river_map, lake_map = wp10.simulate_hydrology(eroded_h16, mask_path='wp1_fractal_mask.png')
    
    # 6. WP3: Biome & Surface Texturing
    print("\n[Step 6] Generating surface texture...")
    wp3.create_surface_texture(f'{output_name}_height_map.png', output_prefix=f'{output_name}_wp3')
    # Output: aethelgard_wp3_surface_texture.png
    
    # Apply rivers/lakes to the texture
    texture = cv2.imread(f'{output_name}_wp3_surface_texture.png')
    texture_rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    hydrology_texture = wp10.apply_hydrology_to_texture(texture_rgb, river_map, lake_map)
    cv2.imwrite(f'{output_name}_surface_texture_hydrology.png', cv2.cvtColor(hydrology_texture, cv2.COLOR_RGB2BGR))
    
    # 7. WP4: Cloud Layer
    print("\n[Step 7] Creating atmospheric clouds...")
    wp4.create_cloud_layer(eroded_h16.shape)
    # Output: wp4_cloud_map.png
    
    # 8. WP5: Final 2D Map Rendering (Shading)
    print("\n[Step 8] Rendering final shaded maps...")
    wp5.render_final_maps(
        texture_path=f'{output_name}_surface_texture_hydrology.png',
        heightmap_path=f'{output_name}_height_map.png',
        cloud_path='wp4_cloud_map.png',
        mask_path='wp1_fractal_mask.png',
        output_prefix=f'{output_name}_wp5'
    )
    # Output: aethelgard_wp5_shaded_clouds.png
    
    # 9. Final 3D Globe Animation
    print("\n[Step 9] Creating 3D rotating globe...")
    globe.create_rotating_globe(
        f'{output_name}_wp5_shaded_clouds.png', 
        f'{output_name}_orbit.mp4', 
        frames=120, 
        fps=30
    )
    
    print(f"\n=== Pipeline Complete! Result: {output_name}_orbit.mp4 ===")

if __name__ == "__main__":
    run_full_pipeline()
