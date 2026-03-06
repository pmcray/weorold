import cv2
from wp6_random_seeding import generate_random_landmask
from wp2_heightmap_synthesis import synthesize_heightmap
from wp3_biome_texturing import create_surface_texture
from wp5_final_renderer import render_final_maps

def generate_random_planet(planet_name="RandomWorld", land_threshold=0.55):
    print(f"\n--- Generating Random Planet: {planet_name} ---")
    h, w = 1024, 2048
    
    # 1. WP6: Random Seeding
    mask = generate_random_landmask(h, w, land_threshold=land_threshold)
    mask_path = f'{planet_name.lower()}_mask.png'
    cv2.imwrite(mask_path, mask)
    
    # 2. WP2: Heightmap Synthesis
    height_path = f'{planet_name.lower()}_height_map.png'
    synthesize_heightmap(mask_path, height_path)
    
    # 3. WP3: Biome Texturing
    create_surface_texture(height_path, mask_path, output_prefix=f'{planet_name.lower()}_wp3')
    
    # 4. Final Render (2D)
    render_final_maps(
        texture_path=f'{planet_name.lower()}_wp3_surface_texture.png',
        heightmap_path=height_path,
        mask_path=mask_path,
        output_prefix=f'{planet_name.lower()}_wp5'
    )
    
    print(f"Random planet {planet_name} generated successfully.")

if __name__ == "__main__":
    generate_random_planet("Aethelgard", land_threshold=0.6)
