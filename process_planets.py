from wp3_biome_texturing import create_surface_texture
from wp5_final_renderer import render_final_maps

def process_planet(name):
    print(f"\n--- Processing {name} ---")
    heightmap = f'{name.lower()}_height_map.png'
    mask = f'{name.lower()}_mask.png'
    
    # 1. Create Texture
    create_surface_texture(heightmap, mask, output_prefix=f'{name.lower()}_wp3')
    
    # 2. Final Render (2D)
    render_final_maps(
        texture_path=f'{name.lower()}_wp3_surface_texture.png',
        heightmap_path=heightmap,
        mask_path=mask,
        output_prefix=f'{name.lower()}_wp5'
    )

if __name__ == "__main__":
    process_planet("Earth")
    process_planet("Mars")
