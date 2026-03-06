import argparse
import cv2
from wp6_random_seeding import generate_random_landmask
from wp2_heightmap_synthesis import synthesize_heightmap
from wp3_biome_texturing import create_surface_texture
from wp5_final_renderer import render_final_maps

def main():
    parser = argparse.ArgumentParser(description="Weorold Interactive Planet Generator")
    parser.add_argument("--name", type=str, default="InteractiveWorld", help="Name of the planet")
    parser.add_argument("--hydro", type=float, default=0.7, help="Hydrographic percentage (0.0 to 1.0)")
    parser.add_argument("--roughness", type=float, default=0.5, help="Terrain roughness (0.0 to 1.0)")
    parser.add_argument("--scale", type=float, default=200.0, help="Noise scale for continents")
    
    args = parser.parse_args()
    
    # Map hydrographic percentage to land threshold
    # 0% hydro -> 100% land (threshold 0)
    # 100% hydro -> 0% land (threshold 1)
    # But noise is normalized 0-1, so we need to find the right threshold.
    # Usually a threshold of 0.5 gives roughly 50% land.
    # We'll use a linear mapping as a starting point.
    land_threshold = args.hydro
    
    print(f"\n--- Generating {args.name} ---")
    print(f"Hydrographic: {args.hydro*100}% | Scale: {args.scale}")
    
    h, w = 1024, 2048
    
    # 1. Seeding
    mask = generate_random_landmask(h, w, land_threshold=land_threshold, scale=args.scale)
    mask_path = f'{args.name.lower()}_mask.png'
    cv2.imwrite(mask_path, mask)
    
    # 2. Heightmap
    height_path = f'{args.name.lower()}_height_map.png'
    synthesize_heightmap(mask_path, height_path)
    
    # 3. Biomes
    create_surface_texture(height_path, mask_path, output_prefix=f'{args.name.lower()}_wp3')
    
    # 4. Final Render
    render_final_maps(
        texture_path=f'{args.name.lower()}_wp3_surface_texture.png',
        heightmap_path=height_path,
        output_prefix=f'{args.name.lower()}_wp5'
    )
    
    print(f"Planet {args.name} generated successfully.")

if __name__ == "__main__":
    main()
