import json

def make_cell(source_list, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source_list[:-1]] + [source_list[-1]] if source_list else []
    }

cells = []

# Header
cells.append(make_cell(["# Ultra-Realistic Alien World Generator (Full Pipeline)", "", "This notebook guides you through the process of converting a rough sketch into a photorealistic 3D alien planet, incorporating tectonics, hydraulic erosion, and hydrology."], "markdown"))

# Environment Setup
cells.append(make_cell([
    "import cv2",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "from IPython.display import Image, Video, display",
    "import os",
    "",
    "import wp1_fractal_coastline as wp1",
    "import wp2_heightmap_synthesis as wp2",
    "import wp3_biome_texturing as wp3",
    "import wp4_cloud_layer as wp4",
    "import wp5_final_renderer as wp5",
    "import wp6_random_seeding as wp6",
    "import wp8_real_world_data as wp8",
    "import wp9_tectonics as wp9",
    "import wp10_hydrology as wp10",
    "import wp11_erosion as wp11",
    "import wp12_full_pipeline as pipeline",
    "import weorold_ultra as globe",
    "",
    "def show_img(path, title=None, width=None):",
    "    if title: print(f'### {title} ###')",
    "    display(Image(filename=path, width=width))",
    "",
    "print('Environment Ready.')"
]))

# Random Generation
cells.append(make_cell(["## Option A: Random World Generation", "Don't have a sketch? Generate a random landmask using fractal noise."], "markdown"))
cells.append(make_cell([
    "RANDOM_NAME = 'Novus'",
    "random_mask = wp6.generate_random_landmask(1024, 2048, land_threshold=0.55, scale=250.0)",
    "cv2.imwrite('novus_mask.png', random_mask)",
    "show_img('novus_mask.png', title='Randomly Generated Landmask', width=800)"
]))

# Earth & Mars
cells.append(make_cell(["## Option B: Real-World Planet Synthesis", "Synthesize Earth (with sea level rise) or Mars (as a 'Green Mars')."], "markdown"))
cells.append(make_cell([
    "# Synthesize 'Drowned Earth'",
    "wp8.synthesize_real_world_map('Earth', wp8.create_earth_mask, sea_level_offset=0.1)",
    "show_img('earth_height_map.png', title='Earth Heightmap (Synthesized)', width=600)",
    "",
    "# Synthesize 'Green Mars'",
    "wp8.synthesize_real_world_map('Mars', wp8.create_mars_mask, sea_level_offset=-0.2)",
    "show_img('mars_height_map.png', title='Mars Heightmap (Synthesized)', width=600)"
]))

# Full Pipeline with Sketch
cells.append(make_cell(["## Option C: Sketch-to-Planet Pipeline", "The original workflow: fractalize a hand-drawn sketch and turn it into a world."], "markdown"))

# WP1
cells.append(make_cell(["### Step 1: Coastline Fractalization", "We take the input sketch and use a Signed Distance Field (SDF) displaced by fractal noise."], "markdown"))
cells.append(make_cell([
    "INPUT_SKETCH = 'Motoki_Aspsp_uk_Fig02_c_Islands.jpg'",
    "wp1.process_sketch_to_fractal_mask(INPUT_SKETCH, upscale_factor=4, noise_strength=20.0)",
    "show_img('wp1_comparison.png', title='Original Sketch vs Fractal Coastline', width=800)"
]))

# WP2 & WP9
cells.append(make_cell(["### Step 2: Heightmap & Tectonics", "Synthesizing base elevation and adding tectonic plates."], "markdown"))
cells.append(make_cell([
    "wp2.synthesize_heightmap('wp1_fractal_mask.png')",
    "base_h16 = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)",
    "tectonic_h16 = wp9.apply_tectonics_to_heightmap(base_h16, tectonic_influence=0.3)",
    "cv2.imwrite('debug_height_tectonic.png', (tectonic_h16 // 256).astype(np.uint8))",
    "show_img('debug_height_tectonic.png', title='Heightmap with Tectonics', width=600)"
]))

# WP11
cells.append(make_cell(["### Step 3: Hydraulic Erosion", "Simulating thousands of raindrops to carve realistic river valleys and mountain ridges."], "markdown"))
cells.append(make_cell([
    "eroded_h16 = wp11.simulate_hydraulic_erosion(tectonic_h16, num_particles=100000)",
    "cv2.imwrite('aethelgard_height_map.png', eroded_h16)",
    "show_img('aethelgard_height_map.png', title='Heightmap after Hydraulic Erosion', width=600)"
]))

# WP10 & WP3
cells.append(make_cell(["### Step 4: Hydrology & Biomes", "Simulating water flow and assigning realistic colors based on moisture and temperature."], "markdown"))
cells.append(make_cell([
    "river_map, lake_map = wp10.simulate_hydrology(eroded_h16, mask_path='wp1_fractal_mask.png')",
    "wp3.create_surface_texture('aethelgard_height_map.png', output_prefix='aethelgard_wp3')",
    "",
    "texture = cv2.imread('aethelgard_wp3_surface_texture.png')",
    "texture_rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)",
    "hydrology_texture = wp10.apply_hydrology_to_texture(texture_rgb, river_map, lake_map)",
    "cv2.imwrite('aethelgard_surface_texture_hydrology.png', cv2.cvtColor(hydrology_texture, cv2.COLOR_RGB2BGR))",
    "",
    "show_img('aethelgard_surface_texture_hydrology.png', title='Surface Texture with Hydrology', width=800)"
]))

# WP4 & WP5
cells.append(make_cell(["### Step 5: Clouds & Final Rendering", "Generating atmospheric clouds and applying Phong shading for 3D depth."], "markdown"))
cells.append(make_cell([
    "wp4.create_cloud_layer(eroded_h16.shape)",
    "wp5.render_final_maps(",
    "    texture_path='aethelgard_surface_texture_hydrology.png',",
    "    heightmap_path='aethelgard_height_map.png',",
    "    cloud_path='wp4_cloud_map.png',",
    "    mask_path='wp1_fractal_mask.png',",
    "    output_prefix='aethelgard_wp5'",
    ")",
    "show_img('aethelgard_wp5_shaded_clouds.png', title='Final Shaded 2D Map', width=800)"
]))

# Final Result
cells.append(make_cell(["## Final Result: 3D Rotating Globe", "Projecting the final textures onto a 3D sphere with smooth rotation."], "markdown"))
cells.append(make_cell([
    "globe.create_rotating_globe('aethelgard_wp5_shaded_clouds.png', 'aethelgard_orbit.mp4', frames=60, fps=20)",
    "print('\\nDisplaying Final Globe Animation:')",
    "display(Video('aethelgard_orbit.mp4', embed=True, width=600))"
]))

# Zoom / Interactive Section
cells.append(make_cell(["## Advanced: Zoom & Field of View", "Adjust the FOV (zoom) by modifying the projection parameters."], "markdown"))
cells.append(make_cell([
    "# Adjust fov_scale (smaller = more zoomed in)",
    "def render_zoomed_globe(texture_path, output_path, zoom=1.0):",
    "    # Note: This is a simplified call to show how zoom (fov_scale) would work",
    "    print(f'Rendering globe with zoom factor {zoom}...')",
    "    # In weorold_ultra.py, you can change fov_scale = 1.2 / zoom",
    "    globe.create_rotating_globe(texture_path, output_path, frames=30, fps=15)",
    "",
    "render_zoomed_globe('aethelgard_wp5_shaded_clouds.png', 'aethelgard_zoom.mp4', zoom=2.0)",
    "display(Video('aethelgard_zoom.mp4', embed=True, width=600))"
]))

nb = {
 "cells": cells,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.8.5"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('weorold_ultra.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook repaired with full pipeline, random generation, and real-world data.")
