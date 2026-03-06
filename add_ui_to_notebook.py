import nbformat as nbf
import os

nb_path = 'weorold_ultra.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

ui_code = """
import ipywidgets as widgets
from IPython.display import display, clear_output
import wp1_fractal_coastline as wp1
import wp2_heightmap_synthesis as wp2
import wp3_biome_texturing as wp3
import wp4_cloud_layer as wp4
import wp5_final_renderer as wp5
import wp6_random_seeding as wp6
import wp8_real_world_data as wp8
import weorold_ultra as globe
import cv2

# --- UI Definitions ---

# Main selection
source_dropdown = widgets.Dropdown(
    options=['Random Map', 'Input Sketch', 'Earth', 'Mars'],
    value='Random Map',
    description='Source:',
    style={'description_width': 'initial'}
)

# Random Map Sliders
water_slider = widgets.IntSlider(value=60, min=0, max=100, step=1, description='Hydrographic %:', style={'description_width': 'initial'})
consolidation_slider = widgets.IntSlider(value=5, min=1, max=10, step=1, description='Consolidation:', style={'description_width': 'initial'})

# Real World Sliders
sea_level_slider = widgets.FloatSlider(value=0.0, min=-0.5, max=0.5, step=0.01, description='Sea Level Offset:', style={'description_width': 'initial'})

# Generate Button
gen_button = widgets.Button(description="🚀 Generate Planet", button_style='success', layout=widgets.Layout(width='200px', height='40px'))
output = widgets.Output()

def update_ui(change):
    with output:
        clear_output()
        print("### World Generator Options ###")
        if source_dropdown.value == 'Random Map':
            display(widgets.VBox([water_slider, consolidation_slider]))
        elif source_dropdown.value == 'Earth' or source_dropdown.value == 'Mars':
            display(sea_level_slider)
        elif source_dropdown.value == 'Input Sketch':
            print("Note: Ensure 'input_sketch.png' is in the root directory.")
        display(gen_button)

source_dropdown.observe(update_ui, names='value')

def generate_planet(b):
    with output:
        clear_output()
        src = source_dropdown.value
        print(f"--- 🌎 Starting Generation from {src} ---")
        
        name = "Aethelgard" # Default random name
        h, w = 1024, 2048
        mask_path = 'world_mask.png'
        height_path = 'world_height_map.png'
        
        try:
            if src == 'Random Map':
                # Map water% to land_threshold: noise > threshold = land. 
                # So threshold = water% / 100.
                threshold = water_slider.value / 100.0
                # Map consolidation 1-10 to scale 50-1000
                scale = 50 + (consolidation_slider.value - 1) * (950 / 9)
                mask = wp6.generate_random_landmask(h, w, land_threshold=threshold, scale=scale)
                cv2.imwrite(mask_path, mask)
                wp2.synthesize_heightmap(mask_path, height_path)
            
            elif src == 'Earth':
                wp8.synthesize_real_world_map("Earth", wp8.create_earth_mask, sea_level_offset=sea_level_slider.value)
                mask_path = 'earth_mask.png'
                height_path = 'earth_height_map.png'
                name = "Earth"
                
            elif src == 'Mars':
                wp8.synthesize_real_world_map("Mars", wp8.create_mars_mask, sea_level_offset=sea_level_slider.value)
                mask_path = 'mars_mask.png'
                height_path = 'mars_height_map.png'
                name = "Mars"
                
            elif src == 'Input Sketch':
                sketch_file = 'input_sketch.png'
                if not os.path.exists(sketch_file):
                    # Try alternate name from debug
                    if os.path.exists('Motoki_Aspsp_uk_Fig02_c_Islands.jpg'):
                        sketch_file = 'Motoki_Aspsp_uk_Fig02_c_Islands.jpg'
                    else:
                        print(f"Error: {sketch_file} not found. Please place it in the same folder.")
                        return
                wp1.process_sketch_to_fractal_mask(sketch_file, upscale_factor=4)
                mask_path = 'wp1_fractal_mask.png'
                wp2.synthesize_heightmap(mask_path, height_path)
                
            # Full Pipeline
            print("🎨 Texturing and final rendering...")
            wp3.create_surface_texture(height_path, mask_path, output_prefix=f'{name.lower()}_wp3')
            wp4.create_cloud_layer((h, w), output_path='wp4_cloud_map.png')
            wp5.render_final_maps(
                texture_path=f'{name.lower()}_wp3_surface_texture.png',
                heightmap_path=height_path,
                mask_path=mask_path,
                output_prefix=f'{name.lower()}_wp5'
            )
            
            # Show result
            final_texture = f'{name.lower()}_wp5_shaded_clouds.png'
            show_img(final_texture, title="Final Generated Map", width=1000)
            
            # Generate 3D animation
            print("🎬 Rendering 3D animation (ultra-realistic globe)...")
            video_path = f'{name.lower()}_rotation.mp4'
            globe.create_rotating_globe(final_texture, video_path, frames=60, fps=20)
            display(Video(video_path, embed=True, width=600))
            print(f"--- ✅ Done! Video saved as {video_path} ---")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred during generation: {e}")

gen_button.on_click(generate_planet)

display(widgets.VBox([
    widgets.HTML("<h2>World Generator Interface</h2>"),
    source_dropdown,
    output
]))
update_ui(None)
"""

# Insert the UI cell at index 2 (after Environment Ready)
new_cell = nbf.v4.new_code_cell(ui_code.strip())
# Ensure it doesn't already have a UI cell (prevent duplicates if script run multiple times)
if not any("World Generator Interface" in str(c.source) for c in nb.cells):
    nb.cells.insert(2, new_cell)
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("UI cell added to notebook.")
else:
    print("UI cell already exists.")
