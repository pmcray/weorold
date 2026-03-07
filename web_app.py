import os
import cv2
from fastapi import FastAPI, BackgroundTasks, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio

# Import pipeline
import wp1_fractal_coastline as wp1
import wp2_heightmap_synthesis as wp2
import wp3_biome_texturing as wp3
import wp4_cloud_layer as wp4
import wp5_final_renderer as wp5
import wp9_tectonics as wp9
import wp10_hydrology as wp10
try:
    import wp11_erosion_gpu as wp11_gpu
    USE_GPU = True
except ImportError:
    import wp11_erosion as wp11
    USE_GPU = False
import weorold_ultra as globe

app = FastAPI(title="Weorold Generator API")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def run_pipeline(planet_name: str, erosion_particles: int, tectonic_influence: float, temperature: float, moisture: float):
    print(f"Starting generation for {planet_name}")
    wp1.process_sketch_to_fractal_mask(None, upscale_factor=4, noise_strength=20.0)
    wp2.synthesize_heightmap('wp1_fractal_mask.png')
    
    base_h16 = cv2.imread('wp2_height_map.png', cv2.IMREAD_UNCHANGED)
    tectonic_h16 = wp9.apply_tectonics_to_heightmap(base_h16, tectonic_influence=tectonic_influence)
    
    if USE_GPU:
        eroded_h16 = wp11_gpu.simulate_hydraulic_erosion_gpu(tectonic_h16, num_particles=erosion_particles)
    else:
        eroded_h16 = wp11.simulate_hydraulic_erosion(tectonic_h16, num_particles=150000)
        
    cv2.imwrite(f'static/{planet_name}_height_map.png', eroded_h16)
    
    river_map, lake_map = wp10.simulate_hydrology(eroded_h16, mask_path='wp1_fractal_mask.png')
    
    wp3.create_surface_texture(f'static/{planet_name}_height_map.png', temperature=temperature, moisture=moisture, output_prefix=f'static/{planet_name}_wp3')
    texture = cv2.imread(f'static/{planet_name}_wp3_surface_texture.png')
    texture_rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    hydrology_texture = wp10.apply_hydrology_to_texture(texture_rgb, river_map, lake_map)
    cv2.imwrite(f'static/{planet_name}_final_texture.png', cv2.cvtColor(hydrology_texture, cv2.COLOR_RGB2BGR))
    
    wp4.create_cloud_layer(eroded_h16.shape)
    wp5.render_final_maps(
        texture_path=f'static/{planet_name}_final_texture.png',
        heightmap_path=f'static/{planet_name}_height_map.png',
        cloud_path='wp4_cloud_map.png',
        mask_path='wp1_fractal_mask.png',
        output_prefix=f'static/{planet_name}_wp5'
    )
    
    globe.create_rotating_globe(f'static/{planet_name}_wp5_shaded_clouds.png', f'static/{planet_name}_orbit.mp4', frames=60, fps=30)
    print(f"Generation for {planet_name} complete.")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return """
    <html>
        <head><title>Weorold Generator</title></head>
        <body>
            <h1>Weorold Planet Generator</h1>
            <form action="/generate" method="post">
                <label>Planet Name:</label> <input type="text" name="planet_name" value="earth2"><br><br>
                <label>Erosion Particles:</label> <input type="number" name="erosion_particles" value="500000"><br><br>
                <label>Tectonic Influence:</label> <input type="number" step="0.1" name="tectonic_influence" value="0.5"><br><br>
                <label>Temperature:</label> <input type="number" step="0.1" name="temperature" value="0.5"><br><br>
                <label>Moisture:</label> <input type="number" step="0.1" name="moisture" value="0.5"><br><br>
                <input type="submit" value="Generate Planet">
            </form>
        </body>
    </html>
    """

@app.post("/generate")
async def generate_planet(
    background_tasks: BackgroundTasks,
    planet_name: str = Form(...),
    erosion_particles: int = Form(...),
    tectonic_influence: float = Form(...),
    temperature: float = Form(...),
    moisture: float = Form(...)
):
    background_tasks.add_task(run_pipeline, planet_name, erosion_particles, tectonic_influence, temperature, moisture)
    return HTMLResponse(f"Generation started for {planet_name}. Check back later at <a href='/static/{planet_name}_orbit.mp4'>Video</a>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
