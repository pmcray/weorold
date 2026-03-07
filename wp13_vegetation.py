import cv2
import numpy as np

def generate_vegetation_scatter(biome_path, temp_path, moisture_path, output_prefix='wp13'):
    """
    Generates high-resolution density maps for different flora types based on biomes.
    """
    print("Generating vegetation scatter maps...")
    try:
        biome = cv2.imread(biome_path)
        temp_map = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        moisture_map = cv2.imread(moisture_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error loading maps: {e}. Please ensure wp3 biome maps are generated first.")
        return

    height, width, _ = biome.shape
    
    # We will output density maps for:
    # 1. Broadleaf Trees
    # 2. Conifer Trees
    # 3. Shrubs / Grass
    # 4. Cacti / Desert Flora

    density_broadleaf = np.zeros((height, width), dtype=np.float32)
    density_conifer = np.zeros((height, width), dtype=np.float32)
    density_shrub = np.zeros((height, width), dtype=np.float32)
    density_cactus = np.zeros((height, width), dtype=np.float32)

    # Base rules:
    # High temp, High moisture -> Broadleaf
    # Low temp, High/Med moisture -> Conifer
    # Mid moisture -> Shrubs
    # High temp, Low moisture -> Cacti

    density_broadleaf = np.clip((temp_map - 0.5) * 2 * moisture_map, 0, 1)
    density_conifer = np.clip((0.6 - temp_map) * 2 * moisture_map, 0, 1)
    density_shrub = np.clip(np.sin(moisture_map * np.pi) * temp_map, 0, 1)
    density_cactus = np.clip((temp_map - 0.6) * 2 * (1.0 - moisture_map), 0, 1)

    # Add random noise to make it scattered
    noise_broadleaf = np.random.uniform(0.5, 1.0, (height, width))
    noise_conifer = np.random.uniform(0.5, 1.0, (height, width))
    noise_shrub = np.random.uniform(0.2, 1.0, (height, width))
    noise_cactus = np.random.uniform(0.8, 1.0, (height, width))

    density_broadleaf *= noise_broadleaf
    density_conifer *= noise_conifer
    density_shrub *= noise_shrub
    density_cactus *= noise_cactus

    # Thresholding for scatter maps (pixels represent flora placement)
    density_broadleaf = np.where(density_broadleaf > 0.8, 255, 0)
    density_conifer = np.where(density_conifer > 0.8, 255, 0)
    density_shrub = np.where(density_shrub > 0.6, 255, 0)
    density_cactus = np.where(density_cactus > 0.9, 255, 0)

    cv2.imwrite(f'{output_prefix}_scatter_broadleaf.png', density_broadleaf.astype(np.uint8))
    cv2.imwrite(f'{output_prefix}_scatter_conifer.png', density_conifer.astype(np.uint8))
    cv2.imwrite(f'{output_prefix}_scatter_shrub.png', density_shrub.astype(np.uint8))
    cv2.imwrite(f'{output_prefix}_scatter_cactus.png', density_cactus.astype(np.uint8))
    print(f"Vegetation maps saved with prefix {output_prefix}.")

if __name__ == "__main__":
    generate_vegetation_scatter('wp3_biome_map.png', 'wp3_temperature_map.png', 'wp3_moisture_map.png')
