import os
import json
import zipfile
import cv2

def export_for_unity(planet_name, heightmap_path, biome_path, moisture_path, temp_path):
    print(f"Exporting {planet_name} maps for Unity...")
    export_dir = f"{planet_name}_unity_export"
    os.makedirs(export_dir, exist_ok=True)
    
    files_to_zip = []
    
    # Unity prefers raw 16-bit for heightmaps or standard PNGs
    h16 = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    if h16 is not None:
        h_path = os.path.join(export_dir, f"{planet_name}_heightmap.raw")
        with open(h_path, 'wb') as f:
            f.write(h16.tobytes())
        files_to_zip.append(h_path)
    
    # Copy other maps
    maps = {
        "biome": biome_path,
        "moisture": moisture_path,
        "temperature": temp_path
    }
    
    for name, path in maps.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            out_path = os.path.join(export_dir, f"{planet_name}_{name}.png")
            cv2.imwrite(out_path, img)
            files_to_zip.append(out_path)
            
    # Metadata
    metadata = {
        "planetName": planet_name,
        "width": h16.shape[1] if h16 is not None else 1024,
        "height": h16.shape[0] if h16 is not None else 512,
        "heightFormat": "R16",
        "hasBiomes": os.path.exists(biome_path)
    }
    
    meta_path = os.path.join(export_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    files_to_zip.append(meta_path)
    
    # Create Zip
    zip_path = f"{planet_name}_unity_export.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))
            
    print(f"Export complete: {zip_path}")
    return zip_path

if __name__ == "__main__":
    export_for_unity("test_planet", "wp2_height_map.png", "wp3_biome_map.png", "wp3_moisture_map.png", "wp3_temperature_map.png")
