import cv2 # OpenCV for image loading and basic manipulation (better than Pillow for this)
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent browser crashes/hangs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

# --- Configuration ---
MAP_IMAGE_PATH = 'Motoki_Aspsp_uk_Fig02_c_Islands.jpg' # Replace with the path to your image
OUTPUT_IMAGE_PATH = 'globe_projection.png'
OUTPUT_ANIMATION_PATH = 'globe_rotation.mp4'
GLOBE_RADIUS = 1.0 # Arbitrary radius for the 3D projection
OCEAN_COLOR = (50, 150, 255) # RGB for ocean (blue shades)
CONTINENT_COLOR = (150, 200, 50) # RGB for continents (yellow-green shades)
BACKGROUND_COLOR = (0, 0, 0) # Black background for space

# --- Step 1: Load and Process the Mercator Map ---
def load_and_process_map(image_path):
    """
    Loads the map image, converts it to grayscale, and identifies land/ocean.
    Assumes black outlines are land, white background is ocean initially.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Map image not found at {image_path}")

    # Invert colors if necessary: make land white, ocean black for easier processing
    _, binary_map = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV) # outlines become white

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape
    colored_map = np.zeros((height, width, 3), dtype=np.uint8) # RGB image for colored map

    # Start with ocean
    colored_map[:] = OCEAN_COLOR

    # Draw filled contours for continents
    cv2.drawContours(colored_map, contours, -1, CONTINENT_COLOR, cv2.FILLED)

    # Convert to a boolean mask for easier projection
    # True for land, False for ocean
    # Using a simpler check: if the pixel color matches CONTINENT_COLOR
    land_mask = np.all(colored_map == np.array(CONTINENT_COLOR, dtype=np.uint8), axis=-1)

    cv2.imwrite('debug_2d_map.png', cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR))
    return land_mask, colored_map # Return mask and the 2D colored map for reference

# --- Step 2: Mercator Projection to Spherical Coordinates ---
def mercator_to_spherical(x_img, y_img, img_width, img_height):
    """
    Converts 2D Mercator pixel coordinates (x_img, y_img) to
    spherical coordinates (longitude, latitude).
    """
    longitude = (x_img / img_width) * 360 - 180 
    longitude_rad = np.deg2rad(longitude)

    y_normalized = 2 * (y_img / img_height) - 1 
    latitude_rad = 2 * (np.arctan(np.exp(np.pi * y_normalized)) - np.pi / 4)
    latitude = np.rad2deg(latitude_rad) 

    return longitude_rad, latitude_rad

# --- Step 3: Spherical Coordinates to 3D Cartesian Coordinates ---
def spherical_to_cartesian(lon_rad, lat_rad, radius):
    """
    Converts spherical coordinates (longitude, latitude) to 3D Cartesian (x, y, z).
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def prepare_globe_data(map_path):
    land_mask, colored_map_2d = load_and_process_map(map_path)
    height, width = land_mask.shape
    
    # Mesh generation
    u = np.linspace(0, 2 * np.pi, width)  # Longitude
    v = np.linspace(-np.pi / 2, np.pi / 2, height) # Latitude

    x_globe = GLOBE_RADIUS * np.outer(np.cos(u), np.cos(v)).T
    y_globe = GLOBE_RADIUS * np.outer(np.sin(u), np.cos(v)).T
    z_globe = GLOBE_RADIUS * np.outer(np.ones(np.size(u)), np.sin(v)).T

    # Vectorized color assignment
    globe_colors = np.zeros((height, width, 3))
    # Normalize colors to 0-1 range for matplotlib
    globe_colors[land_mask] = np.array(CONTINENT_COLOR) / 255.0
    globe_colors[~land_mask] = np.array(OCEAN_COLOR) / 255.0
    
    return x_globe, y_globe, z_globe, globe_colors, height, width

# --- Main Program ---
def create_globe_from_map(map_path, output_path):
    x_globe, y_globe, z_globe, globe_colors, height, width = prepare_globe_data(map_path)

    # Prepare for 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BACKGROUND_COLOR) # Set background to black (space)

    # Plot the surface
    ax.plot_surface(x_globe, y_globe, z_globe,
                    facecolors=globe_colors,
                    rcount=min(height, 200), ccount=min(width, 200), # Cap resolution for performance
                    linewidth=0, antialiased=True, shade=False)

    # Set view angle to see more of the globe
    ax.view_init(elev=20, azim=30)
    ax.set_box_aspect([1,1,1]) # Equal aspect ratio for true sphere

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Alien Planet Globe - {timestamp}")
    ax.axis('off') # Hide axes

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"Globe image saved to {output_path}")

def create_rotating_globe(map_path, output_path, frames=60, fps=15):
    print("Generating rotation animation... this may take a moment.")
    x_globe, y_globe, z_globe, globe_colors, height, width = prepare_globe_data(map_path)

    fig = plt.figure(figsize=(8, 8)) # Slightly smaller for animation speed
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axis('off')

    # Plot once
    ax.plot_surface(x_globe, y_globe, z_globe,
                    facecolors=globe_colors,
                    rcount=min(height, 150), ccount=min(width, 150), # Lower res for animation
                    linewidth=0, antialiased=True, shade=False)
    
    ax.set_box_aspect([1,1,1])

    def update(frame):
        # Rotate azimuth
        angle = (frame / frames) * 360
        ax.view_init(elev=20, azim=angle)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
    
    writer_type = 'ffmpeg' if output_path.endswith('.mp4') else 'pillow'
    ani.save(output_path, writer=writer_type, fps=fps)
    
    plt.close(fig)
    print(f"Animation saved to {output_path}")

# --- Execute ---
if __name__ == "__main__":
    create_globe_from_map(MAP_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    # Uncomment to generate animation
    create_rotating_globe(MAP_IMAGE_PATH, OUTPUT_ANIMATION_PATH, frames=60, fps=20)
    
    print("\nNote: For a truly smooth and photorealistic result, you'd typically use dedicated 3D rendering software (like Blender, Three.js, etc.) and apply the processed 2D map as a texture onto a sphere model.")
    print("This script uses Matplotlib's 3D plotting, which is good for visualization but has limitations for high-fidelity rendering.")