import trimesh
import numpy as np

def create_connector(start_point, end_point, radius, sections=12):
    """
    Creates a cylinder mesh between two 3D points.
    """
    # Vector representing the cylinder's axis and its length
    vec = np.array(end_point) - np.array(start_point)
    length = np.linalg.norm(vec)

    # Create a cylinder primitive along the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)

    # Find the transformation to align the cylinder with the vector between the points
    transform = trimesh.geometry.align_vectors([0, 0, 1], vec)
    
    # Position the cylinder's midpoint correctly
    midpoint = (np.array(start_point) + np.array(end_point)) / 2
    transform[:3, 3] = midpoint
    
    # Apply the transformation
    cylinder.apply_transform(transform)
    return cylinder

def generate_form():
    """
    Main function to generate the complete 3D form and export it as an STL file.
    """
    print("Initializing generation of 3D form...")

    # --- 1. Define Sphere Positions and Radii ---
    # These coordinates are chosen to create a 3D layout inspired by the 2D diagram.
    # Feel free to tweak these values to alter the final shape.
    positions = {
        '1': [0, -5, 0],    # Central, lowest sphere
        '2': [18, 20, 5],   # Top-right
        '4': [-15, 18, -8],  # Top-left, further back
        '5': [-20, 15, 6],  # Top-left, further forward
        '7': [20, 5, 0],    # Mid-right
        '8': [0, -5, -20]   # Directly below sphere 1
    }

    radii = {
        '1': 6.0,   # Largest sphere
        '2': 4.0,
        '4': 4.0,
        '5': 4.0,
        '7': 4.5,   # Slightly larger to accommodate connections
        '8': 4.0
    }
    
    # List to hold all the individual mesh components
    components = []

    # --- 2. Create the Sphere Meshes ---
    print("Creating primary sphere meshes...")
    for key, pos in positions.items():
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=radii[key])
        sphere.apply_translation(pos)
        components.append(sphere)

    # --- 3. Create the Connectors (Tubes) ---
    # The "stippled" areas are represented by thicker tubes for simplicity.
    print("Creating connectors...")
    
    # Thin connectors
    components.append(create_connector(positions['5'], positions['4'], radius=0.8))
    components.append(create_connector(positions['1'], positions['8'], radius=1.2))

    # Stippled / Thicker connectors
    # Note: A true "web" is complex. We use thick tubes as a proxy.
    components.append(create_connector(positions['4'], positions['1'], radius=2.5))
    components.append(create_connector(positions['5'], positions['7'], radius=2.8))
    components.append(create_connector(positions['7'], positions['1'], radius=3.0))
    components.append(create_connector(positions['2'], positions['7'], radius=2.0))
    
    # Special loop on sphere 1 (modeled as a torus)
    torus = trimesh.creation.torus(major_radius=radii['1'] + 2.0, minor_radius=0.7, major_sections=24, minor_sections=12)
    # Position and orient the torus next to sphere 1
    torus.apply_translation([positions['1'][0] + radii['1'] + 2.0, positions['1'][1], positions['1'][2]])
    components.append(torus)

    # --- 4. Combine all components into a single mesh ---
    print("Merging all components into a single mesh. This may take a moment...")
    # The union operation combines all meshes into one continuous object
    final_mesh = trimesh.boolean.union(components)

    # --- 5. Export the Final Mesh ---
    try:
        file_name = 'form_imitation.stl'
        final_mesh.export(file_name)
        print(f"\n✅ Success! 3D form saved as '{file_name}'")
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        print("Please ensure you have write permissions in the current directory.")


if __name__ == '__main__':
    generate_form()