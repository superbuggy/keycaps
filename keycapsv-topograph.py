import trimesh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box, LineString
from noise import pnoise2
from skimage.measure import find_contours

# --- PARAMETERS ---

# Grid layout
ROWS = 3
COLS = 5
SPACING = 19.05  # Standard 1U spacing in mm

# Keycap Dimensions (based on MBK standard)
CAP_WIDTH = 17.4   # X-dimension
CAP_DEPTH = 16.4   # Y-dimension
TOP_THICKNESS = 2.5  # Thickness of the top plate

# Ergonomic Parameters
CORNER_RADIUS = 2.0  # How rounded the keycap corners are
DISH_DEPTH = 2.2     # How deep the central depression is

# Stem Dimensions (for Kailh Choc switches)
STEM_HEIGHT = 3.5
# Topographic Pattern Parameters
PATTERN_HEIGHT = 0.5      # How high the pattern rises from the keycap top
RIDGE_THICKNESS = 0.3     # Thickness of the contour lines
NOISE_SCALE = 0.08        # Controls the "zoom" level of the noise. Smaller values = larger features.
NOISE_OCTAVES = 4         # Adds detail to the noise. Higher values = more complexity.
CONTOUR_LEVELS = 15       # The number of contour lines to draw.

OUTPUT_FILENAME = 'topographic_keycaps.stl'


def create_topographic_pattern(width, depth, keycap_shape):
    """
    Generates a 3D mesh of a topographic map pattern using Perlin noise.
    """
    print("  Generating topographic pattern...")
    # 1. Create a grid of points to evaluate the noise
    resolution = 100
    x = np.linspace(-width/2, width/2, resolution)
    y = np.linspace(-depth/2, depth/2, resolution)
    
    # The 'base' for pnoise must be an integer.
    base = np.random.randint(0, 1000)
    
    # 2. Generate a 2D Perlin noise field (our "height map")
    height_map = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            height_map[i, j] = pnoise2(x[i] * NOISE_SCALE, 
                                       y[j] * NOISE_SCALE, 
                                       octaves=NOISE_OCTAVES, 
                                       base=base)

    # 3. Find contour lines at different levels of the height map
    all_ridges = []
    min_h, max_h = height_map.min(), height_map.max()
    
    for level in np.linspace(min_h, max_h, CONTOUR_LEVELS):
        contours = find_contours(height_map, level)
        
        for contour in contours:
            # The contour points are in pixel space, so we scale them back to model space
            scaled_contour = contour * (np.array([width, depth]) / resolution) - np.array([width/2, depth/2])
            
            if len(scaled_contour) > 2:
                line = LineString(scaled_contour)
                
                # We only keep contours that are fully contained within the keycap's rounded shape.
                if keycap_shape.contains(line):
                    ridge = line.buffer(RIDGE_THICKNESS / 2, cap_style=2) # cap_style=2 is flat
                    if isinstance(ridge, Polygon):
                        all_ridges.append(ridge)
                    elif isinstance(ridge, MultiPolygon):
                        all_ridges.extend(list(ridge.geoms))

    if not all_ridges: return None

    # 4. Convert the 2D shapes to a 3D mesh
    combined_ridges = MultiPolygon(all_ridges)
    path_2d = trimesh.load_path(combined_ridges)
    pattern_mesh = path_2d.extrude(height=PATTERN_HEIGHT)
    
    if isinstance(pattern_mesh, list):
        if not pattern_mesh: return None
        pattern_mesh = trimesh.util.concatenate(pattern_mesh)
    
    return pattern_mesh


def create_stems(width):
    """
    Creates the stem mount for the keycap.
    """
    # Define geometry for a single "I"-shaped post
    spine_width, spine_depth = 0.6, 1.2
    wing_width, wing_depth = 1.8, 0.5
    spine = trimesh.creation.box(bounds=[(-spine_width / 2, -spine_depth / 2, 0), (spine_width / 2, spine_depth / 2, STEM_HEIGHT)])
    wing = trimesh.creation.box(bounds=[(-wing_width / 2, -wing_depth / 2, 0), (wing_width / 2, wing_depth / 2, STEM_HEIGHT)])
    wing1, wing2 = wing.copy(), wing.copy()
    wing1.apply_translation([0, (spine_depth - wing_depth) / 2, 0])
    wing2.apply_translation([0, -(spine_depth - wing_depth) / 2, 0])
    single_post = trimesh.boolean.union([spine, wing1, wing2])

    # Rotate the post 90 degrees
    angle = np.deg2rad(90)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    single_post.apply_transform(rotation_matrix)

    # Create and position two posts
    post_x_offset = width / 6.0
    final_post1, final_post2 = single_post.copy(), single_post.copy()
    final_post1.apply_translation([post_x_offset, 0, 0])
    final_post2.apply_translation([-post_x_offset, 0, 0])
    
    return trimesh.boolean.union([final_post1, final_post2])


def generate_keycap_grid():
    """
    Main function to generate the full grid of keycaps and export the STL.
    """
    print("--- Starting Keycap Generation ---")
    all_keycaps = []
    
    for row in range(ROWS):
        for col in range(COLS):
            print(f"Generating keycap ({row+1}, {col+1})...")
            
            # 1. Create the stems
            stems = create_stems(CAP_WIDTH)
            
            # 2. Create the flat, rounded-corner top plate and get its 2D shape
            b = box(-CAP_WIDTH/2 + CORNER_RADIUS, -CAP_DEPTH/2 + CORNER_RADIUS, 
                    CAP_WIDTH/2 - CORNER_RADIUS, CAP_DEPTH/2 - CORNER_RADIUS)
            rounded_rect_shape = b.buffer(CORNER_RADIUS)
            top_base = trimesh.load_path(rounded_rect_shape).extrude(height=TOP_THICKNESS)

            # 3. Create the unique topographic pattern
            pattern = create_topographic_pattern(CAP_WIDTH, CAP_DEPTH, rounded_rect_shape)
            
            full_top = top_base
            if pattern:
                # Place pattern on top of the base plate
                pattern.apply_translation([0, 0, TOP_THICKNESS])
                
                # Use concatenate instead of boolean.union to avoid ValueError.
                full_top = trimesh.util.concatenate([top_base, pattern])

            # 4. Apply deformation to the unified top surface for the fingertip dish
            vertices = full_top.vertices
            
            # FIX: Select vertices based on face normals to ensure only the top "skin" is deformed.
            # This is more robust than using Z-height alone.
            face_normals = full_top.face_normals
            # Find faces that point mostly up (Z-component of normal > 0.5)
            top_face_indices = np.where(face_normals[:, 2] > 0.5)[0]
            # Get the unique vertices from these top-facing faces
            top_vertex_indices = np.unique(full_top.faces[top_face_indices])

            max_effect_radius = min(CAP_WIDTH, CAP_DEPTH) * 0.75
            
            for i in top_vertex_indices:
                v = vertices[i]
                dist = np.linalg.norm(v[:2])
                if dist < max_effect_radius:
                    scale = (np.cos(dist / max_effect_radius * np.pi) + 1) / 2
                    z_displacement = scale * DISH_DEPTH
                    vertices[i][2] -= z_displacement
            full_top.vertices = vertices

            # 5. Assemble the final keycap
            full_top.apply_translation([0, 0, STEM_HEIGHT])

            # Use concatenate for the final assembly as well.
            final_keycap = trimesh.util.concatenate([full_top, stems])

            # 6. Position the keycap in the grid
            x_pos = col * SPACING
            y_pos = row * SPACING
            final_keycap.apply_translation([x_pos, y_pos, 0])
            all_keycaps.append(final_keycap)

    print("\nMerging all keycaps into a single file...")
    final_grid = trimesh.util.concatenate(all_keycaps)
    final_grid.export(OUTPUT_FILENAME)
    print(f"\n✅ Success! 3D model saved as '{OUTPUT_FILENAME}'")


if __name__ == '__main__':
    generate_keycap_grid()
