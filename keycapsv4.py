import trimesh
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, box, LineString

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
# Hybrid Pattern Parameters
PATTERN_HEIGHT = 0.6  # How high the pattern rises from the keycap top
RIDGE_THICKNESS = 0.4 # Thickness of the concentric walls
NUM_CONCENTRIC_STEPS = 3 # Number of concentric lines per cell
VORONOI_POINTS = 45   # Number of points to generate the Voronoi cells
DISTORTION_AMOUNT = 0.3 # How much to distort the Voronoi cells. 0 is no distortion.
EDGE_SUBDIVISIONS = 3 # How many new points to add to each edge for smoother distortion.

OUTPUT_FILENAME = 'hybrid_keycaps.stl'


def create_hybrid_pattern(width, depth):
    """
    Generates a 3D mesh of a hybrid pattern by distorting Voronoi cells to imitate organic growth.
    """
    print("  Generating hybrid pattern...")
    # 1. Generate random points for the Voronoi diagram, centered around (0,0).
    points = (np.random.rand(VORONOI_POINTS, 2) - 0.5)
    points[:, 0] *= width
    points[:, 1] *= depth

    # 2. Compute the Voronoi diagram
    vor = Voronoi(points)
    
    # Bounding box for clipping polygons
    bounding_box = Polygon([(-width/2, -depth/2), (-width/2, depth/2), (width/2, depth/2), (width/2, -depth/2)])
    
    all_ridges = []

    # 3. Process each Voronoi region
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not -1 in region:
            polygon_vertices = vor.vertices[region]
            shape = Polygon(polygon_vertices).intersection(bounding_box)

            if isinstance(shape, Polygon) and shape.area > 0.1:
                # 4. Distort the polygon to make it look organic
                distorted_coords = []
                coords = list(shape.exterior.coords)
                for i in range(len(coords) - 1):
                    p1 = np.array(coords[i])
                    p2 = np.array(coords[i+1])
                    distorted_coords.append(p1)
                    
                    # Subdivide the edge and apply perpendicular noise
                    edge_vec = p2 - p1
                    edge_normal = np.array([-edge_vec[1], edge_vec[0]])
                    edge_normal /= (np.linalg.norm(edge_normal) + 1e-6)

                    for j in range(1, EDGE_SUBDIVISIONS + 1):
                        t = j / (EDGE_SUBDIVISIONS + 1)
                        mid_point = p1 + t * edge_vec
                        offset = (np.random.rand() - 0.5) * 2 * DISTORTION_AMOUNT
                        distorted_coords.append(mid_point + edge_normal * offset)
                
                distorted_shape = Polygon(distorted_coords)

                # 5. Create concentric ridges from the distorted shape
                for i in range(1, NUM_CONCENTRIC_STEPS + 1):
                    offset = i * (RIDGE_THICKNESS * 1.5)
                    inner_poly = distorted_shape.buffer(-offset)
                    if not inner_poly.is_empty:
                        ridge = inner_poly.difference(inner_poly.buffer(-RIDGE_THICKNESS))
                        if isinstance(ridge, Polygon): all_ridges.append(ridge)
                        elif isinstance(ridge, MultiPolygon): all_ridges.extend(list(ridge.geoms))

    if not all_ridges: return None

    # 6. Convert the 2D shapes to a 3D mesh
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
            
            # 2. Create the flat, rounded-corner top plate
            b = box(-CAP_WIDTH/2 + CORNER_RADIUS, -CAP_DEPTH/2 + CORNER_RADIUS, 
                    CAP_WIDTH/2 - CORNER_RADIUS, CAP_DEPTH/2 - CORNER_RADIUS)
            rounded_rect = b.buffer(CORNER_RADIUS)
            top_base = trimesh.load_path(rounded_rect).extrude(height=TOP_THICKNESS)

            # 3. Create the unique hybrid pattern
            pattern = create_hybrid_pattern(CAP_WIDTH, CAP_DEPTH)
            
            full_top = top_base
            if pattern:
                # Place pattern on top of the base plate and union them
                pattern.apply_translation([0, 0, TOP_THICKNESS])
                full_top = trimesh.boolean.union([top_base, pattern])

            # 4. Apply deformation to the unified top surface for the fingertip dish
            vertices = full_top.vertices
            max_z = full_top.bounds[1][2]
            z_tolerance = 0.1
            top_vertex_indices = np.where(vertices[:, 2] > max_z - PATTERN_HEIGHT - z_tolerance)[0]
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
            final_keycap = trimesh.boolean.union([full_top, stems])

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
