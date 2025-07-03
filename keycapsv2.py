import trimesh
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon

# --- PARAMETERS ---

# Grid layout
ROWS = 3
COLS = 5
SPACING = 19.05  # Standard 1U spacing in mm

# Keycap Dimensions (based on MBK standard)
CAP_WIDTH = 17.4   # X-dimension
CAP_DEPTH = 16.4   # Y-dimension
TOP_THICKNESS = 1.5  # Thickness of the top plate

# Stem Dimensions (for Kailh Choc switches)
STEM_HEIGHT = 3.5
# Tessellation Parameters
PATTERN_HEIGHT = 1.0  # How high the pattern rises from the keycap top
RIDGE_THICKNESS = 0.4 # Thickness of the concentric walls
NUM_CONCENTRIC_STEPS = 3 # Number of concentric lines per cell
VORONOI_POINTS = 25   # Number of points to generate the Voronoi cells

OUTPUT_FILENAME = 'tessellated_keycaps.stl'


def create_tessellated_top(width, depth):
    """
    Generates a 3D mesh of a Voronoi tessellation pattern.
    """
    # 1. Generate random points for the Voronoi diagram
    points = np.random.rand(VORONOI_POINTS, 2)
    points[:, 0] *= width
    points[:, 1] *= depth

    # 2. Compute the Voronoi diagram
    vor = Voronoi(points)
    
    # Bounding box for clipping polygons that go to infinity
    bounding_box = Polygon([(0, 0), (0, depth), (width, depth), (width, 0)])
    
    all_ridges = []

    # 3. Process each Voronoi region
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not -1 in region: # Ensure region is a finite polygon
            # Get polygon vertices and clip to the keycap boundary
            polygon_vertices = vor.vertices[region]
            shape = Polygon(polygon_vertices).intersection(bounding_box)

            if isinstance(shape, Polygon) and shape.area > 0.1:
                # 4. Create concentric ridges
                for i in range(1, NUM_CONCENTRIC_STEPS + 1):
                    offset = i * (RIDGE_THICKNESS * 1.5)
                    # Inset the polygon
                    inner_poly = shape.buffer(-offset)
                    if not inner_poly.is_empty:
                        # Create the ridge by subtracting a further inset polygon
                        ridge = inner_poly.difference(inner_poly.buffer(-RIDGE_THICKNESS))
                        if isinstance(ridge, Polygon):
                            all_ridges.append(ridge)
                        elif isinstance(ridge, MultiPolygon):
                            all_ridges.extend(list(ridge.geoms))

    if not all_ridges:
        return None

    # 5. Combine all ridge polygons into a single MultiPolygon object.
    combined_ridges = MultiPolygon(all_ridges)

    # 6. Convert the 2D Shapely object to a 3D Trimesh object
    path_2d = trimesh.load_path(combined_ridges)
    tessellation_mesh = path_2d.extrude(height=PATTERN_HEIGHT)
    
    if isinstance(tessellation_mesh, list):
        if not tessellation_mesh:
            return None
        tessellation_mesh = trimesh.util.concatenate(tessellation_mesh)
    
    return tessellation_mesh


def create_keycap_body(width):
    """
    Creates the stem mount for the keycap.
    This version models two "I"-shaped grip posts, rotates each one 90 degrees
    on its own Z-axis, and places them correctly relative to the keycap width.
    """
    # 1. Define geometry for a single "I"-shaped post, oriented with its height along the Y-axis.
    # The "I" is built from a central spine and two wings.
    spine_width = 0.6   # Spine dimension along X
    spine_depth = 1.2   # Spine dimension along Y
    wing_width = 1.8    # Wing dimension along X
    wing_depth = 0.5    # Wing dimension along Y

    spine = trimesh.creation.box(bounds=[(-spine_width / 2, -spine_depth / 2, 0),
                                         (spine_width / 2, spine_depth / 2, STEM_HEIGHT)])
    
    wing = trimesh.creation.box(bounds=[(-wing_width / 2, -wing_depth / 2, 0),
                                        (wing_width / 2, wing_depth / 2, STEM_HEIGHT)])

    # Place the wings at the ends of the spine
    wing1 = wing.copy()
    wing1.apply_translation([0, (spine_depth - wing_depth) / 2, 0])

    wing2 = wing.copy()
    wing2.apply_translation([0, -(spine_depth - wing_depth) / 2, 0])
    
    single_post = trimesh.boolean.union([spine, wing1, wing2])

    # 2. Rotate the single post 90 degrees around its own Z-axis before copying it.
    angle = np.deg2rad(90)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    single_post.apply_transform(rotation_matrix)

    # 3. Create two posts and position them 1/3 of the way in from the left and right.
    # The offset from the center is 1/6 of the total keycap width.
    post_x_offset = width / 6.0
    
    final_post1 = single_post.copy()
    final_post1.apply_translation([post_x_offset, 0, 0])
    
    final_post2 = single_post.copy()
    final_post2.apply_translation([-post_x_offset, 0, 0])

    # 4. Combine the two posts into the final stem structure
    stem_structure = trimesh.boolean.union([final_post1, final_post2])
    
    return stem_structure


def generate_keycap_grid():
    """
    Main function to generate the full grid of keycaps and export the STL.
    """
    print("--- Starting Keycap Generation ---")
    all_keycaps = []
    
    for row in range(ROWS):
        for col in range(COLS):
            print(f"Generating keycap ({row+1}, {col+1})...")
            
            # 1. Create the stems, with their base at z=0
            stems = create_keycap_body(CAP_WIDTH)
            
            # 2. Create the top surface, starting with a solid base plate
            top_base = trimesh.creation.box(bounds=[(-CAP_WIDTH/2, -CAP_DEPTH/2, 0),
                                                    (CAP_WIDTH/2, CAP_DEPTH/2, TOP_THICKNESS)])

            # 3. Create the unique tessellated pattern
            tessellation_pattern = create_tessellated_top(CAP_WIDTH, CAP_DEPTH)
            
            full_top = top_base
            if tessellation_pattern:
                # Center the pattern and place it on top of the base plate
                center_offset = tessellation_pattern.bounds.mean(axis=0)
                tessellation_pattern.apply_translation(-center_offset)
                tessellation_pattern.apply_translation([0, 0, TOP_THICKNESS])
                
                # Union the base plate and the pattern
                full_top = trimesh.boolean.union([top_base, tessellation_pattern])

            # 4. Assemble the final keycap without perimeter walls.
            # Move the top surface up so the stems can sit underneath.
            full_top.apply_translation([0, 0, STEM_HEIGHT])
            
            # Union the top surface and the stems.
            final_keycap = trimesh.boolean.union([full_top, stems])

            # 5. Position the completed keycap in the grid
            x_pos = col * SPACING
            y_pos = row * SPACING
            final_keycap.apply_translation([x_pos, y_pos, 0])
            
            all_keycaps.append(final_keycap)

    print("\nMerging all keycaps into a single file. This may take a moment...")
    # 6. Combine all keycaps into one mesh
    final_grid = trimesh.util.concatenate(all_keycaps)

    # Export the final result
    final_grid.export(OUTPUT_FILENAME)
    print(f"\n✅ Success! 3D model saved as '{OUTPUT_FILENAME}'")


if __name__ == '__main__':
    generate_keycap_grid()
