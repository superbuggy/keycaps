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
CAP_HEIGHT = 8.0     # Total height
WALL_THICKNESS = 1.2
TOP_THICKNESS = 1.5

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
    
    # FIX: The extrude method can return a list if the path has multiple
    # disconnected components. We concatenate them into a single mesh to ensure
    # a Trimesh object is always returned.
    if isinstance(tessellation_mesh, list):
        if not tessellation_mesh:
            return None
        tessellation_mesh = trimesh.util.concatenate(tessellation_mesh)
    
    return tessellation_mesh


def create_keycap_body(width, depth, height):
    """
    Creates the base of an MBK-style keycap, including the stem mount.
    This version accurately models the reinforcing ribs and separate grip posts.
    """
    # 1. Create the main outer body
    outer_box = trimesh.creation.box(bounds=[(-width/2, -depth/2, 0), (width/2, depth/2, height)])

    # 2. Create the hollow inner part to be subtracted
    inner_width = width - 2 * WALL_THICKNESS
    inner_depth = depth - 2 * WALL_THICKNESS
    inner_height = height - TOP_THICKNESS
    inner_box = trimesh.creation.box(bounds=[(-inner_width/2, -inner_depth/2, 0), 
                                             (inner_width/2, inner_depth/2, inner_height)])
    
    # Create the hollow keycap shell first
    keycap_shell = trimesh.boolean.difference([outer_box, inner_box])

    # 3. Create the accurate stem mount structure based on the provided image
    
    # a) Create the low-profile reinforcing cross ribs
    rib_height = 1.0
    rib_arm_length = 5.8 / 2
    rib_arm_width = 1.2 / 2
    
    rib_arm_x = trimesh.creation.box(bounds=[(-rib_arm_length, -rib_arm_width, 0),
                                             (rib_arm_length, rib_arm_width, rib_height)])
    rib_arm_y = trimesh.creation.box(bounds=[(-rib_arm_width, -rib_arm_length, 0),
                                             (rib_arm_width, rib_arm_length, rib_height)])
    rib_structure = trimesh.boolean.union([rib_arm_x, rib_arm_y])

    # b) Create the two vertical posts that grip the switch
    post_y_offset = 3.5 / 2  # Distance from center to each post
    post_diameter = 1.1
    
    post1 = trimesh.creation.cylinder(radius=post_diameter/2, height=STEM_HEIGHT)
    post1.apply_translation([0, post_y_offset, 0])
    
    post2 = trimesh.creation.cylinder(radius=post_diameter/2, height=STEM_HEIGHT)
    post2.apply_translation([0, -post_y_offset, 0])
    
    # c) Combine the ribs and the posts into one stem structure
    stem_and_ribs = trimesh.boolean.union([rib_structure, post1, post2])
    
    # 4. Combine the shell with the new, accurate stem structure
    final_body = trimesh.boolean.union([keycap_shell, stem_and_ribs])
    
    return final_body


def generate_keycap_grid():
    """
    Main function to generate the full grid of keycaps and export the STL.
    """
    print("--- Starting Keycap Generation ---")
    all_keycaps = []
    
    for row in range(ROWS):
        for col in range(COLS):
            print(f"Generating keycap ({row+1}, {col+1})...")
            
            # 1. Create the keycap base
            keycap_body = create_keycap_body(CAP_WIDTH, CAP_DEPTH, CAP_HEIGHT)
            
            # 2. Create the unique tessellated top
            tessellation_top = create_tessellated_top(CAP_WIDTH, CAP_DEPTH)
            
            if tessellation_top:
                # Create a solid base for the pattern to sit on
                top_base = trimesh.creation.box(bounds=[(-CAP_WIDTH/2, -CAP_DEPTH/2, 0),
                                                        (CAP_WIDTH/2, CAP_DEPTH/2, TOP_THICKNESS)])
                
                # FIX: Center the tessellation pattern at the origin before placing it.
                # The original was created in a positive quadrant, which caused alignment issues.
                # The old, complex centering logic was flawed and has been removed.
                center_offset = tessellation_top.bounds.mean(axis=0)
                tessellation_top.apply_translation(-center_offset)

                # Now that the pattern is centered, move its bottom to sit on top of the base slab.
                pattern_z_height = tessellation_top.bounds[1][2] - tessellation_top.bounds[0][2]
                tessellation_top.apply_translation([0, 0, TOP_THICKNESS + pattern_z_height / 2])

                # Combine the pattern and its base
                full_top = trimesh.boolean.union([top_base, tessellation_top])
                
                # Move the combined top to the keycap's height
                full_top.apply_translation([0, 0, CAP_HEIGHT - TOP_THICKNESS])
                
                # 3. Union the body and the top
                final_keycap = trimesh.boolean.union([keycap_body, full_top])
            else:
                # Fallback to a simple flat top if pattern generation fails
                final_keycap = keycap_body

            # 4. Position the keycap in the grid
            x_pos = col * SPACING
            y_pos = row * SPACING
            final_keycap.apply_translation([x_pos, y_pos, 0])
            
            all_keycaps.append(final_keycap)

    print("\nMerging all keycaps into a single file. This may take a moment...")
    # 5. Combine all keycaps into one mesh
    final_grid = trimesh.util.concatenate(all_keycaps)

    # Export the final result
    final_grid.export(OUTPUT_FILENAME)
    print(f"\n✅ Success! 3D model saved as '{OUTPUT_FILENAME}'")


if __name__ == '__main__':
    generate_keycap_grid()
