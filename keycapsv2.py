import trimesh
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, box

# --- PARAMETERS ---

# Grid layout
ROWS = 3
COLS = 5
SPACING = 19.05  # Standard 1U spacing in mm

# Keycap Dimensions (based on MBK standard)
CAP_WIDTH = 17.4   # X-dimension
CAP_DEPTH = 16.4   # Y-dimension
TOP_THICKNESS = 1.5  # Thickness of the top plate

# Ergonomic Parameters
CORNER_RADIUS = 2.0  # How rounded the keycap corners are
DISH_DEPTH = 0.6     # How deep the central depression is

# Stem Dimensions (for Kailh Choc switches)
STEM_HEIGHT = 3.5
# Tessellation Parameters
PATTERN_HEIGHT = 0.8  # How high the pattern rises from the keycap top
RIDGE_THICKNESS = 0.4 # Thickness of the concentric walls
NUM_CONCENTRIC_STEPS = 3 # Number of concentric lines per cell
VORONOI_POINTS = 25   # Number of points to generate the Voronoi cells

OUTPUT_FILENAME = 'tessellated_keycaps.stl'


def create_tessellated_top(width, depth):
    """
    Generates a 3D mesh of a Voronoi tessellation pattern.
    """
    # 1. Generate random points for the Voronoi diagram, centered around (0,0).
    points = (np.random.rand(VORONOI_POINTS, 2) - 0.5)
    points[:, 0] *= width
    points[:, 1] *= depth

    # 2. Compute the Voronoi diagram
    vor = Voronoi(points)
    
    # Bounding box for clipping polygons that go to infinity
    bounding_box = Polygon([(-width/2, -depth/2), (-width/2, depth/2), (width/2, depth/2), (width/2, -depth/2)])
    
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


def create_stems(width):
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
            stems = create_stems(CAP_WIDTH)
            
            # 2. Create the flat, rounded-corner top plate
            b = box(-CAP_WIDTH/2 + CORNER_RADIUS, -CAP_DEPTH/2 + CORNER_RADIUS, 
                    CAP_WIDTH/2 - CORNER_RADIUS, CAP_DEPTH/2 - CORNER_RADIUS)
            rounded_rect = b.buffer(CORNER_RADIUS)
            top_base = trimesh.load_path(rounded_rect).extrude(height=TOP_THICKNESS)

            # 3. Create the unique tessellated pattern
            tessellation_pattern = create_tessellated_top(CAP_WIDTH, CAP_DEPTH)
            
            full_top = top_base
            if tessellation_pattern:
                # Place the pattern on top of the base plate
                tessellation_pattern.apply_translation([0, 0, TOP_THICKNESS])
                
                # FIX: Union the base and pattern into a single unified object *before* deformation.
                full_top = trimesh.boolean.union([top_base, tessellation_pattern])

            # 4. Apply a deformation to the unified top surface to create the fingertip dish.
            vertices = full_top.vertices
            
            # Identify only the vertices on the top surface to be deformed.
            # We consider any vertex near the max height of the object to be on the top.
            max_z = full_top.bounds[1][2]
            z_tolerance = 0.1
            top_vertex_indices = np.where(vertices[:, 2] > max_z - PATTERN_HEIGHT - z_tolerance)[0]
            
            max_effect_radius = min(CAP_WIDTH, CAP_DEPTH) / 2.0
            
            for i in top_vertex_indices:
                v = vertices[i]
                dist = np.linalg.norm(v[:2]) # Distance from center in XY plane
                if dist < max_effect_radius:
                    # Use a cosine curve for a smooth falloff
                    scale = (np.cos(dist / max_effect_radius * np.pi) + 1) / 2
                    z_displacement = scale * DISH_DEPTH
                    vertices[i][2] -= z_displacement
            
            full_top.vertices = vertices # Apply the transformed vertices back to the mesh

            # 5. Assemble the final keycap without perimeter walls.
            # Move the top surface up so the stems can sit underneath.
            full_top.apply_translation([0, 0, STEM_HEIGHT])
            
            # Union the top surface and the stems.
            final_keycap = trimesh.boolean.union([full_top, stems])

            # 6. Position the completed keycap in the grid
            x_pos = col * SPACING
            y_pos = row * SPACING
            final_keycap.apply_translation([x_pos, y_pos, 0])
            
            all_keycaps.append(final_keycap)

    print("\nMerging all keycaps into a single file. This may take a moment...")
    # 7. Combine all keycaps into one mesh
    final_grid = trimesh.util.concatenate(all_keycaps)

    # Export the final result
    final_grid.export(OUTPUT_FILENAME)
    print(f"\n✅ Success! 3D model saved as '{OUTPUT_FILENAME}'")


if __name__ == '__main__':
    generate_keycap_grid()
