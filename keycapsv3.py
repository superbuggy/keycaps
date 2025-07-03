import trimesh
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, MultiPolygon, box

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
# Mycorrhizal Pattern Parameters
PATTERN_HEIGHT = 0.6  # How high the pattern rises from the keycap top
BRANCH_RADIUS = 0.25  # Thickness of the mycorrhizal branches
N_ATTRACTORS = 300    # Number of points to guide the growth. More points = denser pattern.
BRANCH_LENGTH = 1.0   # The length of each new branch segment
INFLUENCE_RADIUS = 4.0 # How far a branch can "see" an attractor point
KILL_RADIUS = 1.5      # How close a branch must be to "consume" an attractor point

OUTPUT_FILENAME = 'mycorrhizal_keycaps.stl'


def create_mycorrhizal_pattern(width, depth):
    """
    Generates a 3D mesh of a mycorrhizal (root-like) pattern using a space colonization algorithm.
    """
    print("  Generating mycorrhizal pattern...")
    # 1. Scatter attraction points within the keycap bounds
    attractors = (np.random.rand(N_ATTRACTORS, 2) - 0.5)
    attractors[:, 0] *= width * 0.95
    attractors[:, 1] *= depth * 0.95

    # 2. Initialize the tree with a few roots around the edges
    root_points = np.array([
        [width * 0.4, -depth * 0.45],
        [-width * 0.4, -depth * 0.45],
        [0, depth * 0.45]
    ])
    tree_nodes = list(root_points)
    tree_parents = {0: -1, 1: -1, 2: -1} # Roots have no parents

    # 3. Run the growth simulation
    for i in range(50): # Max iterations to prevent infinite loops
        if len(attractors) == 0: break

        attractor_map = {}
        # Find the closest tree node for each attractor
        distances, closest_node_indices = cKDTree(tree_nodes).query(attractors)
        
        for attr_idx, node_idx in enumerate(closest_node_indices):
            if distances[attr_idx] < INFLUENCE_RADIUS:
                if node_idx not in attractor_map: attractor_map[node_idx] = []
                attractor_map[node_idx].append(attractors[attr_idx])

        new_nodes_this_iteration = []
        for node_idx, influencing_attractors in attractor_map.items():
            parent_node = tree_nodes[node_idx]
            avg_direction = np.mean(np.array(influencing_attractors) - parent_node, axis=0)
            avg_direction /= np.linalg.norm(avg_direction)
            
            new_node = parent_node + avg_direction * BRANCH_LENGTH
            new_nodes_this_iteration.append(new_node)
            tree_parents[len(tree_nodes) + len(new_nodes_this_iteration) - 1] = node_idx

        if not new_nodes_this_iteration: break
            
        tree_nodes.extend(new_nodes_this_iteration)
        
        # Remove attractors that have been reached
        distances, _ = cKDTree(tree_nodes).query(attractors)
        attractors = attractors[distances > KILL_RADIUS]

    # 4. Convert the 2D skeleton to a 3D mesh
    components = []
    for child_idx, parent_idx in tree_parents.items():
        if parent_idx == -1: continue
        
        start_node = tree_nodes[parent_idx]
        end_node = tree_nodes[child_idx]
        
        # Create a cylinder for the branch
        vec = end_node - start_node
        length = np.linalg.norm(vec)
        if length < 1e-6: continue
        
        branch_cyl = trimesh.creation.cylinder(radius=BRANCH_RADIUS, height=length, sections=6)
        transform = trimesh.geometry.align_vectors([0, 0, 1], np.hstack([vec, 0]))
        transform[:2, 3] = (start_node + end_node) / 2
        branch_cyl.apply_transform(transform)
        components.append(branch_cyl)

    # Add spheres at each joint for smoother connections
    for node in tree_nodes:
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=BRANCH_RADIUS * 1.1)
        sphere.apply_translation(np.hstack([node, 0]))
        components.append(sphere)
        
    if not components: return None
    
    return trimesh.boolean.union(components)


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

            # 3. Create the unique mycorrhizal pattern
            pattern = create_mycorrhizal_pattern(CAP_WIDTH, CAP_DEPTH)
            
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
