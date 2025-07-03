import trimesh
import numpy as np
from scipy.spatial import cKDTree

# --- PARAMETERS ---
# You can tweak these values to change the final appearance of the structure.

# Bounding box for the growth volume
BOUNDS = np.array([30, 30, 50])

# Number of attraction points to guide the growth. More points = denser structure.
N_ATTRACTORS = 4000

# The distance at which an attractor influences a branch
INFLUENCE_RADIUS = 8.0

# The distance at which an attractor is "consumed" and removed
KILL_RADIUS = 3.0

# The fixed length of each new branch segment
BRANCH_LENGTH = 1.0

# The thickness of the branches
BRANCH_RADIUS = 0.4

# The size and frequency of the "thorns" on the branches
SPIKE_DENSITY = 0.2  # Approximate number of spikes per unit of branch length
SPIKE_LENGTH = 1.2
SPIKE_RADIUS = 0.3

# The name for the output file
OUTPUT_FILENAME = 'organic_web_form.stl'


def create_spike(position, direction, length, radius, sections=8):
    """Creates a cone mesh oriented along a specific direction."""
    cone = trimesh.creation.cone(radius=radius, height=length, sections=sections)
    # Align the cone's axis (Z-axis) with the desired direction vector
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    transform[:3, 3] = position + direction * (length / 2)
    cone.apply_transform(transform)
    return cone

def generate_organic_structure():
    """
    Main function to generate the organic structure using a space colonization algorithm
    and export it as an STL file.
    """
    print("--- Starting Organic Form Generation ---")

    # --- 1. Initialize Attraction Points ---
    print(f"Step 1: Scattering {N_ATTRACTORS} attraction points...")
    attractors = (np.random.rand(N_ATTRACTORS, 3) - 0.5) * BOUNDS

    # --- 2. Initialize Tree Structure ---
    # We start with a single root node at the bottom center
    root_node = np.array([[0, 0, -BOUNDS[2] / 2]])
    tree_nodes = [root_node[0]]
    # Keep track of parent-child relationships for drawing branches
    tree_parents = {-1: -1} # Root has no parent
    
    # Use a k-d tree for efficiently finding nearest neighbors
    node_tree = cKDTree(root_node)
    
    print("Step 2: Growing the branching structure. This may take a while...")
    # --- 3. Run Growth Simulation ---
    # We iterate until the structure stops growing or for a max number of iterations
    for i in range(150): # Max iterations to prevent infinite loops
        if len(attractors) == 0:
            print("All attractors consumed. Growth finished.")
            break

        # For each node, find attractors that want to connect to it
        attractor_map = {}
        # Find the closest tree node for each attractor
        distances, closest_node_indices = cKDTree(tree_nodes).query(attractors)
        
        for attr_idx, node_idx in enumerate(closest_node_indices):
            if distances[attr_idx] < INFLUENCE_RADIUS:
                if node_idx not in attractor_map:
                    attractor_map[node_idx] = []
                attractor_map[node_idx].append(attractors[attr_idx])

        # Grow new branches based on the influence of the attractors
        new_nodes = []
        new_parents = {}
        for node_idx, influencing_attractors in attractor_map.items():
            parent_node = tree_nodes[node_idx]
            
            # Calculate the average direction towards the influencing attractors
            avg_direction = np.mean(np.array(influencing_attractors) - parent_node, axis=0)
            avg_direction /= np.linalg.norm(avg_direction)
            
            # Create a new node along this direction
            new_node = parent_node + avg_direction * BRANCH_LENGTH
            new_nodes.append(new_node)
            new_parents[len(tree_nodes) + len(new_nodes) - 1] = node_idx

        if not new_nodes:
            print(f"No new growth in iteration {i+1}. Halting.")
            break
            
        # Add the new nodes to our tree
        tree_nodes.extend(new_nodes)
        tree_parents.update(new_parents)
        
        # Remove attractors that are now too close to any node in the tree
        node_tree = cKDTree(tree_nodes)
        distances, _ = node_tree.query(attractors)
        attractors = attractors[distances > KILL_RADIUS]
        
        print(f"Iteration {i+1}: Grew {len(new_nodes)} new branches. {len(attractors)} attractors remaining.")

    print(f"\nStep 3: Growth complete. Generated {len(tree_nodes)} nodes.")
    
    # --- 4. Convert Skeleton to Mesh ---
    print("Step 4: Converting skeleton to a solid mesh with spikes...")
    components = []
    
    # Create cylinders for each branch and add spikes
    for child_idx, parent_idx in tree_parents.items():
        if parent_idx == -1: continue # Skip root's dummy parent
        
        start_node = tree_nodes[parent_idx]
        end_node = tree_nodes[child_idx]
        
        # Create the main branch cylinder
        vec = end_node - start_node
        length = np.linalg.norm(vec)
        if length == 0: continue
        direction = vec / length
        
        branch_cyl = trimesh.creation.cylinder(radius=BRANCH_RADIUS, height=length, sections=12)
        transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
        transform[:3, 3] = (start_node + end_node) / 2
        branch_cyl.apply_transform(transform)
        components.append(branch_cyl)

        # Add spikes along the branch
        num_spikes = int(length * SPIKE_DENSITY)
        for _ in range(num_spikes):
            # Get a random point along the branch
            pos_on_branch = start_node + direction * length * np.random.rand()
            # Get a random perpendicular direction for the spike
            random_vec = np.random.randn(3)
            spike_dir = np.cross(direction, random_vec)
            if np.linalg.norm(spike_dir) > 1e-6:
                spike_dir /= np.linalg.norm(spike_dir)
                spike = create_spike(pos_on_branch, spike_dir, SPIKE_LENGTH, SPIKE_RADIUS)
                components.append(spike)

    # Add spheres at each joint for smoother connections
    for node in tree_nodes:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=BRANCH_RADIUS * 1.1)
        sphere.apply_translation(node)
        components.append(sphere)

    # --- 5. Union all components and Export ---
    print("\nStep 5: Merging all components. This is the final and most intensive step...")
    if not components:
        print("Error: No components were generated. Cannot create mesh.")
        return
    
    # Using the default boolean engine to avoid dependency on Blender.
    # This resolves the "ImportError: `blender` is not in `PATH`" issue.
    final_mesh = trimesh.boolean.union(components)
    
    # Check if the mesh is watertight (good for 3D printing)
    if final_mesh.is_watertight:
        print("Generated mesh is watertight.")
    else:
        print("Warning: Generated mesh is not watertight. It may have holes.")
        # Attempt to fill holes
        final_mesh.fill_holes()

    # Export the final result
    final_mesh.export(OUTPUT_FILENAME)
    print(f"\n✅ Success! 3D form saved as '{OUTPUT_FILENAME}'")


if __name__ == '__main__':
    generate_organic_structure()
