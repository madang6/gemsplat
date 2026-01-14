#%%
import os
import yaml
import argparse
import numpy as np
import random
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Define the Node class
class Node:
    def __init__(self, position, parent=None, color=None):
        self.position = position  # NumPy array
        self.parent = parent      # Parent node
        self.children = []        # List of child nodes
        self.color = color        # Color assigned to this node
        self.cost = 0.0           # Cost from the start node

# Define the RRT class
class RRT:
    def __init__(self, start, bounds, obstacles, algorithm='RRT', dimension=3, step_size=1.0, max_iter=10000,
                  collision_check_resolution=0.1, exact_step=False, bounded_step = False, prevent_edge_overlap=False):
        
        self.algorithm = algorithm
        self.dimension = dimension
        self.exact_step = exact_step
        self.bounded_step = bounded_step
        self.prevent_edge_overlap = prevent_edge_overlap
        self.goal_node = Node(np.array(start), color="blue")
        self.bounds = bounds          # Bounds of the environment: List of (min, max) tuples for each dimension
        self.obstacles = obstacles    # List of obstacle functions
        self.step_size = step_size
        self.collision_check_resolution = collision_check_resolution
        self.max_iter = max_iter
        self.nodes = [self.goal_node]     # List to store all nodes
        self.edges = []  # List to store edges as (from_point, to_point, color)
        self.branch_colors = []  # To store colors for branches
        self.color_map = plt.cm.get_cmap('hsv', 10)  # Colormap with 10 distinct colors
        self.sampled_points = []  # List to store sampled points

        self.min_edge_separation = 0.1  # Adjust this value as needed

    def is_collision(self, point):
        # Collision checking against obstacles
        for obs in self.obstacles:
            if obs(point):
                return True
        return False
    
    def is_collision_free(self, from_point, to_point):
        # Check for collisions along the path from from_point to to_point
        distance = np.linalg.norm(to_point - from_point)
        num_samples = int(np.ceil(distance / self.collision_check_resolution))
        t_values = np.linspace(0, 1, num_samples)
        for t in t_values:
            point = from_point + t * (to_point - from_point)
            if self.is_collision(point):
                return False
        return True

    def sample_free(self):
        # Sample a random point within the bounds
        point = np.array([
            random.uniform(self.bounds[d][0], self.bounds[d][1])
            for d in range(self.dimension)
        ])
        return point

    def nearest(self, point):
        # Find the nearest node in the tree to the sampled point
        node_positions = np.array([node.position for node in self.nodes])
        tree = cKDTree(node_positions)
        _, idx = tree.query(point)
        return self.nodes[idx]

    def steer(self, from_node, to_point):
        # Move from from_node towards to_point by step_size
        direction = to_point - from_node.position
        length = np.linalg.norm(direction)
        if length == 0:
            return from_node.position
        
        if self.exact_step:
            # Make the new position exactly step_size away
            direction = (direction / length) * self.step_size
        elif self.bounded_step:
            # Make the new position fall within a range of values
            step_size = max(0.75, min(self.step_size, length))
            direction = (direction / length) * step_size
        else:
            
            direction = (direction / length) * min(self.step_size, length)
        new_position = from_node.position + direction
        return new_position

    def is_within_bounds(self, point):
        # Check if the point is within the environment bounds
        return all([
            self.bounds[d][0] <= point[d] <= self.bounds[d][1]
            for d in range(self.dimension)
        ])

    def at_edge(self, point):
        # Check if the point is at the edge of the environment
        return any([
            np.isclose(point[d], self.bounds[d][0]) or np.isclose(point[d], self.bounds[d][1])
            for d in range(self.dimension)
        ])
    
    def update_descendant_costs(self, node):
        for child in node.children:
            old_cost = child.cost
            # Update the cost of the child
            child.cost = node.cost + np.linalg.norm(child.position - node.position)
            # Continue updating costs for the descendants of the child
            self.update_descendant_costs(child)

    def update_subtree_color(self, node, new_color):
        """
        Recursively updates the color of a node and all its descendants.

        Args:
            node (Node): The node whose color needs to be updated.
            new_color: The new color to assign.
        """
        node.color = new_color
        for child in node.children:
            self.update_subtree_color(child, new_color)

    def build_rrt(self):
        node_positions = np.array([node.position for node in self.nodes])
        tree = cKDTree(node_positions)

        # Define the search radius for RRT*
        if self.algorithm == 'RRT*':
            # Radius can be tuned based on the space size and number of nodes
            gamma_rrt_star = 8 * (1 + 1 / self.dimension)**(1 / self.dimension)
            radius = min(gamma_rrt_star * (np.log(len(self.nodes)) / len(self.nodes))**(1 / self.dimension), self.step_size)
        else:
            radius = 0  # Not used in RRT

        rewire_count = 0  # Initialize rewiring counter
        total_distance_saved = 0.0  # Counter for total distance saved

        for i in range(self.max_iter):
            # Rebuild KD-tree periodically for efficiency
            if i % 1 == 0:
                node_positions = np.array([node.position for node in self.nodes])
                tree = cKDTree(node_positions)
                if self.algorithm == 'RRT*':
                    radius = gamma_rrt_star * (np.log(len(self.nodes)) / len(self.nodes))**(1 / self.dimension)

            rnd_point = self.sample_free()
            self.sampled_points.append(rnd_point)  # Store the sampled point

            nearest_node = self.nearest(rnd_point)
            new_position = self.steer(nearest_node, rnd_point)

            if not self.is_within_bounds(new_position):
                continue  # Skip if out of bounds

            if self.is_collision(new_position):
                continue  # Skip if collision detected

            # Determine the color for the new node
            if nearest_node.parent is None:
                # This is a direct child of the root
                branch_index = len(self.branch_colors)
                color = self.color_map(branch_index % 10)  # Cycle through colors
                self.branch_colors.append(color)
            else:
                # Inherit the color from the parent
                color = nearest_node.color

            # Initialize the new node's cost
            new_node_cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)

            # RRT* specific: Find the best parent from nearby nodes
            if self.algorithm == 'RRT*':
                # Query nodes within the search radius
                indices = tree.query_ball_point(new_position, r=radius)
                min_cost = new_node_cost
                best_parent = nearest_node

                for idx in indices:
                    candidate_node = self.nodes[idx]
                    tentative_cost = candidate_node.cost + np.linalg.norm(new_position - candidate_node.position)
                    if self.is_collision_free(candidate_node.position, new_position) and tentative_cost < min_cost:
                        min_cost = tentative_cost
                        best_parent = candidate_node

                # Update the new node's cost and parent
                new_node_cost = min_cost
                new_node = Node(new_position, parent=best_parent, color=best_parent.color)
                new_node.cost = new_node_cost
                best_parent.children.append(new_node)
                self.nodes.append(new_node)
                self.edges.append((best_parent.position, new_node.position, best_parent.color))

                print(f"Iteration {i}: Added new node at {new_node.position} with parent at {best_parent.position} (Cost: {new_node.cost:.2f})")

            else:
                # RRT: Use nearest node as parent
                new_node = Node(new_position, parent=nearest_node, color=color)
                nearest_node.children.append(new_node)
                self.nodes.append(new_node)
                self.edges.append((nearest_node.position, new_node.position, color))

                print(f"Iteration {i}: Added new node at {new_node.position} with parent at {nearest_node.position} (Cost: {new_node.cost:.2f})")

            # RRT* specific: Rewire the tree
            if self.algorithm == 'RRT*':
                # Query nodes within the search radius to potentially rewire
                indices = tree.query_ball_point(new_position, r=radius)
                for idx in indices:
                    candidate_node = self.nodes[idx]
                    if candidate_node is best_parent:
                        continue  # Skip the parent
                    tentative_cost = new_node.cost + np.linalg.norm(candidate_node.position - new_position)
                    if tentative_cost < candidate_node.cost and self.is_collision_free(new_position, candidate_node.position):
                        # Rewire: Change parent to new_node

                        old_parent = candidate_node.parent
                        if old_parent:
                            try:
                                old_parent.children.remove(candidate_node)
                            except ValueError:
                                pass  # Node already removed from parent's children
                            self.edges = [edge for edge in self.edges if not (
                                np.array_equal(edge[0], old_parent.position) and 
                                np.array_equal(edge[1], candidate_node.position)
                            )]
                        candidate_node.parent = new_node
                        candidate_node.cost = tentative_cost
                        new_node.children.append(candidate_node)
                        self.edges.append((new_node.position, candidate_node.position, new_node.color))

                        # **Update costs of descendants**
                        self.update_descendant_costs(candidate_node)

                        # **Update colors of candidate_node and its descendants**
                        self.update_subtree_color(candidate_node, new_node.color)

                        # **Calculate distance saved**
                        distance_saved = (old_parent.cost + np.linalg.norm(candidate_node.position - old_parent.position)) - candidate_node.cost
                        total_distance_saved += distance_saved

                        # **Print Statement: Rewiring Action**
                        print(f"Iteration {i}: Rewired node at ({candidate_node.position[0]:.2f}, {candidate_node.position[1]:.2f}) "
                            f"from parent at ({old_parent.position[0]:.2f}, {old_parent.position[1]:.2f}) "
                            f"to new parent at ({new_node.position[0]:.2f}, {new_node.position[1]:.2f}) "
                            f"(Cost Reduced by: {distance_saved:.2f})")
                        
                        rewire_count += 1

            # Stop expanding if the new node is at the edge
            if self.at_edge(new_position):
                continue  # You can choose to stop expanding in this direction

        print("RRT/RRT* construction completed.")
        print(f"Total Nodes Added: {len(self.nodes)}")
        print(f"Total Rewiring Actions: {rewire_count}")
        print(f"Total Distance Saved: {total_distance_saved:.2f} units")

    def prune_redundant_leaves(self, exclusion_radius):
        """
        Prunes leaf nodes and their parent edges if the leaf node lies within a fixed radius
        of any other node in the tree (excluding its parent). Continues recursively until no more nodes can be pruned.
        """
        print("Starting pruning of redundant leaf nodes...")
        nodes_pruned = True
        iteration = 0
        while nodes_pruned:
            nodes_pruned = False
            iteration += 1
            leaf_nodes = [node for node in self.nodes if not node.children and node.parent is not None]
            if not leaf_nodes:
                break  # No more leaf nodes to check

            # Build KD-tree for current node positions
            node_positions = np.array([node.position for node in self.nodes])
            node_tree = cKDTree(node_positions)

            nodes_to_remove = set()
            for node in leaf_nodes:
                if node in nodes_to_remove:
                    continue

                # Exclude parent node in the proximity check
                parent_position = node.parent.position

                # Query nodes within exclusion radius
                indices = node_tree.query_ball_point(node.position, r=exclusion_radius)
                nearby_nodes = [self.nodes[idx] for idx in indices if self.nodes[idx] is not node and self.nodes[idx] is not node.parent]

                if nearby_nodes:
                    # Mark the leaf node for removal
                    nodes_to_remove.add(node)
                    nodes_pruned = True  # Set flag to continue pruning
                    print(f"Iteration {iteration}: Pruned leaf node at {node.position} due to proximity to other nodes.")

            # Remove the nodes and their parent edges
            for node in nodes_to_remove:
                self.remove_node_and_edge(node)

        print("Pruning completed.")

    def get_node_depths(self):
        """
        Returns a dictionary mapping nodes to their depth from the root node.
        """
        node_depths = {}
        queue = [(self.goal_node, 0)]  # Assuming goal_node is the root
        while queue:
            current_node, depth = queue.pop(0)
            node_depths[current_node] = depth
            for child in current_node.children:
                queue.append((child, depth + 1))
        return node_depths
    
    def remove_node_and_edge(self, node):
        """
        Removes the node and the edge connecting it to its parent.
        """
        parent = node.parent
        if parent:
            try:
                parent.children.remove(node)
            except ValueError:
                pass  # Node already removed from parent's children
            # Remove the edge between parent and node
            self.edges = [edge for edge in self.edges if not (
                (np.array_equal(edge[0], parent.position) and np.array_equal(edge[1], node.position)) or
                (np.array_equal(edge[0], node.position) and np.array_equal(edge[1], parent.position))
            )]
        # Remove the node from the tree
        if node in self.nodes:
            self.nodes.remove(node)

    def get_path_from_leaf_to_root(self, leaf_node):
        """
        Returns the list of positions along the path from the given leaf node to the root node.

        Args:
            leaf_node (Node): The leaf node from which to start.

        Returns:
            path (list of numpy arrays): The list of positions from root to leaf.
        """
        path = []
        current_node = leaf_node
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        # Reverse the path to go from root to leaf
        path.reverse()
        return path

    def visualize(self, show_sampled_points=False):
        if self.dimension == 1:
            self._visualize_1d()
        elif self.dimension == 2:
            self._visualize_2d(show_sampled_points)
        elif self.dimension == 3:
            self._visualize_3d()
        else:
            print(f"Visualization not supported for dimension {self.dimension}")

    def _visualize_2d(self, show_sampled_points=False):
        fig, ax = plt.subplots()

        # Plot the obstacles
        for obs in self.obstacles:
            obs.plot(ax)

        # Plot edges next
        for edge in self.edges:
            xs = [edge[0][0], edge[1][0]]
            ys = [edge[0][1], edge[1][1]]
            ax.plot(xs, ys, color=edge[2], linewidth=2, zorder=2)

        # Plot the nodes
        node_positions = np.array([node.position for node in self.nodes])
        node_colors = [node.color for node in self.nodes]
        ax.scatter(node_positions[:, 0], node_positions[:, 1],
        color=node_colors, s=5, zorder=3, label='Nodes')

        if show_sampled_points:
            # Plot the sampled points with a gradient color map
            sampled_positions = np.array(self.sampled_points)
            num_samples = len(self.sampled_points)
            # Normalize the sample indices to [0,1]
            norm = Normalize(vmin=0, vmax=num_samples - 1)
            # Choose a colormap
            cmap = plt.cm.viridis  # You can choose any colormap you like
            # Map the normalized indices to colors
            # colors = cmap(norm(range(num_samples)))
            colors = "gray"
            # Plot the sampled points
            ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1],
                    color=colors, s=2, marker='.', zorder=4, label='Sampled Points')

        # Plot the start node
        ax.scatter(self.goal_node.position[0], self.goal_node.position[1], color='red', s=50, zorder=5, label='Final Location')

        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
        # plt.show()

    def _visualize_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for node in self.nodes:
            if node.parent is not None:
                xs = [node.parent.position[0], node.position[0]]
                ys = [node.parent.position[1], node.position[1]]
                zs = [node.parent.position[2], node.position[2]]
                ax.plot(xs, ys, zs, color='blue')

        # Plot the obstacles
        for obs in self.obstacles:
            obs.plot(ax)

        # Plot the start node
        ax.scatter(
            self.start.position[0],
            self.start.position[1],
            self.start.position[2],
            color='red',
            s=100,
            label='Final Location'
        )

        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.set_zlim(self.bounds[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()

# Define obstacle functions
class Obstacle:
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, point):
        # Generic obstacle function
        return False

    def plot(self, ax):
        # Generic plot function
        pass

class CircleObstacle(Obstacle):
    def __init__(self, center, radius):
        super().__init__(dimension=2)
        self.center = np.array(center)
        self.radius = radius

    def __call__(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

    def plot(self, ax):
        circle = plt.Circle(self.center, self.radius, color='r', alpha=0.3)
        ax.add_patch(circle)

class SphereObstacle(Obstacle):
    def __init__(self, center, radius):
        super().__init__(dimension=3)
        self.center = np.array(center)
        self.radius = radius

    def __call__(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

    def plot(self, ax):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.center[0] + self.radius * np.cos(u) * np.sin(v)
        y = self.center[1] + self.radius * np.sin(u) * np.sin(v)
        z = self.center[2] + self.radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color='r', alpha=0.3)

def main():
    # Function to get a configuration option with a default value and prompt
    def get_config_option(option_name, prompt, valid_options=None, default=None):
        if option_name in config:
            value = config[option_name]
            print(f"{option_name} set to {value} from config.yml")
            if valid_options and value not in valid_options:
                print(f"Invalid value for {option_name} in config.yml. Using default or prompting.")
                value = None
        else:
            value = None

        if value is None:
            value = input(prompt).strip()
            if valid_options:
                while value not in valid_options:
                    print(f"Invalid input. Valid options are: {', '.join(valid_options)}")
                    value = input(prompt).strip()
            if default and not value:
                value = default
        return value
    
    # Path to the configuration file
    config_file = '/home/admin/StanfordMSL/gemsplat/gemsplat/scripts/rrt3config.yml'

    # Initialize an empty dictionary for config
    config = {}

    # Check if the configuration file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) or {}
        print("Configuration loaded from config.yml")
    else:
        print("Configuration file not found. Proceeding with interactive inputs.")

    # Ask the user to select the algorithm
    algorithm_input = get_config_option(
        'algorithm',
        "Select algorithm (RRT/RRT*): ",
        valid_options=['RRT', 'RRT*'],
        default='RRT'
    ).upper()

    # Select dimension: 1, 2, or 3
    dimension = get_config_option(
        'dimension',
        "Enter the dimension (2 or 3): ",
        valid_options=['2', '3'],
        default='2'
    )
    dimension = int(dimension)

    # Choose whether to prevent edge overlap
    prevent_edge_overlap_input = get_config_option(
        'prevent_edge_overlap',
        "Prevent edge overlap? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    prevent_edge_overlap = prevent_edge_overlap_input.lower() == 'y'

    # Choose whether to use exact edge lengths
    exact_step_input = get_config_option(
        'exact_step',
        "Exact edge lengths? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    exact_step = exact_step_input.lower() == 'y'

    bounded_step = False
    use_branch_pruning = False
    if not exact_step:
        # Choose whether to bound edge lengths
        bounded_step_input = get_config_option(
            'bounded_step',
            "Bound edge lengths? (y/n): ",
            valid_options=['y', 'n'],
            default='y'
        )
        bounded_step = bounded_step_input.lower() == 'y'

    # Define environment bounds based on dimension
    if dimension == 2:
        bounds = [(-4.274513955108515, -4.003210651313112), (7.732183279589786, 3.396624125902586)]
        final_location = [2.35034751, 3.05062047]
        obstacles = [
            # CircleObstacle(center=[0, 1.5], radius=1.0),
            CircleObstacle(center=[1.0, -0.5], radius=0.75),
            CircleObstacle(center=[-2.0, -1.5], radius=0.75),
            CircleObstacle(center=[2.0, 1.5], radius=1.0)
        ]
    elif dimension == 3:
        bounds = [(0, 100), (0, 100), (0, 100)]
        final_location = [50, 50, 50]
        obstacles = [
            SphereObstacle(center=[30, 30, 30], radius=10),
            SphereObstacle(center=[70, 70, 70], radius=15),
        ]
    else:
        print(f"Dimension {dimension} is not supported.")
        return

    # Initialize RRT
    rrt = RRT(
        start=final_location,
        bounds=bounds,
        obstacles=obstacles,
        dimension=dimension,
        step_size=1.0,
        collision_check_resolution=0.1,
        max_iter=2500,
        exact_step=exact_step,
        bounded_step=bounded_step,
        algorithm=algorithm_input,  # Pass the selected algorithm
        prevent_edge_overlap=prevent_edge_overlap
    )

    # Build RRT
    rrt.build_rrt()
    rrt.visualize(show_sampled_points=True)

    # Prune redundant leaf nodes
    # exclusion_radius = 1.5  # Define your exclusion radius
    # rrt.prune_redundant_leaves(exclusion_radius)

    # Prune the tree
    # rrt.post_process_tree(min_edge_separation=1.0, min_leaf_separation=1.0)

    # Visualize the RRT
    # rrt.visualize(show_sampled_points=True)
    print(f"Algorithm Choice: {algorithm_input}, Dimension Choice: {dimension}, Exact Step: {exact_step}, Bounded Step: {bounded_step}")

#%%
if __name__ == "__main__":
    #%%
    main()
#%%