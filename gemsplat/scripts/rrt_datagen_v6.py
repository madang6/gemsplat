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
                 exact_step=False, collision_check_resolution=0.1, bounded_step = False, use_branch_pruning=False,
                 use_leaf_pruning=False, prevent_edge_overlap=False):
        self.algorithm = algorithm
        self.dimension = dimension
        self.exact_step = exact_step
        self.bounded_step = bounded_step
        self.use_branch_pruning = use_branch_pruning
        self.use_leaf_pruning = use_leaf_pruning  # Controls pruning
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

    def edge_key(edge):
        # Round positions to a fixed number of decimal places
        start_pos = tuple(np.round(edge[0], decimals=5))
        end_pos = tuple(np.round(edge[1], decimals=5))
        return (start_pos, end_pos)

    def get_leaf_nodes(self):
        # Identify leaf nodes (nodes with no children)
        leaf_nodes = [node for node in self.nodes if len(node.children) == 0]
        return leaf_nodes

    def get_node_depth(self, node):
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def find_node_by_position(self, position):
        for node in self.nodes:
            if np.allclose(node.position, position, atol=1e-5):
                return node
        return None

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
    
    def edges_overlap(self, from_point, to_point):
        # Check if the new edge overlaps with any existing edge
        for edge in self.edges:
            edge_from = edge[0]
            edge_to = edge[1]
            if self.edge_intersect(from_point, to_point, edge_from, edge_to):
                return True
        return False

    def edge_intersect(self, p1, p2, q1, q2):
        # For 2D, implement line segment intersection
        if self.dimension == 2:
            return self.line_segments_intersect_2d(p1, p2, q1, q2)
        else:
            # For higher dimensions, approximate overlap by checking proximity
            return self.lines_close(p1, p2, q1, q2)

    def line_segments_intersect_2d(self, p1, p2, q1, q2):
        # Check if line segments p1-p2 and q1-q2 intersect
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    def lines_close(self, p1, p2, q1, q2, threshold=0.01):
        # Check if lines are closer than a threshold
        dist = self.min_distance_between_segments(p1, p2, q1, q2)
        return dist < threshold
    
    def do_edges_intersect(self, a_start, a_end, b_start, b_end):
        if self.dimension == 2:
            # Edge intersection in 2D using the CCW algorithm
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            A = a_start
            B = a_end
            C = b_start
            D = b_end

            return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
        else:
            # For other dimensions, implement appropriate method or return False
            return False  # Placeholder for 1D and 3D
        
    def edge_to_edge_distance(self, a_start, a_end, b_start, b_end):
        if self.dimension == 2:
            # Calculate the minimum distance between two line segments in 2D
            # If the segments intersect, the distance is zero
            if self.do_edges_intersect(a_start, a_end, b_start, b_end):
                return 0.0
            # Compute the minimum distance between endpoints and the opposite segment
            distances = [
                self.point_to_segment_distance_2d(a_start, b_start, b_end),
                self.point_to_segment_distance_2d(a_end, b_start, b_end),
                self.point_to_segment_distance_2d(b_start, a_start, a_end),
                self.point_to_segment_distance_2d(b_end, a_start, a_end),
            ]
            return min(distances)
        else:
            # For other dimensions, implement appropriate method or return a large value
            return float('inf')  # Placeholder for 1D and 3D
        
    def point_to_segment_distance(self, point, seg_start, seg_end):
        # Compute the distance from a point to a line segment in your environment's dimension
        # Here's an example for 2D:
        point = np.array(point)
        seg_start = np.array(seg_start)
        seg_end = np.array(seg_end)

        if np.array_equal(seg_start, seg_end):
            return np.linalg.norm(point - seg_start)

        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        seg_len = np.linalg.norm(seg_vec)
        seg_unit = seg_vec / seg_len

        projection = np.dot(point_vec, seg_unit)
        if projection < 0:
            closest_point = seg_start
        elif projection > seg_len:
            closest_point = seg_end
        else:
            closest_point = seg_start + projection * seg_unit

        return np.linalg.norm(point - closest_point)
    
    def distance_point_to_segment(self, point, segment_start, segment_end):
        # Vector from segment_start to segment_end
        v = segment_end - segment_start
        # Vector from segment_start to point
        w = point - segment_start

        c1 = np.dot(w, v)
        c2 = np.dot(v, v)

        if c2 == 0:
            # segment_start and segment_end are the same point
            return np.linalg.norm(point - segment_start)

        b = c1 / c2
        if b < 0:
            # Closest point is segment_start
            return np.linalg.norm(point - segment_start)
        elif b > 1:
            # Closest point is segment_end
            return np.linalg.norm(point - segment_end)
        else:
            # Closest point is within the segment
            pb = segment_start + b * v
            return np.linalg.norm(point - pb)

    def min_distance_between_segments(self, p1, p2, q1, q2):
        # Compute minimum distance between two line segments in higher dimensions
        u = p2 - p1
        v = q2 - q1
        w0 = p1 - q1
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

        denominator = a * c - b * b
        if denominator == 0:
            sc, tc = 0.0, d / b if b != 0 else 0.0
        else:
            sc = (b * e - c * d) / denominator
            tc = (a * e - b * d) / denominator

        sc = np.clip(sc, 0.0, 1.0)
        tc = np.clip(tc, 0.0, 1.0)

        point_on_p = p1 + sc * u
        point_on_q = q1 + tc * v
        distance = np.linalg.norm(point_on_p - point_on_q)
        return distance
    
    def is_leaf_near_edge(self, leaf_node, edge_proximity_threshold):
        leaf_pos = leaf_node.position

        # Collect edges excluding the one connected to the leaf node
        connected_edge = None
        if leaf_node.parent:
            connected_edge = (tuple(np.round(leaf_node.parent.position, decimals=5)),
                            tuple(np.round(leaf_node.position, decimals=5)))

        for edge in self.edges:
            edge_start, edge_end, _ = edge  # Assuming edge is (from_point, to_point, color)
            edge_key = (tuple(np.round(edge_start, decimals=5)), tuple(np.round(edge_end, decimals=5)))

            if connected_edge and (edge_key == connected_edge or edge_key == connected_edge[::-1]):
                continue  # Skip the edge connected to the leaf node

            # Calculate distance from leaf node to the edge
            distance = self.distance_point_to_segment(leaf_pos, np.array(edge_start), np.array(edge_end))

            if distance <= edge_proximity_threshold:
                return True  # Leaf node is near another edge

        return False  # Leaf node is not near any other edges
    
    def prune_tree(self, distance_threshold=1.0, edge_proximity_threshold=1.0):
        """
        Prunes the RRT* tree by:
        - Removing leaves within a certain distance of each other.
        - Removing leaves that are close to existing edges.
        - Removing overlapping edges.

        Args:
            distance_threshold (float): Distance threshold for pruning leaves close to each other.
            edge_proximity_threshold (float): Distance threshold for pruning leaves near edges.
        """
        total_leaves_pruned = 0

        # Remove overlapping edges
        self.remove_overlapping_edges()

        # Prune leaves considering proximity between leaves and edges
        while True:
            leaves_pruned = self.prune_leaves(distance_threshold, edge_proximity_threshold)
            total_leaves_pruned += leaves_pruned
            if leaves_pruned == 0:
                break  # Exit the loop when no more leaves are pruned

        print(f"Total leaves pruned: {total_leaves_pruned}")
    
    def prune_leaves(self, distance_threshold=1.0, edge_proximity_threshold=1.0):
        """
        Prunes leaves based on proximity to other leaves and edges.

        Returns:
            int: Number of leaves pruned in this pass.
        """
        leaf_nodes = self.get_leaf_nodes()
        print(f"Number of leaf nodes before pruning: {len(leaf_nodes)}")

        if len(leaf_nodes) <= 1:
            print("No pruning needed.")
            return 0  # Nothing to prune

        # Build a KD-tree of leaf node positions
        leaf_positions = np.array([node.position for node in leaf_nodes])
        leaf_tree = cKDTree(leaf_positions)

        # Keep track of nodes to remove
        nodes_to_remove = set()
        for i, node in enumerate(leaf_nodes):
            if node in nodes_to_remove:
                continue  # Node already marked for removal

            # Check proximity to other leaf nodes
            idxs = leaf_tree.query_ball_point(node.position, r=distance_threshold)
            for idx in idxs:
                neighbor_node = leaf_nodes[idx]
                if neighbor_node is node or neighbor_node in nodes_to_remove:
                    continue  # Skip self and already marked nodes
                # Decide which node to remove using hybrid criteria
                depth_node = self.get_node_depth(node)
                depth_neighbor = self.get_node_depth(neighbor_node)
                if depth_node < depth_neighbor:
                    # Node is closer to root; keep node, remove neighbor
                    nodes_to_remove.add(neighbor_node)
                elif depth_node > depth_neighbor:
                    # Neighbor is closer to root; remove node
                    nodes_to_remove.add(node)
                    break  # Current node is marked for removal; no need to check further
                else:
                    # Depths are equal; compare costs
                    cost_node = node.cost
                    cost_neighbor = neighbor_node.cost
                    if cost_node <= cost_neighbor:
                        # Node has lower cost; keep node, remove neighbor
                        nodes_to_remove.add(neighbor_node)
                    else:
                        # Neighbor has lower cost; remove node
                        nodes_to_remove.add(node)
                        break  # Current node is marked for removal

            if node in nodes_to_remove:
                continue  # Node is already marked for removal

            # Check proximity to other edges
            if self.is_leaf_near_edge(node, edge_proximity_threshold):
                nodes_to_remove.add(node)

        if not nodes_to_remove:
            print("No more leaves to prune.")
            return 0  # No leaves pruned in this pass

        print(f"Number of leaves to remove in this pass: {len(nodes_to_remove)}")

        # Remove the marked leaf nodes
        for node in nodes_to_remove:
            self.remove_node(node)

        return len(nodes_to_remove)

    def remove_node(self, node):
        if node.parent:
            try:
                node.parent.children.remove(node)
            except ValueError:
                pass  # Node already removed from parent's children
        # Remove edges associated with this node
        self.edges = [edge for edge in self.edges if not (
            np.array_equal(edge[0], node.position) or np.array_equal(edge[1], node.position)
        )]
        # Remove the node from the nodes list
        self.nodes.remove(node)

    def remove_overlapping_edges(self):
        # Use a set to track unique edges
        unique_edges = set()
        edges_to_remove = []
        for edge in self.edges:
            # Create a tuple of sorted node positions to handle undirected edges
            edge_k = (tuple(np.round(edge[0], decimals=5)), tuple(np.round(edge[1], decimals=5)))
            if edge_key in unique_edges or edge_key[::-1] in unique_edges:
                # Edge is a duplicate; mark for removal
                edge1_key = edge_key(edge1)
                edge2_key = edge_key(edge2)
                edges_to_remove.add(edge_to_remove_key)
            else:
                unique_edges.add(edge_key)
        # Remove duplicate edges
        for edge in edges_to_remove:
            self.edges.remove(edge)
            # Also update nodes' children and parents if necessary
            start_node = self.find_node_by_position(edge[0])
            end_node = self.find_node_by_position(edge[1])
            if start_node and end_node:
                if end_node in start_node.children:
                    start_node.children.remove(end_node)
                if start_node is end_node.parent:
                    end_node.parent = None

    def update_descendant_costs(self, node):
        for child in node.children:
            old_cost = child.cost
            # Update the cost of the child
            child.cost = node.cost + np.linalg.norm(child.position - node.position)
            # Continue updating costs for the descendants of the child
            self.update_descendant_costs(child)

    def get_best_path_nodes(self):
        """
        Retrieves all nodes that form the paths from the goal node to the leaf nodes.
        """
        path_nodes = set()
        goal_node = self.goal_node  # Assuming self.goal_node is your central/root node

        if goal_node is None:
            return path_nodes  # No nodes to collect

        # Use a depth-first search (DFS) to traverse from the goal node to all leaf nodes
        stack = [goal_node]
        while stack:
            current_node = stack.pop()
            path_nodes.add(current_node)
            for child in current_node.children:
                stack.append(child)

        return path_nodes

    def get_edge_cost(self, edge):
        start_pos, end_pos, _ = edge
        return np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    
    def get_optimal_path_edges(self):
        path_nodes = self.get_best_path_nodes()
        optimal_edges = set()
        for node in path_nodes:
            if node.parent:
                parent_pos = tuple(node.parent.position)
                node_pos = tuple(node.position)
                edge = (parent_pos, node_pos)
                optimal_edges.add(edge)
        return optimal_edges

    def remove_edge(self, edge):
        start_pos = tuple(edge[0])
        end_pos = tuple(edge[1])
        # Remove edge from self.edges
        self.edges = [e for e in self.edges if not (
            tuple(e[0]) == start_pos and tuple(e[1]) == end_pos
        )]
        # Update node relationships
        start_node = self.find_node_by_position(start_pos)
        end_node = self.find_node_by_position(end_pos)
        if start_node and end_node:
            if end_node in start_node.children:
                start_node.children.remove(end_node)
            if start_node is end_node.parent:
                end_node.parent = None
                # Optionally remove disconnected nodes

    def select_edge_to_remove(self, edge1, edge2):
        optimal_path_edges = self.get_optimal_path_edges()
        # Convert edges to comparable formats
        edge1_key = (tuple(edge1[0]), tuple(edge1[1]))
        edge2_key = (tuple(edge2[0]), tuple(edge2[1]))
        optimal_edges_keys = {(tuple(e[0]), tuple(e[1])) for e in optimal_path_edges}

        if edge1_key in optimal_edges_keys and edge2_key not in optimal_edges_keys:
            return edge2
        elif edge2_key in optimal_edges_keys and edge1_key not in optimal_edges_keys:
            return edge1
        else:
            # Use other criteria
            edge1_cost = self.get_edge_cost(edge1)
            edge2_cost = self.get_edge_cost(edge2)
            return edge1 if edge1_cost > edge2_cost else edge2
    
    def do_edges_intersect(self, a_start, a_end, b_start, b_end):
        # Use the CCW algorithm for 2D
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A = a_start
        B = a_end
        C = b_start
        D = b_end

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
    
    def remove_intersecting_edges(self):
        """
        Removes edges that intersect with others in the tree.
        """
        print("Removing intersecting edges...")
        edges_to_remove = set()
        edge_list = self.edges.copy()

        # Build a mapping from hashable edges to actual edges
        edge_mapping = {}
        for edge in self.edges:
            key = edge_key(edge)
            edge_mapping[key] = edge

        # For simplicity, we'll check all pairs (could be optimized)
        for i, edge1 in enumerate(edge_list):
            for edge2 in edge_list[i+1:]:
                edge1_start, edge1_end, _ = edge1
                edge2_start, edge2_end, _ = edge2

                if self.do_edges_intersect(edge1_start, edge1_end, edge2_start, edge2_end):
                    # Decide which edge to remove
                    edge_to_remove = self.select_edge_to_remove(edge1, edge2)
                    # Convert edge to hashable form
                    edge_key = (tuple(edge_to_remove[0]), tuple(edge_to_remove[1]))
                    edges_to_remove.add(edge_key)

        # Remove the intersecting edges
        for edge_key in edges_to_remove:
            edge = edge_mapping.get(edge_key)
            if edge:
                self.remove_edge(edge)

        print(f"Removed {len(edges_to_remove)} intersecting edges.")


    def edge_to_edge_distance(self, a_start, a_end, b_start, b_end):
        # For 2D or 3D, compute the minimum distance between two line segments
        # Use an appropriate algorithm for your environment's dimension
        # Here's a placeholder for 2D:

        # If edges intersect, distance is zero
        if self.do_edges_intersect(a_start, a_end, b_start, b_end):
            return 0.0

        # Compute distances between endpoints and opposite segments
        distances = [
            self.point_to_segment_distance(a_start, b_start, b_end),
            self.point_to_segment_distance(a_end, b_start, b_end),
            self.point_to_segment_distance(b_start, a_start, a_end),
            self.point_to_segment_distance(b_end, a_start, a_end),
        ]
        return min(distances)

    def enforce_minimum_edge_separation(self, min_edge_separation):
        """
        Removes edges that are closer than the minimum separation distance to other edges.
        """
        print("Enforcing minimum edge separation...")
        edges_to_remove = set()
        edge_list = self.edges.copy()
        edge_mapping = {}
        
        # Create edge mapping with consistent keys
        for edge in self.edges:
            key = edge_key(edge)
            edge_mapping[key] = edge

        for i, edge1 in enumerate(edge_list):
            for edge2 in edge_list[i+1:]:
                edge1_key = edge_key(edge1)
                edge2_key = edge_key(edge2)

                if edge1_key in edges_to_remove or edge2_key in edges_to_remove:
                    continue  # Skip edges already marked for removal

                edge1_start, edge1_end, _ = edge1
                edge2_start, edge2_end, _ = edge2

                distance = self.edge_to_edge_distance(edge1_start, edge1_end, edge2_start, edge2_end)
                if distance < min_edge_separation:
                    # Decide which edge to remove
                    edge_to_remove = self.select_edge_to_remove(edge1, edge2)
                    edge_to_remove_key = edge_key(edge_to_remove)
                    edges_to_remove.add(edge_to_remove_key)
                    print(f"Edge {edge_to_remove_key} marked for removal (distance {distance:.5f})")

        # Remove the edges
        for edge_key in edges_to_remove:
            edge = edge_mapping.get(edge_key)
            if edge:
                self.remove_edge(edge)
            else:
                print(f"Edge {edge_key} not found in edge_mapping.")

        print(f"Removed {len(edges_to_remove)} edges due to minimum separation enforcement.")

    def is_node_near_edges(self, node, min_distance):
        node_pos = node.position
        for edge in self.edges:
            edge_start, edge_end, _ = edge
            # Skip the edge connected to the node itself
            if (np.array_equal(edge_start, node_pos) or np.array_equal(edge_end, node_pos)):
                continue

            distance = self.point_to_segment_distance(node_pos, edge_start, edge_end)
            if distance < min_distance:
                return True
        return False

    def select_node_to_remove(self, node1, node2):
        # Prefer to keep nodes that are part of the optimal path
        optimal_path_nodes = set(self.get_best_path_nodes())
        if node1 in optimal_path_nodes and node2 not in optimal_path_nodes:
            return node2
        elif node2 in optimal_path_nodes and node1 not in optimal_path_nodes:
            return node1
        else:
            # Compare node depths or costs
            depth1 = self.get_node_depth(node1)
            depth2 = self.get_node_depth(node2)
            if depth1 < depth2:
                return node2
            elif depth2 < depth1:
                return node1
            else:
                return node2 if node1.cost <= node2.cost else node1

    def prune_leaves_based_on_proximity(self, min_leaf_separation):
        """
        Prunes leaf nodes that are too close to each other or to other edges.
        """
        print("Pruning leaf nodes based on proximity...")
        leaves_to_remove = set()
        leaf_nodes = self.get_leaf_nodes()
        leaf_positions = np.array([node.position for node in leaf_nodes])

        # Build KD-tree for leaf nodes
        leaf_tree = cKDTree(leaf_positions)

        # Prune leaves close to other leaves
        for i, node in enumerate(leaf_nodes):
            if node in leaves_to_remove:
                continue

            idxs = leaf_tree.query_ball_point(node.position, r=min_leaf_separation)
            for idx in idxs:
                neighbor_node = leaf_nodes[idx]
                if neighbor_node is node or neighbor_node in leaves_to_remove:
                    continue

                # Decide which leaf to remove
                node_to_remove = self.select_node_to_remove(node, neighbor_node)
                leaves_to_remove.add(node_to_remove)

        # Prune leaves close to edges
        for node in leaf_nodes:
            if node in leaves_to_remove:
                continue

            # Check proximity to edges
            if self.is_node_near_edges(node, min_leaf_separation):
                leaves_to_remove.add(node)

        # Remove the leaves
        for node in leaves_to_remove:
            self.remove_node(node)

        print(f"Pruned {len(leaves_to_remove)} leaf nodes based on proximity.")

    def build_rrt(self):
         # Initialize the tree with the goal node as the root
        # self.goal_node = self.start #Node(np.array(self.start), color=self.root_color)
        # self.nodes = [self.goal_node]
        # self.goal_node.parent = None
        # self.goal_node.cost = 0.0
        # self.goal_node.color = self.root_color  # Assign a color if needed
        
        node_positions = np.array([node.position for node in self.nodes])
        tree = cKDTree(node_positions)

        # Define the search radius for RRT*
        if self.algorithm == 'RRT*':
            # Radius can be tuned based on the space size and number of nodes
            gamma_rrt_star = 6 * (1 + 1 / self.dimension)**(1 / self.dimension)
            radius = gamma_rrt_star * (np.log(len(self.nodes)) / len(self.nodes))**(1 / self.dimension)
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


    def post_process_tree(self, min_edge_separation=0.1, min_leaf_separation=1.0):
        """
        Post-process the RRT* tree to:
        1. Remove intersecting edges.
        2. Enforce minimum separation between edges.
        3. Prune leaf nodes based on proximity to other leaves and edges.

        Args:
            min_edge_separation (float): Minimum separation distance between edges.
            min_leaf_separation (float): Minimum separation distance between leaves and between leaves and edges.
        """
        print("Starting post-processing of the RRT* tree...")
        self.remove_intersecting_edges()
        self.enforce_minimum_edge_separation(min_edge_separation)
        self.prune_leaves_based_on_proximity(min_leaf_separation)
        print("Post-processing completed.")

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


    def visualize(self, show_sampled_points=False):
        if self.dimension == 1:
            self._visualize_1d()
        elif self.dimension == 2:
            self._visualize_2d(show_sampled_points)
        elif self.dimension == 3:
            self._visualize_3d()
        else:
            print(f"Visualization not supported for dimension {self.dimension}")

        # #FIXME If RRT*, highlight the best path
        # if self.algorithm == 'RRT*':
        #     best_path = self.get_best_path()
        #     if best_path:
        #         best_path = np.array(best_path)
        #         if self.dimension == 1:
        #             plt.plot(best_path[:, 0], np.zeros_like(best_path[:, 0]), color='magenta', linewidth=3, label='Best Path')
        #         elif self.dimension == 2:
        #             plt.plot(best_path[:, 0], best_path[:, 1], color='magenta', linewidth=3, label='Best Path')
        #         elif self.dimension == 3:
        #             ax = plt.gca(projection='3d')
        #             ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], color='magenta', linewidth=3, label='Best Path')
        #         plt.legend()
        
        plt.show()

    # def visualize(self, show_sampled_points=False):
    #     # Visualize the RRT based on the dimension
    #     if self.dimension == 1:
    #         self._visualize_1d()
    #     elif self.dimension == 2:
    #         self._visualize_2d(show_sampled_points)
    #     elif self.dimension == 3:
    #         self._visualize_3d()
    #     else:
    #         print(f"Visualization not supported for dimension {self.dimension}")

    def _visualize_1d(self):
        fig, ax = plt.subplots()
        for node in self.nodes:
            if node.parent is not None:
                xs = [node.parent.position[0], node.position[0]]
                ys = [0, 0]  # All nodes lie along y=0
                ax.plot(xs, ys, color='blue')

        # Plot the obstacles
        for obs in self.obstacles:
            obs.plot(ax)

        # Plot the start node
        ax.scatter(self.start.position[0], 0, color='red', s=50, label='Final Location')

        ax.set_xlim(self.bounds[0])
        ax.set_ylim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
        plt.show()

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

class IntervalObstacle(Obstacle):
    def __init__(self, start, end):
        super().__init__(dimension=1)
        self.start = start
        self.end = end

    def __call__(self, point):
        return self.start <= point[0] <= self.end

    def plot(self, ax):
        ax.axvspan(self.start, self.end, color='r', alpha=0.3)

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
        "Enter the dimension (1, 2, or 3): ",
        valid_options=['1', '2', '3'],
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

        # Choose whether to prune branches smaller than a given length
        branch_pruning_input = get_config_option(
            'use_branch_pruning',
            "Prune branches smaller than a given length? (y/n): ",
            valid_options=['y', 'n'],
            default='n'
        )
        use_branch_pruning = branch_pruning_input.lower() == 'y'

    # Ask the user whether to use leaf pruning
    use_leaf_pruning_input = get_config_option(
        'use_leaf_pruning',
        "Use leaf pruning? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    use_leaf_pruning = use_leaf_pruning_input.lower() == 'y'

    # Define environment bounds based on dimension
    if dimension == 1:
        bounds = [(0, 100)]  # List of tuples for each dimension
        final_location = [50]
        obstacles = [
            IntervalObstacle(start=30, end=40),
            IntervalObstacle(start=60, end=70),
        ]
    elif dimension == 2:
        bounds = [(-6, 6), (-2.5, 2.5)]
        final_location = [0, 0]
        obstacles = [
            CircleObstacle(center=[0, 1.5], radius=0.25),
            CircleObstacle(center=[1.0, -0.5], radius=0.5),
            CircleObstacle(center=[-2.0, -1.5], radius=0.5),
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
        collision_check_resolution=0.05,
        max_iter=1000,
        exact_step=exact_step,
        bounded_step=bounded_step,
        use_branch_pruning=use_branch_pruning,
        use_leaf_pruning=use_leaf_pruning,
        algorithm=algorithm_input,  # Pass the selected algorithm
        prevent_edge_overlap=prevent_edge_overlap
    )

    # Build RRT
    rrt.build_rrt()

    # Prune the tree
    rrt.post_process_tree(min_edge_separation=0.5, min_leaf_separation=0.5)

    # Visualize the RRT
    rrt.visualize(show_sampled_points=True)
    print(f"Algorithm Choice: {algorithm_input}, Dimension Choice: {dimension}, Exact Step: {exact_step}, Bounded Step: {bounded_step}, Branch Pruning: {use_branch_pruning}, Leaf Pruning: {use_leaf_pruning}")

#%%
if __name__ == "__main__":
    #%%
    main()
#%%