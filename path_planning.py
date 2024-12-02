# import csv
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
# import numpy as np

# # Initialize lists for the points
# left_points = []
# right_points = []

# # Read the CSV file
# with open('BrandsHatchLayout.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         if row:  # Check if row is not empty
#             # Skip any empty values or rows with missing data
#             try:
#                 x = float(row[0].strip())  # Convert to float and strip spaces
#                 y = float(row[1].strip())  # Convert to float and strip spaces
#                 side = row[2].strip()  # Get side value
#             except ValueError:
#                 print(f"Skipping row due to conversion error: {row}")
#                 continue  # Skip rows that can't be processed

#             # Add points based on the side
#             if side == 'left':
#                 left_points.append((x, y))
#             elif side == 'right':
#                 right_points.append((x, y))

# # Calculate the midpoints
# def calculate_midpoint(left_cone, right_cone):
#     """Calculate the midpoint between a left cone and a right cone."""
#     mid_x = (left_cone[0] + right_cone[0]) / 2
#     mid_y = (left_cone[1] + right_cone[1]) / 2
#     return (mid_x, mid_y)

# # Ensure both left and right points lists are the same length
# if len(left_points) != len(right_points):
#     print("Warning: The number of left and right points does not match. Adjusting to the minimum length.")
#     min_length = min(len(left_points), len(right_points))
#     left_points = left_points[:min_length]
#     right_points = right_points[:min_length]

# # Calculate the midpoints for the path
# midpoints = [calculate_midpoint(left, right) for left, right in zip(left_points, right_points)]

# # Separate x and y coordinates for plotting
# mid_x, mid_y = zip(*midpoints) if midpoints else ([], [])

# # Perform cubic spline interpolation
# cs_x = CubicSpline(range(len(mid_x)), mid_x)
# cs_y = CubicSpline(range(len(mid_y)), mid_y)

# # Generate a finer range of points for smoother path
# x_fine = np.linspace(0, len(mid_x) - 1, 500)  # More points for smoothness
# y_fine_x = cs_x(x_fine)
# y_fine_y = cs_y(x_fine)

# # Plot the original and smooth paths
# # plt.scatter(mid_x, mid_y, label='Midpoints')
# plt.plot(y_fine_x, y_fine_y, label='Smoothed Path', color='purple', linewidth=2)

# # Plot the cones
# left_x, left_y = zip(*left_points) if left_points else ([], [])
# right_x, right_y = zip(*right_points) if right_points else ([], [])
# plt.scatter(left_x, left_y, label='Left Side', color='red')
# plt.scatter(right_x, right_y, label='Right Side', color='blue')

# # Show the plot
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.title('Smoothed Path with Cubic Spline')
# plt.show()

# import csv
# import matplotlib.pyplot as plt
# import numpy as np

# # Initialize lists for the points
# left_points = []
# right_points = []

# # Read the CSV file
# with open('BrandsHatchLayout.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         if row:  # Check if row is not empty
#             # Skip any empty values or rows with missing data
#             try:
#                 x = float(row[0].strip())  # Convert to float and strip spaces
#                 y = float(row[1].strip())  # Convert to float and strip spaces
#                 side = row[2].strip()  # Get side value
#             except ValueError:
#                 print(f"Skipping row due to conversion error: {row}")
#                 continue  # Skip rows that can't be processed

#             # Add points based on the side
#             if side == 'left':
#                 left_points.append((x, y))
#             elif side == 'right':
#                 right_points.append((x, y))

# # Ensure both left and right points lists are the same length
# if len(left_points) != len(right_points):
#     print("Warning: The number of left and right points does not match. Adjusting to the minimum length.")
#     min_length = min(len(left_points), len(right_points))
#     left_points = left_points[:min_length]
#     right_points = right_points[:min_length]

# # Initialize the path with midpoints
# path = np.array([(0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1]))
#                  for left, right in zip(left_points, right_points)])

# # Calculate the curvature of the path
# def calculate_curvature(path):
#     """
#     Calculate the total curvature of a 2D path.
#     Curvature is calculated using finite differences.
#     """
#     x = path[:, 0]
#     y = path[:, 1]
    
#     # First derivatives
#     dx = np.gradient(x)
#     dy = np.gradient(y)
    
#     # Second derivatives
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
    
#     # Curvature formula
#     numerator = np.abs(dx * ddy - dy * ddx)
#     denominator = (dx**2 + dy**2)**1.5
    
#     # To avoid division by zero, add a small epsilon to the denominator
#     curvature = np.sum(numerator / (denominator + 1e-6))
    
#     return curvature

# # Perform optimization
# num_iterations = 10000
# best_path = path.copy()
# best_curvature = calculate_curvature(best_path)
# print(f"Initial Curvature = {best_curvature:.4f}")

# # Randomized Path Adjustment with Weighting
# for iteration in range(num_iterations):
#     # Create a new path with randomized midpoints
#     randomized_path = path.copy()
#     for i in range(len(left_points)):
#         left_x, left_y = left_points[i]
#         right_x, right_y = right_points[i]

#         # Use weighted randomness for more diversity
#         t = np.random.normal(0.45, 0.55)  # Mean at 0.5, std dev 0.1

#         # Compute the point on the line at t
#         new_x = (1 - t) * left_x + t * right_x
#         new_y = (1 - t) * left_y + t * right_y

#         # Update the randomized path
#         randomized_path[i, 0] = new_x
#         randomized_path[i, 1] = new_y

#     # Calculate curvature of the new path
#     curvature = calculate_curvature(randomized_path)
#     print(f"Iteration {iteration}: Curvature = {curvature:.4f}")
#     # Update the best path if curvature is improved
#     if curvature < best_curvature:
#         best_curvature = curvature
#         best_path = randomized_path.copy()
#         print(f"Iteration {iteration}: New Best Curvature = {best_curvature:.4f}")


# # Plot the results
# left_x, left_y = zip(*left_points)
# right_x, right_y = zip(*right_points)
# path_x, path_y = zip(*path)
# best_x, best_y = zip(*best_path)

# plt.figure(figsize=(10, 6))
# plt.scatter(left_x, left_y, label="Left Cones", color="red")
# plt.scatter(right_x, right_y, label="Right Cones", color="blue")
# plt.plot(path_x, path_y, label="Initial Path (Midpoints)", linestyle="--", color="gray")
# plt.plot(best_x, best_y, label="Optimized Path", color="purple", linewidth=2)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.title("Path Optimization Between Cones")
# plt.show()


# import csv
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
# import numpy as np

# # Initialize lists for the points
# left_points = []
# right_points = []

# # Read the CSV file
# with open('BrandsHatchLayout.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         if row:  # Check if row is not empty
#             # Skip any empty values or rows with missing data
#             try:
#                 x = float(row[0].strip())  # Convert to float and strip spaces
#                 y = float(row[1].strip())  # Convert to float and strip spaces
#                 side = row[2].strip()  # Get side value
#             except ValueError:
#                 print(f"Skipping row due to conversion error: {row}")
#                 continue  # Skip rows that can't be processed

#             # Add points based on the side
#             if side == 'left':
#                 left_points.append((x, y))
#             elif side == 'right':
#                 right_points.append((x, y))

# # Calculate the midpoints
# def calculate_midpoint(left_cone, right_cone):
#     """Calculate the midpoint between a left cone and a right cone."""
#     mid_x = (left_cone[0] + right_cone[0]) / 2
#     mid_y = (left_cone[1] + right_cone[1]) / 2
#     return (mid_x, mid_y)

# # Ensure both left and right points lists are the same length
# if len(left_points) != len(right_points):
#     print("Warning: The number of left and right points does not match. Adjusting to the minimum length.")
#     min_length = min(len(left_points), len(right_points))
#     left_points = left_points[:min_length]
#     right_points = right_points[:min_length]

# # Calculate the midpoints for the path
# midpoints = [calculate_midpoint(left, right) for left, right in zip(left_points, right_points)]

# # Separate x and y coordinates for the midpoints
# mid_x, mid_y = zip(*midpoints) if midpoints else ([], [])

# # Calculate arc length
# def calculate_arc_length(x, y):
#     """Calculate the cumulative arc length for a set of points."""
#     lengths = [0]  # Starting with a length of 0 at the first point
#     for i in range(1, len(x)):
#         dist = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
#         lengths.append(lengths[-1] + dist)
#     return lengths

# # Calculate the arc length for midpoints
# arc_length = calculate_arc_length(mid_x, mid_y)

# # Normalize arc length to [0, 1] range for reparameterization
# arc_length_normalized = np.array(arc_length) / arc_length[-1]

# # Perform cubic spline interpolation on the normalized arc length
# cs_x = CubicSpline(arc_length_normalized, mid_x)
# cs_y = CubicSpline(arc_length_normalized, mid_y)

# # Generate a finer range of points for smoother path
# x_fine = np.linspace(0, 1, 500)  # Normalize the range of arc lengths from 0 to 1
# y_fine_x = cs_x(x_fine)
# y_fine_y = cs_y(x_fine)

# # Function to calculate curvature
# def calculate_curvature(x, y):
#     """Calculate the curvature of the path."""
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
#     return np.sum(np.abs(curvature))  # Sum of absolute curvature

# # Optimization using random perturbation and curvature minimization
# # Optimization using random perturbation and curvature minimization
# def optimize_path(mid_x, mid_y, left_points, right_points, iterations=1000, step_size=0.1):
#     # Convert mid_x and mid_y to lists before copying
#     best_mid_x, best_mid_y = list(mid_x), list(mid_y)
#     best_curvature = calculate_curvature(best_mid_x, best_mid_y)

#     for _ in range(iterations):
#         # Create a new perturbation of the midpoints
#         new_mid_x = best_mid_x.copy()
#         new_mid_y = best_mid_y.copy()

#         # Randomly perturb each midpoint within the bounds of the left and right cone
#         for i in range(len(left_points)):
#             # Randomize x and y within the bounds of left and right cone
#             new_mid_x[i] = np.random.uniform(left_points[i][0], right_points[i][0])
#             new_mid_y[i] = np.random.uniform(left_points[i][1], right_points[i][1])

#         # Reparameterize and compute curvature
#         arc_length_new = calculate_arc_length(new_mid_x, new_mid_y)
#         arc_length_normalized_new = np.array(arc_length_new) / arc_length_new[-1]
        
#         cs_x_new = CubicSpline(arc_length_normalized_new, new_mid_x)
#         cs_y_new = CubicSpline(arc_length_normalized_new, new_mid_y)
        
#         x_fine_new = np.linspace(0, 1, 500)
#         y_fine_x_new = cs_x_new(x_fine_new)
#         y_fine_y_new = cs_y_new(x_fine_new)
        
#         # Calculate the curvature of the new path
#         new_curvature = calculate_curvature(y_fine_x_new, y_fine_y_new)
        
#         # If the new curvature is better (lower), update the best path
#         if new_curvature < best_curvature:
#             best_mid_x, best_mid_y = new_mid_x, new_mid_y
#             best_curvature = new_curvature

#     return best_mid_x, best_mid_y


# # Perform optimization
# optimized_mid_x, optimized_mid_y = optimize_path(mid_x, mid_y, left_points, right_points)

# # Reparameterize and generate the optimized smooth path
# arc_length_optimized = calculate_arc_length(optimized_mid_x, optimized_mid_y)
# arc_length_normalized_optimized = np.array(arc_length_optimized) / arc_length_optimized[-1]

# cs_x_optimized = CubicSpline(arc_length_normalized_optimized, optimized_mid_x)
# cs_y_optimized = CubicSpline(arc_length_normalized_optimized, optimized_mid_y)

# x_fine_optimized = np.linspace(0, 1, 500)
# y_fine_x_optimized = cs_x_optimized(x_fine_optimized)
# y_fine_y_optimized = cs_y_optimized(x_fine_optimized)

# # Plot the optimized path
# plt.plot(y_fine_x_optimized, y_fine_y_optimized, label='Optimized Path', color='green', linewidth=2)

# # Plot the cones (left and right points)
# left_x, left_y = zip(*left_points) if left_points else ([], [])
# right_x, right_y = zip(*right_points) if right_points else ([], [])
# plt.scatter(left_x, left_y, label='Left Side', color='red')
# plt.scatter(right_x, right_y, label='Right Side', color='blue')

# # Show the plot
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.title('Optimized Path with Arc Length Parameterization')
# plt.show()

import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from scipy.optimize import minimize

def read_cones(file_path):
    """Reads cone data from a CSV file."""
    left_points = []
    right_points = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                try:
                    x = float(row[0].strip())
                    y = float(row[1].strip())
                    side = row[2].strip()
                except ValueError:
                    print(f"Skipping row due to conversion error: {row}")
                    continue

                if side == 'left':
                    left_points.append((x, y))
                elif side == 'right':
                    right_points.append((x, y))

    return np.array(left_points), np.array(right_points)

def calculate_weighted_midpoints(left_points, right_points):
    """Calculate weighted midpoints between left and right cones."""
    tree_left = KDTree(left_points)
    tree_right = KDTree(right_points)

    midpoints = []
    for lp in left_points:
        dist, index = tree_right.query(lp)
        rp = right_points[index]
        mid_x = (lp[0] + rp[0]) / 2
        mid_y = (lp[1] + rp[1]) / 2
        midpoints.append((mid_x, mid_y))

    return np.array(midpoints)

def smooth_path(midpoints, window_length=7, polyorder=3):
    """Smooth the midpoints using Savitzky-Golay filter."""
    x, y = midpoints[:, 0], midpoints[:, 1]
    x_smooth = savgol_filter(x, window_length, polyorder)
    y_smooth = savgol_filter(y, window_length, polyorder)
    return np.column_stack((x_smooth, y_smooth))

def interpolate_path(smoothed_midpoints):
    """Perform spline interpolation on the smoothed midpoints."""
    tck, _ = splprep([smoothed_midpoints[:, 0], smoothed_midpoints[:, 1]], s=0.5)
    u_fine = np.linspace(0, 1, 500)
    x_fine, y_fine = splev(u_fine, tck)
    return x_fine, y_fine

def curvature_cost_with_dynamic_boundaries(path, left_points, right_points, boundary_margin=0.5):
    """Cost function to minimize curvature and maintain path within the bounds dynamically."""
    path = path.reshape(-1, 2)
    curvature_penalty = 0
    for i in range(1, len(path) - 1):
        p_prev, p_curr, p_next = path[i - 1], path[i], path[i + 1]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        curvature_penalty += angle ** 2

    boundary_penalty = 0
    for p in path:
        dist_left = np.min(np.linalg.norm(left_points - p, axis=1))
        dist_right = np.min(np.linalg.norm(right_points - p, axis=1))
        if dist_left < boundary_margin:
            boundary_penalty += (boundary_margin - dist_left) ** 2 * 1000
        if dist_right < boundary_margin:
            boundary_penalty += (boundary_margin - dist_right) ** 2 * 1000

    return curvature_penalty + boundary_penalty

def optimize_path_with_constraints(midpoints, left_points, right_points, boundary_margin=0.5):
    """Optimize the path with explicit boundary constraints."""
    initial_guess = midpoints.flatten()
    
    def constraint_func(path):
        path = path.reshape(-1, 2)
        left_dists = np.min(np.linalg.norm(left_points[:, None] - path, axis=2), axis=0)
        right_dists = np.min(np.linalg.norm(right_points[:, None] - path, axis=2), axis=0)
        return np.minimum(left_dists - boundary_margin, right_dists - boundary_margin)
    
    constraints = {
        'type': 'ineq',  # Boundary distances must be >= margin
        'fun': constraint_func
    }

    result = minimize(
        curvature_cost_with_dynamic_boundaries,
        initial_guess,
        args=(left_points, right_points, boundary_margin),
        constraints=constraints,
        method='SLSQP',
        options={'disp': True}
    )
    optimized_path = result.x.reshape(-1, 2)
    return optimized_path

def plot_path(left_points, right_points, midpoints, smoothed_midpoints, x_fine, y_fine, optimized_path):
    """Plot the cones, midpoints, and interpolated path."""
    plt.figure(figsize=(10, 6))
    
    # Plot cones
    plt.scatter(left_points[:, 0], left_points[:, 1], label='Left Cones', color='red')
    plt.scatter(right_points[:, 0], right_points[:, 1], label='Right Cones', color='blue')

    # Plot midpoints
    plt.scatter(midpoints[:, 0], midpoints[:, 1], label='Midpoints', color='green', alpha=0.6)

    # Plot smoothed midpoints
    plt.plot(smoothed_midpoints[:, 0], smoothed_midpoints[:, 1], label='Smoothed Midpoints', color='orange', linestyle='--')

    # Plot interpolated path
    plt.plot(x_fine, y_fine, label='Interpolated Path', color='purple', linewidth=2)

    # Plot optimized path
    plt.plot(optimized_path[:, 0], optimized_path[:, 1], label='Optimized Path', color='black', linewidth=2, linestyle='-.')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Optimized Path with Weighted Midpoints and Spline Interpolation')
    plt.show()

# Main flow
file_path = 'BrandsHatchLayout.csv'  # Update with your file path

# Read cone positions
left_points, right_points = read_cones(file_path)

# Calculate weighted midpoints
midpoints = calculate_weighted_midpoints(left_points, right_points)

# Smooth the midpoints
smoothed_midpoints = smooth_path(midpoints)

# Interpolate the path
x_fine, y_fine = interpolate_path(smoothed_midpoints)

# Optimize the path
optimized_path = optimize_path_with_constraints(smoothed_midpoints, left_points, right_points)

# Plot everything
plot_path(left_points, right_points, midpoints, smoothed_midpoints, x_fine, y_fine, optimized_path)

