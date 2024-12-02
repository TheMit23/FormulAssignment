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

import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists for the points
left_points = []
right_points = []

# Read the CSV file
with open('BrandsHatchLayout.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row:  # Check if row is not empty
            # Skip any empty values or rows with missing data
            try:
                x = float(row[0].strip())  # Convert to float and strip spaces
                y = float(row[1].strip())  # Convert to float and strip spaces
                side = row[2].strip()  # Get side value
            except ValueError:
                print(f"Skipping row due to conversion error: {row}")
                continue  # Skip rows that can't be processed

            # Add points based on the side
            if side == 'left':
                left_points.append((x, y))
            elif side == 'right':
                right_points.append((x, y))

# Ensure both left and right points lists are the same length
if len(left_points) != len(right_points):
    print("Warning: The number of left and right points does not match. Adjusting to the minimum length.")
    min_length = min(len(left_points), len(right_points))
    left_points = left_points[:min_length]
    right_points = right_points[:min_length]

# Initialize the path with midpoints
path = np.array([(0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1]))
                 for left, right in zip(left_points, right_points)])

# Calculate the curvature of the path
def calculate_curvature(path):
    """
    Calculate the total curvature of a 2D path.
    Curvature is calculated using finite differences.
    """
    x = path[:, 0]
    y = path[:, 1]
    
    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5
    
    # To avoid division by zero, add a small epsilon to the denominator
    curvature = np.sum(numerator / (denominator + 1e-6))
    
    return curvature

# Perform optimization
num_iterations = 10000
best_path = path.copy()
best_curvature = calculate_curvature(best_path)
print(f"Initial Curvature = {best_curvature:.4f}")

# Randomized Path Adjustment with Weighting
for iteration in range(num_iterations):
    # Create a new path with randomized midpoints
    randomized_path = path.copy()
    for i in range(len(left_points)):
        left_x, left_y = left_points[i]
        right_x, right_y = right_points[i]

        # Use weighted randomness for more diversity
        t = np.random.normal(0.45, 0.55)  # Mean at 0.5, std dev 0.1

        # Compute the point on the line at t
        new_x = (1 - t) * left_x + t * right_x
        new_y = (1 - t) * left_y + t * right_y

        # Update the randomized path
        randomized_path[i, 0] = new_x
        randomized_path[i, 1] = new_y

    # Calculate curvature of the new path
    curvature = calculate_curvature(randomized_path)
    print(f"Iteration {iteration}: Curvature = {curvature:.4f}")
    # Update the best path if curvature is improved
    if curvature < best_curvature:
        best_curvature = curvature
        best_path = randomized_path.copy()
        print(f"Iteration {iteration}: New Best Curvature = {best_curvature:.4f}")


# Plot the results
left_x, left_y = zip(*left_points)
right_x, right_y = zip(*right_points)
path_x, path_y = zip(*path)
best_x, best_y = zip(*best_path)

plt.figure(figsize=(10, 6))
plt.scatter(left_x, left_y, label="Left Cones", color="red")
plt.scatter(right_x, right_y, label="Right Cones", color="blue")
plt.plot(path_x, path_y, label="Initial Path (Midpoints)", linestyle="--", color="gray")
plt.plot(best_x, best_y, label="Optimized Path", color="purple", linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Path Optimization Between Cones")
plt.show()


