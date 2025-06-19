import math

import joblib
import numpy as np
from matplotlib import pyplot as plt, patches

# Load the scaler from the file
loaded_scaler = joblib.load(f'MODELS/scaler_angle_nn_optuna_ver5.pkl')

def return_next_point(p1, p2, theta, width, length, d, method, loaded_model):
    x_prev, y_prev = p2
    x_before_prev, y_before_prev = p1
    pre_x = x_prev / width
    pre_y = y_prev / length
    angle_of_last_part = math.atan2(y_prev - y_before_prev, x_prev - x_before_prev)
    data_point = [90 - abs(math.degrees(angle_of_last_part)), theta, pre_x, pre_y]
    data_array = np.array(data_point, dtype=np.float32).reshape(1, -1)
    if method == 'DNN':
        new_angle = loaded_model.predict(loaded_scaler.transform(data_array))[0][0]
    elif method == 'XGBoost':
        new_angle = loaded_model.predict(data_array)[0]
    else:  # tabnet
        new_angle = loaded_model.predict(data_array)[0][0]
    # Calculate the new point coordinates
    total_angle = -1 * math.radians(new_angle) + (math.pi / 2 - abs(angle_of_last_part))
    x_new = x_prev + d * math.sin(abs(total_angle))
    y_new = y_prev - d * math.cos(abs(total_angle))

    return (x_new, y_new), total_angle


def generate_points(p1, p2, theta, width, length, d, method):
    points = [p1, p2]
    # Load the model from the file
    if method == "DNN":
        version = 5
        loaded_model = joblib.load(f'MODELS/model_angle_nn_optuna_ver{version}.pkl')
    elif method == "XGBoost":
        version = 8
        loaded_model = joblib.load(f'MODELS/model_angle_XGBoost_ver{version}.pkl')
    else:
        version = 6
        loaded_model = joblib.load(f'MODELS/model_angle_tabnet_ver{version}.pkl')
    while points[-1][0] < width:
        new_point = return_next_point(points[-2], points[-1], theta, width, length, d, method, loaded_model)
        points.append(new_point[0])

    return points


def draw_crack_path(length, width, vertical_distance, horizontal_distance, diameter, precrack, theta, increment, method,
                    fig_needed):
    total_width = width
    total_height = length
    x_hole1 = (width - horizontal_distance) / 2
    y_hole1 = (length - vertical_distance) / 2
    radius = diameter / 2

    # Chart settings
    fig, ax = plt.subplots(dpi=120)
    ax.set_aspect('equal')
    ax.set_xlim(-5, total_width + 5)
    ax.set_ylim(-5, total_height + 5)

    # Add borders of the specimen
    ax.add_patch(patches.Rectangle((0, 0), total_width, total_height, edgecolor='black', facecolor='lightgrey', lw=2))

    # Add holes
    hole_positions = []

    for offset in [-1, 1]:  # -1 for lower holes, 1 for upper holes
        y = total_height / 2 + offset * vertical_distance / 2
        for i in range(3):  # Three horizontal positions
            x = x_hole1 + i * horizontal_distance / 2
            hole_positions.append((x, y))

    for x, y in hole_positions:
        ax.add_patch(patches.Circle((x, y), radius, edgecolor='blue', facecolor='white', lw=2))

    kolejne_punkty = generate_points((0, 0), (precrack, 0), theta, total_width, total_height, increment, method)

    ax.plot([total_width - kolejne_punkty[0][0], total_width - kolejne_punkty[1][0]],
            [total_height / 2 + kolejne_punkty[0][1], total_height / 2 + kolejne_punkty[1][1]],
            lw=2, c='k', solid_capstyle='round')

    x_coords = []
    y_coords = []
    for i in range(1, len(kolejne_punkty) - 1):
        if kolejne_punkty[i][0] > 40:
            break
        # Collect the coordinates
        x_coords.extend([
            total_width - kolejne_punkty[i][0],
            total_width - kolejne_punkty[i + 1][0]
        ])
        y_coords.extend([
            total_height / 2 + kolejne_punkty[i][1],
            total_height / 2 + kolejne_punkty[i + 1][1]
        ])

    # Plot all lines as a single plot
    ax.plot(x_coords, y_coords, lw=1, c='g', label='θ = 45°', marker='o', ms=0.5)

    # Add axis and captions
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'CTS specimen - \ncrack path predictions based on {method}', fontsize=16)

    # Customize grid without changing the ticks
    major_grid_lines = np.arange(length / 2, 20, -5)  # Major grid every 5 mm
    minor_grid_lines = np.arange(length / 2, 20, -1)  # Minor grid every 1 mm
    major_grid_line_distance = -5
    minor_grid_line_distance = -1

    # w - a
    w_minus_a = width - precrack

    ax.hlines(major_grid_lines, xmin=0, xmax=w_minus_a,
              colors='black', linestyles=':', linewidth=0.8, label='_nolegend_')  # Major grid
    ax.hlines(minor_grid_lines, xmin=0, xmax=w_minus_a,
              colors='gray', linestyles=':', linewidth=0.4, label='_nolegend_')  # Minor grid

    major_grid_lines = np.arange(w_minus_a, 0, major_grid_line_distance)  # Major grid every 5 mm
    minor_grid_lines = np.arange(w_minus_a, 0, minor_grid_line_distance)  # Minor grid every 1 mm

    ax.vlines(major_grid_lines, ymin=20, ymax=length / 2,
              colors='black', linestyles=':', linewidth=0.8, label='_nolegend_')  # Major grid
    ax.vlines(minor_grid_lines, ymin=20, ymax=length / 2,
              colors='gray', linestyles=':', linewidth=0.4, label='_nolegend_')  # Minor grid

    # Display chart
    # plt.savefig('CTS_1', dpi=300)
    if fig_needed:
        return fig
    else:
        plt.show()

