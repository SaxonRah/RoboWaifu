import matplotlib.pyplot as plt
import numpy as np
import ast

from scipy.spatial import cKDTree


def load_configurations_from_file(filename):
    """Loads motor configurations from a file."""
    configurations = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line (summary)
            try:
                config = ast.literal_eval(line.strip())
                configurations.append(config)
            except (SyntaxError, ValueError):
                continue
    return configurations


def plot_motor_configurations(configurations):
    """Plots various aspects of the motor configurations in a single figure with optimized performance."""
    if not configurations:
        print("No valid configurations found.")
        return

    # Extract data
    diameters = np.array([config["D_motor (mm)"] for config in configurations])
    torques = np.array([config["Torque (Nm)"] for config in configurations])
    voltages = np.array([config["Voltage (V)"] for config in configurations])
    efficiencies = np.array([config["Efficiency (%)"] for config in configurations])
    currents = np.array([config["Current Draw (A)"] for config in configurations])
    rpm_targets = np.array([config["RPM_target"] for config in configurations])

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scatters = []

    plots = [
        (axes[0, 0], diameters, torques, "Motor Diameter (mm)", "Torque (Nm)", "Torque vs Motor Diameter", 'b'),
        (axes[0, 1], torques, voltages, "Torque (Nm)", "Voltage (V)", "Voltage vs Torque", 'r'),
        (axes[1, 0], currents, efficiencies, "Current Draw (A)", "Efficiency (%)", "Efficiency vs Current Draw", 'g'),
        (axes[1, 1], rpm_targets, efficiencies, "RPM Target", "Efficiency (%)", "Efficiency vs RPM Target", 'm')
    ]

    for ax, x, y, xlabel, ylabel, title, color in plots:
        scatter = ax.scatter(x, y, c=color, s=2, label=ylabel, alpha=0.5)  # Reduce marker size and transparency
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid()
        annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", visible=False,
                                 bbox=dict(boxstyle="round", fc="w"))
        kd_tree = cKDTree(np.column_stack([x, y]))  # Build a KD-tree for each plot
        scatters.append((scatter, annotation, configurations, kd_tree, x, y))

    def on_hover(event):
        """Efficiently detects nearest point using KD-Tree and displays parameters."""
        if event.inaxes is None:
            return

        for scatter, annotation, config_list, tree, x_data, y_data in scatters:
            if event.inaxes != scatter.axes:
                continue
            dist, index = tree.query([event.xdata, event.ydata], k=1)
            if dist < 10:  # Only show annotation for nearby points
                config = config_list[index]
                annotation_text = "\n".join([f"{key}: {value}" for key, value in config.items()])
                annotation.set_text(annotation_text)
                annotation.xy = (x_data[index], y_data[index])
                annotation.set_visible(True)
                event.canvas.draw_idle()
                return

        for _, annotation, _, _, _, _ in scatters:
            annotation.set_visible(False)
        event.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "valid_motor_configurations.txt"  # Ensure the correct filename is used
    configurations = load_configurations_from_file(filename)
    plot_motor_configurations(configurations)
