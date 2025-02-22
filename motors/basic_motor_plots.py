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
    pole_pairs = np.array([config["Pole Pairs (count)"] for config in configurations])
    turns_per_coil = np.array([config["Turns per Coil (count)"] for config in configurations])
    motor_thickness = np.array([config["T_motor (mm)"] for config in configurations])


    # Define the plot configurations with all 10 plots
    fig, axes = plt.subplots(5, 2, figsize=(18, 30))  # Increase figure size

    scatters = []

    plots = [
        (axes[0, 0], diameters, torques, "Motor Diameter (mm)", "Torque (Nm)", "Torque vs Motor Diameter", 'b'),
        (axes[0, 1], torques, voltages, "Torque (Nm)", "Voltage (V)", "Voltage vs Torque", 'r'),
        (axes[1, 0], currents, efficiencies, "Current Draw (A)", "Efficiency (%)", "Efficiency vs Current Draw", 'g'),
        (axes[1, 1], rpm_targets, efficiencies, "RPM Target", "Efficiency (%)", "Efficiency vs RPM Target", 'm'),
        (axes[2, 0], pole_pairs, efficiencies, "Pole Pairs (count)", "Efficiency (%)", "Efficiency vs Pole Pairs", 'c'),
        (axes[2, 1], turns_per_coil, torques, "Turns per Coil (count)", "Torque (Nm)", "Turns per Coil vs Torque", 'y'),
        (axes[3, 0], voltages, efficiencies, "Voltage (V)", "Efficiency (%)", "Efficiency vs Voltage", 'orange'),
        (axes[3, 1], torques, efficiencies, "Torque (Nm)", "Efficiency (%)", "Efficiency vs Torque", 'purple'),
        (axes[4, 0], currents, voltages, "Current Draw (A)", "Voltage (V)", "Voltage vs Current Draw", 'brown'),
        (axes[4, 1], motor_thickness, efficiencies, "Motor Thickness (mm)", "Efficiency (%)",
         "Motor Thickness vs Efficiency", 'lime'),
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
            for _, annotation, _, _, _, _ in scatters:
                annotation.set_visible(False)
            event.canvas.draw_idle()
            return

        hovered = False  # Track if any point is hovered

        for scatter, annotation, config_list, tree, x_data, y_data in scatters:
            if event.inaxes != scatter.axes:
                continue

            dist, index = tree.query([event.xdata, event.ydata], k=1)
            if dist < 10:  # Show annotation only if close enough
                config = config_list[index]
                annotation_text = "\n".join([f"{key}: {value}" for key, value in config.items()])
                annotation.set_text(annotation_text)
                annotation.xy = (x_data[index], y_data[index])
                annotation.set_visible(True)
                hovered = True
            else:
                annotation.set_visible(False)

        if not hovered:
            for _, annotation, _, _, _, _ in scatters:
                annotation.set_visible(False)

        event.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.tight_layout(pad=1)  # Increases padding
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.5, wspace=0.1)
    plt.show()


if __name__ == "__main__":
    filename = "valid_motor_configurations.txt"  # Ensure the correct filename is used
    configurations = load_configurations_from_file(filename)
    plot_motor_configurations(configurations)
