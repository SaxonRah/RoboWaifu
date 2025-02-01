import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pathlib import Path
import json

from AxialMotorCalculator import AxialMotorCalculator
from MotorParameters import MotorParameters, create_default_parameters
from MotorPlot import MotorAnalysisPlotter
from GeneratePrintableMotor import generate_printable_parts
from MotorAnalyzer import MotorAnalyzer


class MotorDesignTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Motor Design Tool")

        # Set up main container
        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Parameters tab
        self.params_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.params_frame, text="Parameters")
        self.setup_parameters_tab()

        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.setup_analysis_tab()

        # 3D Model tab
        self.model_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.model_frame, text="3D Model")
        self.setup_model_tab()

        # Initialize parameters
        self.params = create_default_parameters()
        self.calculator = None
        self.current_configs = []

    def setup_parameters_tab(self):
        # Basic parameters
        basic_frame = ttk.LabelFrame(self.params_frame, text="Basic Parameters", padding="5")
        basic_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Target torque
        ttk.Label(basic_frame, text="Target Torque (Nm):").grid(row=0, column=0, sticky=tk.W)
        self.target_torque = tk.StringVar(value="0.1")
        ttk.Entry(basic_frame, textvariable=self.target_torque).grid(row=0, column=1)

        # Voltage
        ttk.Label(basic_frame, text="Voltage (V):").grid(row=1, column=0, sticky=tk.W)
        self.voltage = tk.StringVar(value="12.0")
        ttk.Entry(basic_frame, textvariable=self.voltage).grid(row=1, column=1)

        # Advanced parameters
        adv_frame = ttk.LabelFrame(self.params_frame, text="Advanced Parameters", padding="5")
        adv_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Wire diameter
        ttk.Label(adv_frame, text="Wire Diameter (mm):").grid(row=0, column=0, sticky=tk.W)
        self.wire_diameter = tk.StringVar(value="0.65")
        ttk.Entry(adv_frame, textvariable=self.wire_diameter).grid(row=0, column=1)

        # Search button
        ttk.Button(self.params_frame, text="Search Configurations",
                   command=self.search_configurations).grid(row=2, column=0, pady=10)

    def setup_analysis_tab(self):
        # Results list
        self.results_frame = ttk.LabelFrame(self.analysis_frame, text="Found Configurations")
        self.results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Graphs frame
        self.graphs_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Graphs")
        self.graphs_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    def setup_model_tab(self):
        # Model parameters
        model_params_frame = ttk.LabelFrame(self.model_frame, text="Model Parameters", padding="5")
        model_params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Output directory
        ttk.Label(model_params_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W)
        self.output_dir = tk.StringVar(value="motor_output")
        ttk.Entry(model_params_frame, textvariable=self.output_dir).grid(row=0, column=1)

        # Generate button
        ttk.Button(self.model_frame, text="Generate 3D Model",
                   command=self.generate_model).grid(row=1, column=0, pady=10)

    def search_configurations(self):
        try:
            # Update parameters
            self.params.target_torque = float(self.target_torque.get())
            self.params.voltage = float(self.voltage.get())
            self.params.wire_diameter = float(self.wire_diameter.get())

            # Create calculator
            self.calculator = AxialMotorCalculator(self.params)

            # Find configurations
            self.current_configs = self.calculator.find_viable_configurations()

            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            # Display results
            for i, config in enumerate(self.current_configs):
                config_str = f"Config {i + 1}: {config.poles}P/{config.coils}C, " \
                             f"{config.turns_per_coil} turns, " \
                             f"{config.estimated_torque:.3f} Nm, " \
                             f"{config.efficiency:.1f}%"

                ttk.Label(self.results_frame, text=config_str).grid(
                    row=i, column=0, sticky=tk.W, padx=5, pady=2)

            # Generate analysis plots
            self.update_analysis_plots()

        except Exception as e:
            messagebox.showerror("Error", f"Configuration search failed: {str(e)}")

    def update_analysis_plots(self):
        # Clear previous plots
        for widget in self.graphs_frame.winfo_children():
            widget.destroy()

        if not self.current_configs:
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Torque vs Efficiency plot
        torques = [c.estimated_torque for c in self.current_configs]
        efficiencies = [c.efficiency for c in self.current_configs]
        ax1.scatter(torques, efficiencies)
        ax1.set_xlabel('Torque (Nm)')
        ax1.set_ylabel('Efficiency (%)')
        ax1.set_title('Torque vs Efficiency')

        # Poles/Coils distribution
        poles = [c.poles for c in self.current_configs]
        coils = [c.coils for c in self.current_configs]
        ax2.scatter(poles, coils)
        ax2.set_xlabel('Poles')
        ax2.set_ylabel('Coils')
        ax2.set_title('Poles vs Coils')

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graphs_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

    def generate_model(self):
        if not self.current_configs:
            messagebox.showerror("Error", "No configurations available. Run search first.")
            return

        try:
            # Use first valid configuration
            config = self.current_configs[0]

            # Create output directory
            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate parts
            print_settings = {
                'wall_thickness': 2.0,
                'tolerance': 0.2,
                'layer_height': 0.2,
                'segments': 100,
                'coil_orientation': 'axial',
                'cutaway': True
            }

            parts = generate_printable_parts(
                config,
                output_prefix="motor",
                output_location=str(output_dir),
                print_settings=print_settings
            )

            messagebox.showinfo("Success",
                                f"3D model files generated in {output_dir}\n" +
                                f"Generated files: {', '.join(parts.keys())}")

        except Exception as e:
            messagebox.showerror("Error", f"Model generation failed: {str(e)}")


def main():
    root = tk.Tk()
    app = MotorDesignTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()