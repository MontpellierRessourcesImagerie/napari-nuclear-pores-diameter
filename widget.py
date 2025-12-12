from pathlib import Path
import numpy as np
import tifffile

import napari
from napari.utils import progress
from napari.utils.notifications import show_error, show_info, show_warning

from qtpy.QtWidgets import (QWidget, QVBoxLayout, 
    QGroupBox, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QComboBox, QLineEdit, 
    QFileDialog, QSpinBox, QDoubleSpinBox)
from qtpy.QtCore import Qt, QThread

from np_analyzer import NuclearPoresAnalyzer
from radial_nuclear_pores import radial_profiles, export_as_csv, as_pxls

class NuclearPoresAnalyzerWidget(QWidget):
    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        viewer = viewer or napari.current_viewer()
        if viewer is None:
            raise ValueError("No Napari viewer instance found.")
        self.viewer = viewer
        self.model = NuclearPoresAnalyzer()
        self.pts_layer_name = "Detected Nuclear Pores"
        self.working_dir = None
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.init_pattern_ui()
        self.init_detection_ui()
        self.init_analysis_ui()

    def init_pattern_ui(self):
        pattern_group = QGroupBox("Pattern Generation")
        pattern_layout = QVBoxLayout()

        r, a, p, s = self.model.get_pattern_properties()

        # Circle Parameters
        self.circle_params_layout = QHBoxLayout()
        circle_radius_label = QLabel("Radius:")
        self.circle_radius_spin = QDoubleSpinBox()
        self.circle_radius_spin.setRange(0.1, 1000.0)
        self.circle_radius_spin.setValue(r)
        self.circle_radius_spin.valueChanged.connect(lambda val: self.model.set_pattern_properties(
            val, self.anisotropy_spin.value(), self.padding_spin.value(), self.sigma_spin.value()))
        self.circle_params_layout.addWidget(circle_radius_label)
        self.circle_params_layout.addWidget(self.circle_radius_spin)
        pattern_layout.addLayout(self.circle_params_layout)

        # Anisotropy input (between 0 and 2 starting at 1)
        anisotropy_layout = QHBoxLayout()
        anisotropy_label = QLabel("Anisotropy:")
        self.anisotropy_spin = QDoubleSpinBox()
        self.anisotropy_spin.setRange(0.0, 2.0)
        self.anisotropy_spin.setSingleStep(0.1)
        self.anisotropy_spin.setValue(a)
        self.anisotropy_spin.valueChanged.connect(lambda val: self.model.set_pattern_properties(
            self.circle_radius_spin.value(), val, self.padding_spin.value(), self.sigma_spin.value()))
        anisotropy_layout.addWidget(anisotropy_label)
        anisotropy_layout.addWidget(self.anisotropy_spin)
        pattern_layout.addLayout(anisotropy_layout)

        # Padding input (int between 0 and 100 starting at 5)
        padding_layout = QHBoxLayout()
        padding_label = QLabel("Padding:")
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(p)
        self.padding_spin.valueChanged.connect(lambda val: self.model.set_pattern_properties(
            self.circle_radius_spin.value(), self.anisotropy_spin.value(), val, self.sigma_spin.value()))
        padding_layout.addWidget(padding_label)
        padding_layout.addWidget(self.padding_spin)
        pattern_layout.addLayout(padding_layout)

        # Sigma for the Gaussian filter
        sigma_layout = QHBoxLayout()
        sigma_label = QLabel("Sigma:")
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 100.0)
        self.sigma_spin.setValue(s)
        self.sigma_spin.valueChanged.connect(lambda val: self.model.set_pattern_properties(
            self.circle_radius_spin.value(), self.anisotropy_spin.value(), self.padding_spin.value(), val))
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.sigma_spin)
        pattern_layout.addLayout(sigma_layout)

        # Buttons on same line: Make pattern and Load pattern
        buttons_layout = QHBoxLayout()
        self.make_pattern_button = QPushButton("Make pattern")
        self.make_pattern_button.clicked.connect(self.generate_pattern)

        self.custom_pattern_button = QPushButton("Load Pattern")
        self.custom_pattern_button.clicked.connect(self.load_custom_pattern)

        buttons_layout.addWidget(self.custom_pattern_button)
        buttons_layout.addWidget(self.make_pattern_button)
        pattern_layout.addLayout(buttons_layout)

        pattern_group.setLayout(pattern_layout)
        self.main_layout.addWidget(pattern_group)

    def init_detection_ui(self):
        detection_group = QGroupBox("Nuclear Pores Detection")
        detection_layout = QVBoxLayout()

        # Method Selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        for method in self.model.get_methods().keys():
            self.method_combo.addItem(method)
        self.method_combo.currentIndexChanged.connect(lambda: self.model.set_method_name(
            self.method_combo.currentText()))
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        detection_layout.addLayout(method_layout)

        # Prominence input
        prominence_layout = QHBoxLayout()
        prominence_label = QLabel("Prominence:")
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.1, 100.0)
        self.prominence_spin.setValue(self.model.get_prominence())
        self.prominence_spin.valueChanged.connect(lambda val: self.model.set_prominence(val))
        prominence_layout.addWidget(prominence_label)
        prominence_layout.addWidget(self.prominence_spin)
        detection_layout.addLayout(prominence_layout)

        # Analyze Button
        self.analyze_button = QPushButton("Detect Nuclear Pores")
        self.analyze_button.clicked.connect(self.run_analysis)
        detection_layout.addWidget(self.analyze_button)

        detection_group.setLayout(detection_layout)
        self.main_layout.addWidget(detection_group)

    def init_analysis_ui(self):
        analysis_group = QGroupBox("Analyze")
        analysis_layout = QVBoxLayout()

        # Pixel size label + input
        pixel_size_layout = QHBoxLayout()
        pixel_size_label = QLabel("Pixel Size (µm):")
        self.pixel_size_input = QLineEdit("0.018")
        pixel_size_layout.addWidget(pixel_size_label)
        pixel_size_layout.addWidget(self.pixel_size_input)
        analysis_layout.addLayout(pixel_size_layout)

        # Initial diameter label + input
        diameter_layout = QHBoxLayout()
        diameter_label = QLabel("Initial Diameter (µm):")
        self.diameter_input = QLineEdit("0.36")
        diameter_layout.addWidget(diameter_label)
        diameter_layout.addWidget(self.diameter_input)
        analysis_layout.addLayout(diameter_layout)

        # Number of steps label + int input:
        steps_layout = QHBoxLayout()
        steps_label = QLabel("Number of steps:")
        self.steps_input = QSpinBox()
        self.steps_input.setRange(12, 64)
        self.steps_input.setValue(36)
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_input)
        analysis_layout.addLayout(steps_layout)

        # Choose method combo box:
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        self.analysis_method_combo = QComboBox()
        self.analysis_method_combo.addItem("peaks")
        self.analysis_method_combo.addItem("gaussian")
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.analysis_method_combo)
        analysis_layout.addLayout(method_layout)

        # Export plots checkbox:
        self.export_plots_checkbox = QCheckBox("Export Plots")
        analysis_layout.addWidget(self.export_plots_checkbox)

        # Working directory button:
        self.working_dir_button = QPushButton("Select Working Directory")
        self.working_dir_button.clicked.connect(self.select_working_directory)
        analysis_layout.addWidget(self.working_dir_button)

        # Run analysis button:
        self.run_analysis_button = QPushButton("Run Radial Analysis")
        self.run_analysis_button.clicked.connect(self.run_radial_analysis)
        analysis_layout.addWidget(self.run_analysis_button)

        analysis_group.setLayout(analysis_layout)
        self.main_layout.addWidget(analysis_group)

    def select_working_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Working Directory", str(Path.home()))
        if dir_path:
            self.working_dir = Path(dir_path)
            print(f"Working directory set to: {dir_path}")

    def generate_pattern(self):
        self.model.set_pattern_properties(
            self.circle_radius_spin.value(),
            self.anisotropy_spin.value(),
            self.padding_spin.value(),
            self.sigma_spin.value()
        )
        self.model.generate_pattern()
        pattern = self.model.get_pattern()
        self.viewer.add_image(pattern, name="Generated Pattern")

    def load_custom_pattern(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Pattern", "", "Image Files (*.tif *.tiff *.png *.jpg)")
        if file_path:
            pattern = tifffile.imread(file_path)
            if pattern.ndim == 2:
                self.model.set_pattern(pattern.astype(np.float32))
                self.viewer.add_image(pattern, name="Custom Pattern")
            else:
                print("Error: Pattern image must be 2D.")
    
    def run_analysis(self):
        if len(self.viewer.layers) == 0:
            print("No image loaded in viewer.")
            return
        image_layer = self.viewer.layers.selection.active
        if image_layer is None or image_layer.ndim != 2:
            print("Please select a 2D image layer.")
            return

        self.model.set_image_input(image_layer.data)

        # Run analysis in a separate thread to avoid blocking the UI
        self.analysis_thread = QThread()
        self.analysis_thread.run = self.analysis_procedure
        self.analysis_thread.finished.connect(self.show_points)
        self.analysis_thread.start()

    def analysis_procedure(self):
        self.model.set_prominence(self.prominence_spin.value())
        self.model.set_method_name(self.method_combo.currentText())
        try:
            self.model.process_points()
        except Exception as e:
            print(f"Error during analysis: {e}")

    def show_points(self):
        points = self.model.get_points()
        if len(points) > 0:
            if self.pts_layer_name in self.viewer.layers:
                self.viewer.layers[self.pts_layer_name].data = points
            else:
                self.viewer.add_points(points, name=self.pts_layer_name, size=5, face_color='red')
        else:
            print("No points detected.")
    
    def run_radial_analysis(self):
        if self.pts_layer_name not in self.viewer.layers:
            show_error("No detected points layer found.")
            return
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, 'colormap'):
            show_error("Please select the nuclear pores image layer.")
            return
        image_data = layer.data
        spots_layer = self.viewer.layers[self.pts_layer_name]
        spots_positions = spots_layer.data
        n_steps = int(self.steps_input.value())
        pxl_size = float(self.pixel_size_input.text())
        init_diameter = float(self.diameter_input.text())
        working_dir = self.working_dir
        if working_dir is None:
            show_error("Please select a working directory.")
            return
        diameter_px = as_pxls(init_diameter, pxl_size)
        plot_path = None if not self.export_plots_checkbox.isChecked() else working_dir
        mode = self.analysis_method_combo.currentText()
        results = radial_profiles(
            image_data, 
            spots_positions, 
            diameter_px, 
            pxl_size, 
            n_steps,
            mode=mode,
            working_dir=plot_path
        )
        point_colors = ['green' if r is not None else 'red' for r in results]
        spots_layer.face_color = point_colors
        export_as_csv(results, working_dir)
        show_info("Radial analysis completed.")

def launch_dev_procedure():
    viewer = napari.Viewer()
    widget = NuclearPoresAnalyzerWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)
    napari.run()


if __name__ == "__main__":
    launch_dev_procedure()
