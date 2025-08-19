import sys
import os
import glob
import io
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QSlider,
    QProgressBar, QTextEdit, QGroupBox, QFileDialog
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap

from msla_antialiasing.core.worker import ProcessingWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mSLA Anti-Aliasing Processor")
        self.setGeometry(100, 100, 800, 600)

        # --- Data Storage ---
        self.slice_files = []
        self.slice_dims = None # Will store (num_slices, height, width)
        self.output_folder = None
        self.lut_data = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top layout for controls
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Left column for settings
        left_column = QVBoxLayout()
        top_layout.addLayout(left_column)

        # Right column for settings
        right_column = QVBoxLayout()
        top_layout.addLayout(right_column)

        # --- File I/O Group ---
        file_io_group = QGroupBox("File I/O")
        file_io_layout = QVBoxLayout()
        file_io_group.setLayout(file_io_layout)
        left_column.addWidget(file_io_group)

        self.btn_load_slices = QPushButton("Load Slices...")
        self.btn_load_slices.clicked.connect(self.load_slices)
        self.btn_set_output = QPushButton("Set Output Folder...")
        self.btn_set_output.clicked.connect(self.set_output_folder)
        self.lbl_slice_info = QLabel("Loaded: 0 slices")
        file_io_layout.addWidget(self.btn_load_slices)
        file_io_layout.addWidget(self.btn_set_output)
        file_io_layout.addWidget(self.lbl_slice_info)

        # --- Voxel Dimensions Group ---
        voxel_group = QGroupBox("Voxel Dimensions (µm)")
        voxel_layout = QHBoxLayout()
        voxel_group.setLayout(voxel_layout)
        left_column.addWidget(voxel_group)

        self.txt_voxel_x = QLineEdit("100")
        self.txt_voxel_y = QLineEdit("100")
        self.txt_voxel_z = QLineEdit("100")
        voxel_layout.addWidget(QLabel("X:"))
        voxel_layout.addWidget(self.txt_voxel_x)
        voxel_layout.addWidget(QLabel("Y:"))
        voxel_layout.addWidget(self.txt_voxel_y)
        voxel_layout.addWidget(QLabel("Z:"))
        voxel_layout.addWidget(self.txt_voxel_z)

        # --- Sliding Window Group ---
        window_group = QGroupBox("Sliding Window Controls")
        window_layout = QVBoxLayout()
        window_group.setLayout(window_layout)
        left_column.addWidget(window_group)

        # Window Size
        self.slider_window_size = QSlider(Qt.Horizontal)
        self.slider_window_size.setRange(10, 200)
        self.slider_window_size.setValue(64)
        self.lbl_window_size = QLabel(f"Window Size: {self.slider_window_size.value()}")
        self.slider_window_size.valueChanged.connect(lambda v: self.lbl_window_size.setText(f"Window Size: {v}"))
        window_layout.addWidget(self.lbl_window_size)
        window_layout.addWidget(self.slider_window_size)

        # Solved Layers
        self.slider_solved_layers = QSlider(Qt.Horizontal)
        self.slider_solved_layers.setRange(1, 100)
        self.slider_solved_layers.setValue(32)
        self.lbl_solved_layers = QLabel(f"Solved Layers: {self.slider_solved_layers.value()}")
        self.slider_solved_layers.valueChanged.connect(lambda v: self.lbl_solved_layers.setText(f"Solved Layers: {v}"))
        window_layout.addWidget(self.lbl_solved_layers)
        window_layout.addWidget(self.slider_solved_layers)

        # Overlap Layers
        self.slider_overlap = QSlider(Qt.Horizontal)
        self.slider_overlap.setRange(5, 100)
        self.slider_overlap.setValue(16)
        self.lbl_overlap = QLabel(f"Overlap Layers: {self.slider_overlap.value()}")
        self.slider_overlap.valueChanged.connect(lambda v: self.lbl_overlap.setText(f"Overlap Layers: {v}"))
        window_layout.addWidget(self.lbl_overlap)
        window_layout.addWidget(self.slider_overlap)

        # --- Solver & Performance Group ---
        solver_group = QGroupBox("Solver & Performance")
        solver_layout = QVBoxLayout()
        solver_group.setLayout(solver_layout)
        right_column.addWidget(solver_group)

        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU", "GPU"])
        self.combo_precision = QComboBox()
        self.combo_precision.addItems(["32-bit Float", "16-bit Float"])
        self.check_matrix_free = QCheckBox("Use Matrix-Free Method")

        solver_layout.addWidget(QLabel("Backend:"))
        solver_layout.addWidget(self.combo_backend)
        solver_layout.addWidget(QLabel("Precision:"))
        solver_layout.addWidget(self.combo_precision)
        solver_layout.addWidget(self.check_matrix_free)

        # --- Material Stage (LUT) Group ---
        lut_group = QGroupBox("Material Stage (LUT)")
        lut_layout = QVBoxLayout()
        lut_group.setLayout(lut_layout)
        right_column.addWidget(lut_group)

        self.btn_load_lut = QPushButton("Load LUT...")
        self.btn_load_lut.clicked.connect(self.load_lut)
        # Placeholder for the LUT plot
        self.lut_plot_placeholder = QLabel("Load a LUT file to see the plot")
        self.lut_plot_placeholder.setMinimumSize(200, 100)
        self.lut_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.lut_plot_placeholder.setStyleSheet("border: 1px solid grey;")
        lut_layout.addWidget(self.btn_load_lut)
        lut_layout.addWidget(self.lut_plot_placeholder)

        right_column.addStretch()


        # --- Execution Group ---
        execution_group = QGroupBox("Execution")
        execution_layout = QVBoxLayout()
        execution_group.setLayout(execution_layout)
        main_layout.addWidget(execution_group)

        self.btn_start = QPushButton("START PROCESSING")
        self.btn_start.clicked.connect(self.start_processing)
        self.progress_bar = QProgressBar()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        execution_layout.addWidget(self.btn_start)
        execution_layout.addWidget(self.progress_bar)
        execution_layout.addWidget(QLabel("Log:"))
        execution_layout.addWidget(self.log_area)

        self.log_message("Application started.")

    def get_window_params(self):
        return {
            "window_size": self.slider_window_size.value(),
            "solved_layers": self.slider_solved_layers.value(),
            "overlap": self.slider_overlap.value()
        }

    def get_solver_params(self):
        return {
            "backend": self.combo_backend.currentText(),
            "precision": self.combo_precision.currentText(),
            "matrix_free": self.check_matrix_free.isChecked()
        }

    def get_voxel_dims(self):
        try:
            vx = float(self.txt_voxel_x.text())
            vy = float(self.txt_voxel_y.text())
            vz = float(self.txt_voxel_z.text())
            return (vx, vy, vz)
        except ValueError:
            self.log_message("Error: Invalid voxel dimensions. Please enter numbers.")
            return None

    def start_processing(self):
        if not self.slice_files:
            self.log_message("Error: No slice data loaded.")
            return
        if self.output_folder is None:
            self.log_message("Error: No output folder selected.")
            return

        voxel_dims = self.get_voxel_dims()
        if voxel_dims is None: return

        # --- Prepare for threading ---
        self.btn_start.setEnabled(False)
        self.log_message("Preparing processing thread...")
        self.progress_bar.setRange(0, self.slice_dims[0])
        self.progress_bar.setValue(0)

        params = {
            'slice_files': self.slice_files,
            'slice_dims': self.slice_dims,
            'output_folder': self.output_folder,
            'voxel_dims': voxel_dims,
            'window_params': self.get_window_params(),
            'solver_params': self.get_solver_params(),
            'lut_data': self.lut_data,
        }

        # Create worker and thread
        self.thread = QThread()
        self.worker = ProcessingWorker(params)
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.log_message.connect(self.log_message)
        self.worker.progress_updated.connect(lambda v: self.progress_bar.setValue(v))

        self.thread.start()

    def on_processing_finished(self):
        self.log_message("Processing thread finished.")
        self.btn_start.setEnabled(True)
        # Clean up thread
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.worker = None

    def log_message(self, message):
        self.log_area.append(message)
        QApplication.processEvents() # Update the UI

    def load_lut(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load LUT File", "", "CSV Files (*.csv);;Text Files (*.txt)")
        if not filepath:
            return

        try:
            # Load LUT data, expecting a single column of 256 values
            lut = np.loadtxt(filepath, dtype=int)
            if len(lut) != 256:
                raise ValueError(f"LUT file must contain exactly 256 values, but found {len(lut)}.")
            if lut.min() < 0 or lut.max() > 255:
                raise ValueError("LUT values must be between 0 and 255.")

            self.lut_data = lut
            self.log_message(f"Successfully loaded LUT from {filepath}")
            self.update_lut_plot()

        except Exception as e:
            self.log_message(f"Error loading LUT file: {e}")
            self.lut_data = None

    def update_lut_plot(self):
        if self.lut_data is None:
            self.lut_plot_placeholder.setText("Load a LUT file to see the plot")
            return

        try:
            fig, ax = plt.subplots(figsize=(3, 1.5), dpi=100)
            ax.plot(self.lut_data)
            ax.set_title("Material Response Curve", fontsize=8)
            ax.set_xlabel("Input Grayscale", fontsize=6)
            ax.set_ylabel("Output Grayscale", fontsize=6)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.grid(True)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())

            self.lut_plot_placeholder.setPixmap(pixmap)
            plt.close(fig)

        except Exception as e:
            self.log_message(f"Error generating LUT plot: {e}")


    def set_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.log_message(f"Output folder set to: {self.output_folder}")

    def load_slices(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Slices")
        if not folder:
            return

        self.log_message(f"Scanning slices from: {folder}")
        supported_formats = ["*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]

        found_files = []
        for fmt in supported_formats:
            found_files.extend(glob.glob(os.path.join(folder, fmt)))

        if not found_files:
            self.log_message("Error: No supported image files found in the selected folder.")
            self.slice_files = []
            self.slice_dims = None
            return

        self.slice_files = sorted(found_files)
        num_slices = len(self.slice_files)
        self.log_message(f"Found {num_slices} image files. Reading dimensions...")

        try:
            # Read only the first image to get dimensions
            first_image = imageio.imread(self.slice_files[0])
            if first_image.ndim == 3: # Convert to grayscale if it's RGB
                first_image = np.dot(first_image[...,:3], [0.2989, 0.5870, 0.1140])

            h, w = first_image.shape
            self.slice_dims = (num_slices, h, w)

            self.lbl_slice_info.setText(f"Loaded: {num_slices} slices ({w}x{h})")
            self.log_message("Slice dimensions confirmed. Ready for processing.")

        except Exception as e:
            self.log_message(f"Error reading slice dimensions: {e}")
            self.slice_files = []
            self.slice_dims = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
