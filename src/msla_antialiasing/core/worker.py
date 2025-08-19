import os
import numpy as np
import imageio.v2 as imageio
from PySide6.QtCore import QObject, Signal

from msla_antialiasing.core.sliding_window import sliding_window
from msla_antialiasing.core.solver import solve_poisson_cpu, solve_poisson_gpu

class ProcessingWorker(QObject):
    """
    A QObject worker to run the processing pipeline in a separate thread.
    """
    progress_updated = Signal(int)
    log_message = Signal(str)
    finished = Signal()

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """
        Starts the processing pipeline.
        """
        try:
            p = self.params
            slice_data = p['slice_data']
            output_folder = p['output_folder']
            slice_files = p['slice_files']
            voxel_dims = p['voxel_dims']
            window_params = p['window_params']
            solver_params = p['solver_params']
            lut_data = p['lut_data']

            self.log_message.emit("Processing thread started.")
            os.makedirs(output_folder, exist_ok=True)

            input_shape = slice_data.shape
            result_volume = np.zeros(input_shape, dtype=np.uint8)
            binary_input = (slice_data > 127).astype(np.uint8)
            total_slices = input_shape[0]

            self.progress_updated.emit(0)

            window_gen = sliding_window(binary_input, **window_params)

            for window_data, save_start, save_end, output_start_slice in window_gen:
                self.log_message.emit(f"Processing window starting at slice {output_start_slice}...")

                if solver_params['backend'] == 'CPU':
                    processed_window = solve_poisson_cpu(
                        window_data,
                        voxel_size=voxel_dims,
                        precision=solver_params['precision'],
                        matrix_free=solver_params['matrix_free']
                    )
                else: # GPU
                    processed_window = solve_poisson_gpu(
                        window_data,
                        voxel_size=voxel_dims,
                        precision=solver_params['precision'],
                        matrix_free=solver_params['matrix_free']
                    )

                solved_part = processed_window[save_start:save_end]
                num_layers_to_save = solved_part.shape[0]
                output_end_slice = output_start_slice + num_layers_to_save

                if output_end_slice > total_slices:
                    num_layers_to_save = total_slices - output_start_slice
                    solved_part = solved_part[:num_layers_to_save]
                    output_end_slice = total_slices

                if num_layers_to_save > 0:
                    result_volume[output_start_slice:output_end_slice] = solved_part

                self.progress_updated.emit(output_end_slice)

            self.log_message.emit("Processing complete. Applying LUT and saving slices...")

            if lut_data is not None:
                self.log_message.emit("Applying custom LUT.")
                lut_np = np.array(lut_data, dtype=np.uint8)
                final_volume = lut_np[result_volume]
            else:
                final_volume = result_volume

            for i in range(total_slices):
                original_filename = os.path.basename(slice_files[i])
                base, ext = os.path.splitext(original_filename)
                output_filename = f"{base}.png"
                output_path = os.path.join(output_folder, output_filename)
                imageio.imwrite(output_path, final_volume[i])

            self.progress_updated.emit(total_slices)
            self.log_message.emit("All slices saved successfully.")

        except Exception as e:
            self.log_message.emit(f"An error occurred in the processing thread: {e}")

        finally:
            self.finished.emit()
