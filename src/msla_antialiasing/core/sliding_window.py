import numpy as np

def sliding_window(data: np.ndarray, window_size: int, solved_layers: int, overlap: int):
    """
    Generator for a sliding window over a 3D data volume.

    Args:
        data (np.ndarray): The 3D numpy array (slices, height, width).
        window_size (int): The total number of slices in each window.
        solved_layers (int): The number of layers to consider "solved" from the center of the window.
        overlap (int): The number of layers to overlap between consecutive windows.

    Yields:
        tuple: A tuple containing (window_data, save_start_index, save_end_index, output_start_slice).
               - window_data (np.ndarray): The chunk of data for the current window.
               - save_start_index (int): The starting index within the window of the part to be saved.
               - save_end_index (int): The ending index within the window of the part to be saved.
               - output_start_slice (int): The starting slice index in the final output volume.
    """
    num_slices = data.shape[0]

    if solved_layers > window_size:
        raise ValueError("solved_layers cannot be greater than window_size.")

    # The step size is how many layers we advance in each iteration.
    # This is typically the number of solved layers.
    step_size = window_size - overlap

    current_slice = 0
    output_slice_idx = 0

    while current_slice < num_slices:
        # Define the window boundaries
        start = current_slice
        end = start + window_size

        # Clamp the window to the data boundaries
        if end > num_slices:
            end = num_slices
            start = max(0, end - window_size)

        window_data = data[start:end, :, :]
        actual_window_size = window_data.shape[0]

        # Determine the "solved" region to be saved from this window
        # For the first window
        if current_slice == 0:
            save_start = 0
            save_end = step_size
        # For the last window
        elif end == num_slices:
            save_start = overlap
            save_end = actual_window_size
        # For intermediate windows
        else:
            save_start = overlap
            save_end = overlap + step_size

        # Ensure save indices are within the actual window bounds
        save_start = min(save_start, actual_window_size)
        save_end = min(save_end, actual_window_size)

        if save_start < save_end:
            yield window_data, save_start, save_end, output_slice_idx

        if end == num_slices:
            break

        current_slice += step_size
        output_slice_idx += (save_end - save_start)
