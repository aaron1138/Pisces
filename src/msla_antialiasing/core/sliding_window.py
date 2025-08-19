import numpy as np
import imageio.v2 as imageio

def _load_image_range(files, start, end, dims):
    """Loads a range of image files into a numpy array."""
    num_slices, h, w = dims
    file_list = files[start:end]

    # Pre-allocate numpy array for the window
    window_data = np.zeros((len(file_list), h, w), dtype=np.uint8)

    for i, f in enumerate(file_list):
        img = imageio.imread(f)
        if img.ndim == 3: # Convert to grayscale if it's RGB
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        window_data[i] = img

    return window_data

def sliding_window(slice_files: list, slice_dims: tuple, window_size: int, solved_layers: int, overlap: int):
    """
    Generator for a sliding window that loads data from files on the fly.

    Args:
        slice_files (list): A list of paths to the image files.
        slice_dims (tuple): A tuple of (num_slices, height, width).
        window_size (int): The total number of slices in each window.
        solved_layers (int): The number of layers to consider "solved" from the center of the window.
        overlap (int): The number of layers to overlap between consecutive windows.

    Yields:
        tuple: A tuple containing (window_data, save_start_index, save_end_index, output_start_slice).
    """
    num_slices, h, w = slice_dims

    if solved_layers > window_size:
        raise ValueError("solved_layers cannot be greater than window_size.")

    step_size = window_size - overlap

    current_slice = 0
    output_slice_idx = 0

    while current_slice < num_slices:
        start = current_slice
        end = start + window_size

        if end > num_slices:
            end = num_slices
            start = max(0, end - window_size)

        # Load only the required data for this window
        window_data = _load_image_range(slice_files, start, end, slice_dims)
        actual_window_size = window_data.shape[0]

        if current_slice == 0:
            save_start = 0
            save_end = step_size
        elif end == num_slices:
            save_start = overlap
            save_end = actual_window_size
        else:
            save_start = overlap
            save_end = overlap + step_size

        save_start = min(save_start, actual_window_size)
        save_end = min(save_end, actual_window_size)

        if save_start < save_end:
            yield window_data, save_start, save_end, output_slice_idx

        if end == num_slices:
            break

        current_slice += step_size
        output_slice_idx += (save_end - save_start)
