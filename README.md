# mSLA Anti-Aliasing Processor

This application is a proof-of-concept for anti-aliasing 3D print slices (for mSLA / LCD printers) using an anisotropic Poisson gradient solver. It provides a graphical user interface to load a stack of 2D image slices, process them to create smooth, anti-aliased grayscale versions, and save the result.

## Features

- **Graphical User Interface**: Built with PySide6 for a user-friendly experience.
- **Anisotropic Solver**: Accounts for different voxel dimensions in X, Y, and Z.
- **Sliding Window Processing**: Efficiently handles large datasets by processing them in chunks.
- **Advanced Solver Options**:
  - **CPU & GPU Backends**: Choose between NumPy (CPU) or CuPy (GPU) for processing.
  - **Matrix & Matrix-Free Modes**: Select between a memory-intensive but fast matrix-based AMG solver or a memory-efficient iterative solver.
  - **Selectable Precision**: Use 32-bit or 16-bit floating-point precision.
- **LUT Support**: Apply a custom material look-up table to the final output.
- **Multi-threaded**: The processing is done in a background thread to keep the UI responsive.

## Installation

### 1. Prerequisites
- Python 3.8+
- For GPU acceleration: An NVIDIA GPU with a compatible CUDA Toolkit installed.

### 2. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Install Dependencies
A `requirements.txt` file is provided to install the necessary Python packages.

```bash
pip install -r requirements.txt
```

### 4. Important: Installing CuPy (for GPU Support)

The `requirements.txt` file includes `cupy-cuda11x`, which is a pre-compiled package for systems with **CUDA Toolkit 11.x**. This may not be the correct version for your system. If you do not have a matching CUDA version, the installation or execution will fail.

**To install the correct CuPy package:**

**a. Check your CUDA version:**
Open a terminal or command prompt and run:
```bash
nvcc --version
```
This will show you the version of the CUDA Toolkit installed on your system.

**b. Choose the right CuPy package:**
- If you have **CUDA 12.x**, you should install `cupy-cuda12x`.
- If you have **CUDA 11.x**, the default `cupy-cuda11x` should work.
- If you have a different version, please refer to the [Official CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html).

**c. Modify `requirements.txt` (if necessary):**
Before running `pip install`, open the `requirements.txt` file and change the line `cupy-cuda11x` to the package that matches your CUDA version (e.g., `cupy-cuda12x`).

**d. If you don't have an NVIDIA GPU:**
You can still use the application in CPU mode. Simply remove the `cupy-cuda11x` line from `requirements.txt` before installing.

## Usage

To run the application, execute the `main.py` script from the project's root directory:

```bash
python src/msla_antialiasing/main.py
```

Alternatively, you can navigate into the `src/msla_antialiasing` directory and run it from there:
```bash
cd src/msla_antialiasing
python main.py
```
The script is configured to work from this location as well.
