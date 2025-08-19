import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import spdiags, csc_matrix
import pyamg
import cupy as cp

def assemble_laplacian_3d(dims, voxel_size, float_type=np.float64):
    """Assembles the 3D Laplacian operator as a sparse matrix with a specific float type."""
    nz, ny, nx = dims
    n = nz * ny * nx

    vx, vy, vz = voxel_size
    dx2, dy2, dz2 = float_type(vx*vx), float_type(vy*vy), float_type(vz*vz)

    # Coefficients for the 7-point stencil
    c_center = float_type(2/dx2 + 2/dy2 + 2/dz2)
    c_x = float_type(-1/dx2)
    c_y = float_type(-1/dy2)
    c_z = float_type(-1/dz2)

    # Create diagonals
    diagonals = [c_center * np.ones(n, dtype=float_type),
                 c_x * np.ones(n-1, dtype=float_type), c_x * np.ones(n-1, dtype=float_type),
                 c_y * np.ones(n-nx, dtype=float_type), c_y * np.ones(n-nx, dtype=float_type),
                 c_z * np.ones(n-nx*ny, dtype=float_type), c_z * np.ones(n-nx*ny, dtype=float_type)]

    offsets = [0, -1, 1, -nx, nx, -nx*ny, nx*ny]

    A = spdiags(diagonals, offsets, n, n, format='lil')
    A = A.astype(float_type) # Ensure matrix is of the correct type

    # Correct boundary conditions for finite differences
    for i in range(n):
        if i % nx == 0: # Left face
            if i > 0: A[i, i-1] = 0
        if (i+1) % nx == 0: # Right face
            if i < n-1: A[i, i+1] = 0

    return A.tocsc()


def solve_poisson_cpu(binary_volume, voxel_size, precision='32-bit Float', matrix_free=False, num_iter=100):
    """
    Solves the Poisson equation for anti-aliasing on the CPU.
    Can use a matrix-based AMG solver or a matrix-free Jacobi iteration.
    """
    float_type = np.float32 if precision == '32-bit Float' else np.float16

    # --- 1. Calculate Guidance Vector Field ---
    # Explicitly set the dtype for distance transforms
    distance = (ndi.distance_transform_edt(binary_volume, sampling=voxel_size) -
                ndi.distance_transform_edt(1 - binary_volume, sampling=voxel_size)).astype(float_type)

    grad_z, grad_y, grad_x = np.gradient(distance) # Voxel size is already in the distance

    norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2).astype(float_type)
    norm[norm == 0] = 1
    g_x, g_y, g_z = (grad_x/norm).astype(float_type), (grad_y/norm).astype(float_type), (grad_z/norm).astype(float_type)

    # --- 2. Compute Divergence of the Guidance Field (RHS) ---
    div_g = (np.gradient(g_z, axis=0) / voxel_size[2] +
             np.gradient(g_y, axis=1) / voxel_size[1] +
             np.gradient(g_x, axis=2) / voxel_size[0]).astype(float_type)

    u = binary_volume.astype(float_type)
    boundary_mask = (binary_volume == 0) | (binary_volume == 1)

    if not matrix_free:
        # --- 3a. Matrix-based AMG Solver ---
        dims = binary_volume.shape
        n = np.prod(dims)

        A = assemble_laplacian_3d(dims, voxel_size, float_type=float_type)
        b = -div_g.flatten()

        # Apply Dirichlet boundary conditions
        boundary_indices = np.where(boundary_mask.flatten())[0]

        # Adjust RHS for known boundary values
        for i in boundary_indices:
            b -= A.getrow(i).dot(u.flatten())[0]

        A = A.tolil()
        for i in boundary_indices:
            A[i, :] = 0
            A[i, i] = 1
        A = A.tocsc()

        b[boundary_indices] = u.flatten()[boundary_indices]

        # Solve the system
        u_flat = pyamg.solve(A, b, verb=False, tol=1e-4)
        u = u_flat.reshape(dims)

    else:
        # --- 3b. Matrix-free Jacobi Iteration (Vectorized) ---
        vx, vy, vz = voxel_size
        dx2, dy2, dz2 = float_type(vx*vx), float_type(vy*vy), float_type(vz*vz)

        c_center_inv = float_type(1.0 / (2/dx2 + 2/dy2 + 2/dz2))

        for _ in range(num_iter):
            u_old = u.copy()

            term_x = (u_old[:, :, :-2] + u_old[:, :, 2:]) / dx2
            term_y = (u_old[:, :-2, :] + u_old[:, 2:, :]) / dy2
            term_z = (u_old[:-2, :, :] + u_old[2:, :, :]) / dz2

            laplacian_u = np.zeros_like(u, dtype=float_type)
            laplacian_u[:, :, 1:-1] += term_x
            laplacian_u[:, 1:-1, :] += term_y
            laplacian_u[1:-1, :, :] += term_z

            new_u = (laplacian_u - div_g) * c_center_inv

            u[~boundary_mask] = new_u[~boundary_mask]

    # --- 4. Finalize ---
    u = np.clip(u, 0, 1)
    return (u * 255).astype(np.uint8)

def solve_poisson_gpu(binary_volume, voxel_size, precision='32-bit Float', matrix_free=False, num_iter=100):
    """
    Solves the Poisson equation for anti-aliasing on the GPU using CuPy.
    """
    cupy_float_type = cp.float32 if precision == '32-bit Float' else cp.float16
    numpy_float_type = np.float32 if precision == '32-bit Float' else np.float16

    # --- 0. Data Preparation on CPU ---
    # Perform CPU-bound tasks first with correct precision
    distance_cpu = (ndi.distance_transform_edt(binary_volume, sampling=voxel_size) -
                    ndi.distance_transform_edt(1 - binary_volume, sampling=voxel_size)).astype(numpy_float_type)

    # --- 1. Move data to GPU and Calculate Guidance Field ---
    distance = cp.asarray(distance_cpu)
    binary_volume_gpu = cp.asarray(binary_volume)

    grad_z, grad_y, grad_x = cp.gradient(distance)

    norm = cp.sqrt(grad_x**2 + grad_y**2 + grad_z**2).astype(cupy_float_type)
    norm[norm == 0] = 1
    g_x, g_y, g_z = (grad_x/norm).astype(cupy_float_type), (grad_y/norm).astype(cupy_float_type), (grad_z/norm).astype(cupy_float_type)

    # --- 2. Compute Divergence of the Guidance Field (RHS) ---
    div_g = (cp.gradient(g_z, axis=0) / voxel_size[2] +
             cp.gradient(g_y, axis=1) / voxel_size[1] +
             cp.gradient(g_x, axis=2) / voxel_size[0]).astype(cupy_float_type)

    u = binary_volume_gpu.astype(cupy_float_type)
    boundary_mask = (binary_volume_gpu == 0) | (binary_volume_gpu == 1)

    # --- 3. Matrix-free Jacobi Iteration (Vectorized on GPU) ---
    vx, vy, vz = voxel_size
    dx2, dy2, dz2 = cupy_float_type(vx*vx), cupy_float_type(vy*vy), cupy_float_type(vz*vz)

    c_center_inv = cupy_float_type(1.0 / (2/dx2 + 2/dy2 + 2/dz2))

    for _ in range(num_iter):
        u_old = u.copy()

        term_x = (u_old[:, :, :-2] + u_old[:, :, 2:]) / dx2
        term_y = (u_old[:, :-2, :] + u_old[:, 2:, :]) / dy2
        term_z = (u_old[:-2, :, :] + u_old[2:, :, :]) / dz2

        laplacian_u = cp.zeros_like(u, dtype=cupy_float_type)
        laplacian_u[:, :, 1:-1] += term_x
        laplacian_u[:, 1:-1, :] += term_y
        laplacian_u[1:-1, :, :] += term_z

        new_u = (laplacian_u - div_g) * c_center_inv

        u = cp.where(boundary_mask, u, new_u)

    # --- 4. Finalize and move data back to CPU ---
    u = cp.clip(u, 0, 1)
    return cp.asnumpy((u * 255).astype(cp.uint8))
