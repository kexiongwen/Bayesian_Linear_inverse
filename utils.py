import math
import torch
import numpy as np
from math import log10


M_PI = math.pi

def MAXX(x, y):
    return x if x > y else y


def radonTransform(nt, nx, ny):
    """
    Compute a matrix representing the Radon transform for given parameters.

    This function calculates a matrix used in the Radon transform, which is useful in tomographic
    reconstructions. The matrix represents the linear operation of the Radon transform on an image,
    given specific parameters for the number of angles, and the dimensions of the image.

    Args:
        nt (int): The number of angles for the Radon transform.
        nx (int): The number of pixels in the x-dimension (width) of the image.
        ny (int): The number of pixels in the y-dimension (height) of the image.

    Returns:
        numpy.ndarray: A 2D array representing the Radon transform with size nt*(nx x ny).

    Note:
        The returned matrix can be used to compute the Radon transform of an image by matrix-vector multiplication.
        The matrix is flipped vertically at the end before returning.

    """
    # Calculate the x and y origins
    xOrigin = int(MAXX(0, math.floor(nx / 2)))
    yOrigin = int(MAXX(0, math.floor(ny / 2)))

    # Define the radial and axial resolutions
    Dr = 1
    Dx = 1

    # Calculate the size of the radial dimension based on the input dimensions
    rsize = math.ceil(math.sqrt(float(nx * nx + ny * ny) * Dx) / (2 * Dr)) + 1
    nr = 2 * rsize + 1

    # Initialize x and y tables
    xTable = np.zeros((1, nx))
    yTable = np.zeros((1, ny))
    yTable[0, 0] = (-yOrigin - 0.5) * Dx
    xTable[0, 0] = (-xOrigin - 0.5) * Dx

    # Populate yTable and xTable values
    for i in range(1, ny):
        yTable[0, i] = yTable[0, i - 1] + Dx
    for ii in range(1, nx):
        xTable[0, ii] = xTable[0, ii - 1] + Dx

    # Define angular resolution
    Dtheta = M_PI / nt

    # Calculate the percentage of non-zero values in the sparse matrix
    percent_sparse = 2. / float(nr)
    nzmax = int(math.ceil(float(nr * nt * nx * ny * percent_sparse)))

    # Initialize the sparse matrix and its indices
    R = np.zeros((nr * nt, nx * ny))
    weight = np.zeros((1, nzmax))
    irs = np.zeros((1, nzmax))
    jcs = np.zeros((1, R.shape[1] + 1))

    k = 0
    for m in range(ny):
        for n in range(nx):
            jcs[0, m * nx + n] = k
            for j in range(nt):
                # Calculate the angle
                angle = M_PI / 2 - j * Dtheta
                cosine = math.cos(angle)
                sine = math.sin(angle)
                xCos = yTable[0, m] * cosine + rsize * Dr
                ySin = xTable[0, n] * sine
                rldx = (xCos + ySin) / Dr
                rLow = math.floor(rldx)
                pixelLow = 1 - rldx + rLow

                # Check bounds and update the matrix
                if 0 <= rLow < (nr - 1):
                    irs[0, k] = nr * j + rLow  # irs stores the row indices of the non-zero values
                    weight[0, k] = pixelLow
                    k = k + 1
                    irs[0, k] = nr * j + rLow + 1
                    weight[0, k] = 1 - pixelLow
                    k = k + 1
        jcs[0, nx * ny] = k

    # Fill the sparse matrix R using the calculated weights
    for col in range(nx * ny):
        for row in range(2 * nt):
            R[int(irs[0, col * 2 * nt + row]), col] = weight[0, col * 2 * nt + row]

    return np.flipud(R)


def initial(A,y,device):
    
    _,P = A.shape    
    pixel = int(P**0.5)
    
    y0 = torch.from_numpy(y).to(device)
    A0 = torch.from_numpy(A.copy()).to(device)
    
    return torch.linalg.solve(A0.T@A0+10*torch.diag(torch.ones(pixel**2,device = device)),A0.T@y0)


def PSNR(ground_truth, predict):
    """
    """
    ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    predict = (predict - predict.min()) / (predict.max() - predict.min())
    mse = np.mean((ground_truth - predict) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return -1
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return np.round(psnr,2)
