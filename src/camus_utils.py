import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

"""
This file contains utility functions to read .mhd/.raw cine files in the CAMUS dataset.
For the purpose of documentation, the following terms are used:
    1) echo cine - files following the format of "patientxxxx_xCH_sequence.mhd"
    2) ES/ED frame - files following the format of "patientxxxx_xCH_Ex.mhd"
    3) ground truth - files following the format of "patientxxxx_xCH_Ex_gt.mhd"

TODO: determine if spacing (pixel size information) is required; currently it is not used to display the cine frames.
"""

def load_cine(file):
    """
    Returns an image array of the cine/frame in the form of [height x width x frames] and the pixel size information of the file.
    :param file: str; path to the .mhd file to be read
    :return: [cine, spacing] array;
             cine - array representing the cine/frame with dimensions of [height x width x frames]
             spacing - an array representing the pixel size
                       e.g. [1, 2, 3] = pixel size of 1mm x 2mm (height x width), with each frame lasting 3 time units
    """
    # Reads the image using SimpleITK
    itk_image = sitk.ReadImage(file)

    # Convert the image to numpy array of dimensions [height x width x frames]
    im = sitk.GetArrayFromImage(itk_image)
    cine = np.swapaxes(im, 0, 2)
    cine = np.swapaxes(cine, 0, 1)

    # Read the spacing of each dimension in [height x width x frames] order
    space = np.asarray(itk_image.GetSpacing())
    spacing = np.array([space[1], space[2], space[0]])

    return cine, spacing

def display_cine_frame(cine, frame):
    """
    Use to display a single echo cine frame from the file loaded using the load_cine() function
    :param cine: [cine, spacing] array; output from load_cine(), must be an echo cine (_sequence)
    :param frame: int, number of the frame to be displayed (zero-indexed)
    """
    plt.imshow(cine[0][:, :, frame], cmap='gray', vmin=0, vmax=255)

def display_frame(frame):
    """
    Use to display the ES/ED frame or ground truth frame from the file loaded using the load_cine() function
    :param frame: [cine, spacing] array; output from load_cine(), must be an ES/ED frame (_Ex) or a ground truth (_Ex_gt)
    """
    plt.imshow(frame[0], cmap='gray')