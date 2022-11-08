import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2

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


def load_image(filepath: str):
    """
    Load each image as SimpleITK.Image file given the path to the image/sequence
    :param filepath: the filepath to the image file. Makre sure it has a .mhd suffix
    """
    image = sitk.ReadImage(filepath, sitk.sitkVectorUInt8)
    return image


def write_video_data(image, output_name, reslice_axis=2, fps=None):
    """
    Create a video clip scrolling through a 3D image along a specific axis and
    write the result to disk. The result can be either the individual frames
    or a video clip. The later depends on OpenCV which is not installed by
    default. The former needs to be assembled into a video using an external tool
    (e.g. ffmpeg).
    Args:
        image (SimpleITK.Image): 3D input image.
                                    NOTE: Pixel spacings in the non-scrolling axes
                                        are expected to be the same (isotropic in
                                        that plane). Use the make_isotropic function
                                        if they aren't.
                                        Pixel type is either sitkUInt8 or sitkVectorUInt8
                                        with three components which are assumed to be RGB.
        output_name (str): A file name prefix. If the frames-per-second parameter is None then this is
                            used as the prefix for saving individual frames in png format. Otherwise,
                            this is used as the prefix for saving the video in mp4 format.
        reslice_axis (int): Number in [0,1,2]. The axis along which we scroll.
        fps (int): Frames per second.
    """
    # Frames are always along the third axis, so set the axis of interest to be the
    # third axis via PermuteAxes
    permute_axes = [0, 1, 2]
    permute_axes[2] = reslice_axis
    permute_axes[reslice_axis] = 2
    image = sitk.PermuteAxes(image, permute_axes)
    print(image.GetDepth())

    if fps is None:  # write slices as individual frames
        sitk.WriteImage(
            image, [f"{output_name}{i:03d}.png" for i in range(image.GetDepth())]
        )
    else:  # use OpenCV to write slices as mp4 video
        video_writer = cv2.VideoWriter(
            filename=output_name + ".mp4",
            fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps=fps,
            frameSize=image.GetSize()[0:2],
            isColor=image.GetNumberOfComponentsPerPixel() == 3,
        )
        for i in range(image.GetDepth()):
            video_writer.write(sitk.GetArrayViewFromImage(image[:, :, i]))
        video_writer.release()