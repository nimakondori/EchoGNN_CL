import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def _loadvideo(filename: str):
    """
    Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

    :param filename: str, path to video to load
    :return: numpy array of dimension H*W*T
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v[:, :, count] = frame

    return v


if __name__ == "__main__":
    file_name = "0X1A6699291D449F74.avi"
    file_path = "C:\\Users\\nimako\\Documents\\workspace\\datasets\\echonet\\Videos"
    ed_frame = 47
    es_frame = 63

    video = _loadvideo(os.path.join(file_path, file_name))
    print(video.shape)
    for i in range(ed_frame-2, ed_frame+2):
        frame = video[:, :, i]
        plt.imsave(f"frame_{i}.png", frame, cmap="gray")

    for i in range(es_frame-3, es_frame+3):
        frame = video[:, :, i]
        plt.imsave(f"frame_{i}.png", frame, cmap="gray")