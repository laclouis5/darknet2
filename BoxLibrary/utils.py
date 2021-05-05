from enum import Enum
import os
import PIL
from collections import defaultdict
from pathlib import Path



class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EveryPointInterpolation = 0
    ElevenPointInterpolation = 1


class EvaluationMethod(Enum):
    """
    Method to compute precision and recall.

        Created by Louis Lac (2020)
    """
    IoU = 0
    Distance = 1


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    Relative = 0
    Absolute = 1


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    GroundTruth = 0
    Detected = 1


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    XYWH = 0  # (x_min, y_min, width, height)
    XYX2Y2 = 1  # (x_min, y_min, x_max, y_max)
    XYC = 2  # (x_mid, y_mid, width, height)


# Equivalent to xyx2y2_to_xywh()
def convertToAbsCenterValues(xmin, ymin, xmax, ymax):
    x = (xmax + xmin) / 2.0
    y = (ymax + ymin) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return (x, y, w, h)


def convertToRelativeValues(size, box):
    # box is (xmin, ymin, xmax, ymax)
    x = (box[1] + box[0]) / 2.0 / size[0]
    y = (box[3] + box[2]) / 2.0 / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]

    return (x, y, w, h)


def convertToAbsoluteValues(size, box):
    # box is (x, y, w, h)
    xIn = (box[0] - box[2] / 2) * size[0]
    yIn = (box[1] - box[3] / 2) * size[1]
    xEnd = (box[0] + box[2] / 2) * size[0]
    yEnd = (box[1] + box[3] / 2) * size[1]

    return (xIn, yIn, xEnd, yEnd)


def files_with_extension(folder, extension):
    return [os.path.join(folder, item)
            for item in os.listdir(folder)
            if os.path.splitext(item)[1] == extension]


def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def image_size(image):
    return PIL.Image.open(image).size


def dictGrouping(iterable, key):
    d = defaultdict(list)
    for element in iterable:
        d[key(element)].append(element)
    return d