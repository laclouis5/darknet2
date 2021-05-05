import cv2 as cv
import numpy as np

def radians(degrees):
    """
    Convert degrees to radians.

    Args:
        - degrees (float): angle in degrees

    Returns:
        - float: angle in randians
    """
    return degrees * np.pi / 180

def degrees(radians):
    """
    Convert radians to degrees.

    Args:
        - degrees (float): angle in radians

    Returns:
        - float: angle in degrees
    """
    return radians / np.pi * 180

def bgr_to_gray(image):
    """
    Convert a BGR image to grayscale.

    Args:
        - image (NPArray): image to convert.

    Returns:
        - NPArray: a copy of the image converted to grayscale
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def bgr_to_rgb(image):
    """
    Convert a BGR image to RGB image.

    Args:
        - image (NPArray): image to convert.

    Returns:
        - NPArray: a copy of the image converted to RGB
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def egi(image):
    """
    Returns the Excess Green Index of an image. Image can be RGB or BGR.

    Args:
        - image (NPArray): input image

    Returns:
        - NPArray: image converted to EGI
    """
    return np.sum(np.array([-1, 2, -1]) * image, axis=2)

def egi_mask(image, thresh=30):
    """
    Computed a binary mask with 'true' for pixels representing vegetation.

    Args:
        - image (NPArray): input image to filter
        - thresh (float): a hyperparameter to select less or more pixels

    Returns:
        - (NPArray): binary mask of pixels representing vegetation
    """
    img_h, img_w = image.shape[:2]
    small_area = int(0.25/100 * img_w * img_h)

    image_np = image.astype(np.float)
    image_egi = egi(image)
    image_gf = filters.gaussian(image_egi, sigma=1, mode="reflect")
    image_bin = image_gf > thresh
    image_out = morphology.remove_small_objects(image_bin, small_area)
    image_out = morphology.remove_small_holes(image_out, small_area)

    return image_out

def get_transformation(image_size,
    rx=0, ry=0, rz=0,
    dx=0, dy=0, dz=0
):
    """
    Returns a 3x3 transformation matrix to apply to 2D images. Anchor point for
    rotations is the center of the image.

    Args:
        - image_size (float, float): (height, width) of the image
        - rx, ry, rz (float): rotation angle in radian
        - dx, dy, dz (float): translation in pixels

    Returns:
        - (NPArray): transformation matrix
    """
    (h, w) = image_size

    a1 = np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0,    1],
        [0, 0,    1]
    ])

    r_x = np.array([
        [1,          0,           0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx),  np.cos(rx), 0],
        [0,          0,           0, 1]
    ])

    r_y = np.array([
        [np.cos(ry), 0, -np.sin(ry), 0],
        [0,          1,           0, 0],
        [np.sin(ry), 0,  np.cos(ry), 0],
        [0,          0,           0, 1]
     ])

    r_z = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz),  np.cos(rz), 0, 0],
        [0,                    0, 1, 0],
        [0,                    0, 0, 1]
    ])

    r = np.dot(np.dot(r_x, r_y), r_z)
    d = np.hypot(h, w)
    f = d / (2 * np.sin(rz) if np.sin(rz) != 0 else 1)
    dz = dz + f

    t = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

    a2 = np.array([
        [f, 0, w/2, 0],
        [0, f, h/2, 0],
        [0, 0,   1, 0]
    ])

    mat = np.dot(a2, np.dot(t, np.dot(r, a1)))
    return mat

def warp_perspective(img, transformation):
    """
    Returns transformed image according to the given transformation.

    Args:
        - img (NPArray): image to be transformed
        - transformation (NPArray): 3x3 transformation matrix

    Returns:
        - NPArray: transformed image
    """
    (h, w) = img.shape[:2]
    return cv.warpPerspective(img, transformation, (w, h))

def basler3M_calibration_maps(image_size=None):
    """
    Returns callibration map for Basler 3M cameras.

    Args:
        - image_size (int, int): (height, width) of the image to be calibrated. Leave empty if image in the original resolution (1536, 2048).

    Returns:
        - (NPArray, NPArray): maps for respectively X and Y axis
    """
    original_img_size = (2048, 1536)

    mtx = np.array([[1846.48412, 0.0,        1044.42589],
                    [0.0,        1848.52060, 702.441180],
                    [0.0,        0.0,        1.0]])

    dist = np.array([[-0.19601338, 0.07861078, 0.00182995, -0.00168376, 0.02604818]])

    new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(mtx, dist, original_img_size, 0, original_img_size)
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, new_camera_matrix, original_img_size, m1type=cv.CV_32FC1)

    if image_size is not None:
        h, w = image_size
        mapx = cv.resize(mapx, (w, h)) * w / original_img_size[0]
        mapy = cv.resize(mapy, (w, h)) * h / original_img_size[1]

    return (mapx, mapy)

def calibrated(img, mapx=None, mapy=None):
    """
    Calibrate an image according to Basler 3M camera distorsion.

    Args:
        - img (NPArray): image to be calibrated
        - mapx, mapy (NPArray): calibration maps for Basler 3M camera. Leave empty to automatically compute the maps with respect to the input image size (can be slow).

    Returns:
        - (NPArray): calibrated image
    """
    if mapx is not None and mapy is not None:
        return cv.remap(img, mapx, mapy, interpolation=cv.INTER_CUBIC)
    elif mapx is None and mapy is None:
        mapx, mapy = basler3M_calibration_maps(image_size=img.shape[:2])
        return cv.remap(img, mapx, mapy, interpolation=cv.INTER_CUBIC)
    else:
        AssertionError("You should either provide the two calibration maps or none.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = "/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2/bob/im_03301.jpg"
    img = cv.imread(image)

    t = get_transformation(img.shape[:2])

    warped = warp_perspective(img, t)
    cv.imwrite("warped.jpg", bgr_to_rgb(warped))

    plt.imshow(bgr_to_rgb(warped))
    plt.show()
