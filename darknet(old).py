#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"

To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""

#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os
import datetime
import time
import joblib

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
try:
    import cv2 as cv
except:
    pass
from PIL import Image
from skimage import io, filters, morphology
from joblib import Parallel, delayed
from pathlib import Path

# from my_library import read_detection_txt_file, save_yolo_detect_to_txt, yolo_det_to_bboxes, save_bboxes_to_txt, nms, create_dir, parse_yolo_folder, xyx2y2_to_xywh, xywh_to_xyx2y2, remap_yolo_GT_file_labels, remap_yolo_GT_files_labels, clip_box_to_size, optical_flow, convert_to_grayscale, egi_mask, Track, Tracker, associate_boxes_with_image, gts_in_unique_ref, optical_flow_visualisation, evaluate_aggr, read_image_txt_file, associate_tracks_with_image, draw_tracked_confidence_ellipse, OpticalFlow, Equation, move_gts_in_unique_ref, normalized
from my_library import *

from reg_plane import fit_plane, reg_score, BivariateFunction

from BoxLibrary import *
from sort import *
from tqdm.contrib import tenumerate
from tqdm import tqdm


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i, p in enumerate(probs):
        r = r - p
        if r <= 0:
            return i
    return len(probs) - 1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = [k for k, v in os.environ.items()]
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if (
                'CUDA_VISIBLE_DEVICES' in envKeys
                and int(os.environ['CUDA_VISIBLE_DEVICES']) < 0
            ):
                raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
                    # print(os.environ.keys())
                    # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        nameTag = meta.names[i] if altNames is None else altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms)
    free_image(im)
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]

    if nms:
        do_nms_sort(dets, num, meta.classes, nms)

    res = []
    names = altNames or meta.names

    for j in range(num):
        det = dets[j]
        for i in range(meta.classes):
            p = det.prob[i]
            if p > 0:
                b = det.bbox
                name_tag = names[i]
                res.append((name_tag, p, (b.x, b.y, b.w, b.h)))
    res.sort(key=lambda x: x[1], reverse=True)
    free_detections(dets, num)
    return res

netMain = None
metaMain = None
altNames = None

def performDetect(imagePath="data/dog.jpg", thresh=0.25, configPath="./cfg/yolov4.cfg", weightPath="yolov4.weights", metaPath="./cfg/coco.data", showImage=False, makeImageOnly=False, initOnly=False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found

    showImage: bool (default= True)
        Compute (and show) bounding boxes. Changes return.

    makeImageOnly: bool (default= False)
        If showImage is True, this won't actually *show* the image, but will create the array and return it.

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    When showImage is False, list of tuples like
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.

    Otherwise, a dict with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    
    if netMain is None:
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1

    if metaMain is None:
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
        metaMain = load_meta(metaPath.encode("ascii"))

    if altNames is None:
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect

        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                result = match.group(1) if match else None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    if initOnly:
        print("Initialized detector")
        return None

    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    
    # Do the detection
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    
    if showImage:
        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
            print(detections)
        except Exception as e:
            print("Unable to show image: "+str(e))
    return detections

def convertBack(x, y, w, h):
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax

def get_classes(obj):
    metadata = load_meta(obj.encode("ascii"))
    nb_classes = metadata.classes
    return [metadata.names[i].decode() for i in range(nb_classes)]

# from BoundingBox import BoundingBox
# from BoundingBoxes import BoundingBoxes
# from Evaluator import Evaluator
# from test import read_txt_annotation_file, parse_yolo_folder
# from utils import *
#
# def compute_mean_crop_annotation_to_squareaverage_precision(folder, model, config_file, data_obj):
#     files = os.listdir(folder)
#     images = [os.path.join(folder, file) for file in files if os.path.splitext(file)[1] == '.jpg']
#
#     bounding_boxes = parse_yolo_folder(folder)
#     bounding_boxes.mapLabels({0: "mais", 1: 'haricot', 2: 'carotte'})
#
#     for image in images:
#         detections = performDetect(
#             imagePath=image,
#             configPath=config_file,
#             weightPath=model,
#             metaPath=data_obj,
#             showImage=False)
#
#         img_size = Image.open(image).size
#
#         for detection in detections:
#             label, conf = detection[0], detection[1]
#             # Abs XYX2Y2
#             x_min, y_min, x_max, y_max = convertBack(*detection[2])
#
#             bounding_boxes.addBoundingBox(BoundingBox(
#                 imageName=os.path.basename(image),
#                 classId=label,
#                 x=x_min, y=y_min, w=x_max, h=y_max,
#                 bbType=BBType.Detected, classConfidence=conf,
#                 format=BBFormat.XYX2Y2,
#                 imgSize=img_size))
#
#     evaluator = Evaluator()
#     metrics = evaluator.GetPascalVOCMetrics(bounding_boxes)
#     for item in metrics:
#         (prec,  rec) = item["precision"], item["recall"]
#         print("{} - mAP: {:.4} %, TP: {}, FP: {}, tot. pos.: {}".format(item['class'], 100*item['AP'], item["total TP"], item["total FP"], item["total positives"]))

###########################
# Created by Louis LAC 2019
###########################

class YoloModelPath:
    def __init__(self, model_folder):
        cfg = files_with_extension(model_folder, ".cfg")[0]
        weights = files_with_extension(model_folder, ".weights")[0]
        meta = files_with_extension(model_folder, ".data")[0]

        if not os.path.exists(cfg):
            raise ValueError(f"Invalid config file path '{cfg}'")
        if not os.path.exists(weights):
            raise ValueError(f"Invalid weights file path '{weights}'")
        if not os.path.exists(meta):
            raise ValueError(f"Invalid metadata file path '{meta}'")

        self.cfg = cfg
        self.weights = weights
        self.meta = meta

    def get_cfg_weight_meta(self):
        return(self.cfg, self.weights, self.meta)

def save_yolo_detections(network, file_dir, save_dir="", bbCoords=CoordinatesType.Absolute, bbFormat=BBFormat.XYX2Y2, conf_thresh=0.25):
    model = network["model"]
    cfg = network["cfg"]
    obj = network["obj"]

    create_dir(save_dir)
    boxes = BoundingBoxes()
    images = files_with_extension(file_dir, ".jpg")

    for image in images:
        img_size = image_size(image)
        detections = performDetect(image, thresh=conf_thresh, configPath=cfg, weightPath=model, metaPath=obj, showImage=False)
        boxes += Parser.parse_yolo_darknet_detections(detections, image, img_size)

    boxes.save(bbCoords, bbFormat, save_dir)


def performDetectOnFolder(network, directory, conf_thresh=0.5/100):
    (cfg, model, obj) = network.get_cfg_weight_meta()
    images = files_with_extension(directory, ".jpg")
    # images = sorted(images)[::30]
    boxes = BoundingBoxes()

    for image in tqdm(images, desc="Inference", unit="image"):
        img_size = image_size(image)
        detections = performDetect(image, conf_thresh, configPath=cfg, weightPath=model, metaPath=obj, showImage=False)
        boxes += Parser.parse_yolo_darknet_detections(detections, image, img_size)

    return boxes


def performDetectOnTxtFile(txt_file, network, thresh=0.25, n_proc=1):
    (cfg, weights, meta) = network.get_cfg_weight_meta()
    images = read_image_txt_file(txt_file)

    def inner(image):
        detections = performDetect(image,
            thresh=thresh, configPath=cfg, weightPath=weights, metaPath=meta, showImage=False)
        return Parser.parse_yolo_darknet_detections(detections, image, image_size(image))

    boxes = Parallel(n_jobs=n_proc, verbose=10)(delayed(inner)(image) for image in images)

    return BoundingBoxes([box for wrapped in boxes for box in wrapped])


def performDetectOnFolderAndTrack(network, txt_file, conf_thresh=0.25, max_age=1, min_hits=3):
    """
    Takes as input a path to a folder of images in chronological order by name:
        * t=0 -> im_01.jpg
        * t=1 -> im_02.jpg
        * ...
    Takes as input a yolo detector network

    Returns filtered detections using KalmanFilters and Optical Flow estimation.
    """
    # Darknet Yolo model params
    model = network.weights
    cfg = network.cfg
    obj = network.meta

    # Files are in chronological order
    images = []
    with open(txt_file, "r") as f:
        images = f.readlines()
    images = [image.strip() for image in images]

    # Network init
    performDetect("", thresh=conf_thresh, configPath=cfg, weightPath=model, metaPath=obj, showImage=False, initOnly=True)

    # Init various stuff
    create_dir(save_dir)
    labels = get_classes(obj)
    trackers = {label: Sort(max_age, min_hits) for label in labels}
    all_boxes = BoundingBoxes()
    opt_flow = None
    past_image = cv.imread(images[0])
    image_count = len(images) - 1
    opt_flows = []

    # Main loop
    for i, image in enumerate(images[1:]):
        print("{:.4}%".format((i + 1) / image_count * 100))
        print("IMAGE: {}".format(image))
        # Yolo detections
        detections = performDetect(image, thresh=conf_thresh, configPath=cfg, weightPath=model, metaPath=obj, showImage=False)
        det_boxes = Parser.parse_yolo_darknet_detections(detections, image, img_size=image_size(image))

        # Optical flow
        current_image = cv.imread(image)
        opt_flow, past_image = optical_flow(past_image, current_image, opt_flow)
        egi = egi_mask(current_image)
        dx, dy = mean_opt_flow(opt_flow, egi)
        opt_flows.append((dx, dy))
        print("OPT FLOW: {} {}".format(dx, dy))

        # Per label loop
        for label in labels:
            label_boxes = det_boxes.getBoundingBoxByClass(label)
            # Update tracker with detections and optical flow informations
            tracks = trackers[label].update(label_boxes.getDetectionBoxesAsNPArray(), (dx, dy))

            # Save filtered detections
            boxes = [BoundingBox(image, label, *track[:4], CoordinatesType.Absolute, image_size(image), BBType.Detected, 1, BBFormat.XYX2Y2) for track in tracks]
            all_boxes += [box.cliped() for box in boxes if box.centerIsIn()]

    return all_boxes, opt_flows

def drawConstellation(txt_file, nb_samples=10, offset=0):
    # Files are in chronological order
    images = []
    with open(txt_file, "r") as f:
        images = f.readlines()
    images = [image.strip() for image in images]

    x_values = []
    y_values = []
    colors = []

    color_map = {str(k): k / 5 for k in range(6)}
    print(color_map)

    for image in images[offset:nb_samples + offset]:
        gt = os.path.splitext(image.strip())[0] + ".txt"
        boxes = Parser.parse_yolo_gt_file(gt)

        for box in boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
            x_values.append(x)
            y_values.append(y)
            colors.append(color_map[box.getClassId()])

    import matplotlib.pyplot as plt

    image = images[nb_samples + offset - 1]
    (width, height) = image_size(image)
    im = plt.imread(image)
    plt.imshow(im)
    plt.scatter(x_values, y_values, c=colors, cmap="rainbow")
    plt.xlim((0, width))
    plt.ylim((height, 0))
    plt.show()

def drawConstellationFlat(txt_file, folder, opt_flow, label):
    flows = OpticalFlow.read(opt_flow)  # Std way
    # flows = OpticalFlow.read_planes("data/optflow_haricot_seq_plane.txt")  # Plane way

    boxes = Parser.parse_xml_folder(folder, classes=[label])
    images = read_image_txt_file(txt_file)

    # Init stuff and main loop
    for (i, image) in enumerate(images):
        image_boxes = boxes.getBoundingBoxesByImageName(image)
        for box in image_boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            (dx, dy) = OpticalFlow.traverse_backward(flows[:i+1], x, y)
            box.moveBy(dx, dy)

    # Plot stuff
    plt.figure()
    for box in boxes:
        (x, y, _, _) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
        plt.scatter(x, y, c="red", marker=".")
    plt.title("Constellation")
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.ylim([632, 0])
    plt.show()

def drawConstellationFlatBoxes(txt_file, boxes, folder, opt_flow, label):
    flows = OpticalFlow.read(opt_flow)  # Std way
    # flows = OpticalFlow.read_planes("data/optflow_haricot_seq_plane.txt")  # Plane way
    images = read_image_txt_file(txt_file)
    boxes = boxes.getBoundingBoxByClass(label)

    # Init stuff and main loop
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    for (i, image) in enumerate(images):
        image_boxes = boxes_by_name[image]
        for box in image_boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            (dx, dy) = OpticalFlow.traverse_backward(flows[:i+1], x, y)
            box.moveBy(dx, dy)

    # Plot stuff
    plt.figure()
    for box in boxes:
        (x, y, _, _) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
        plt.scatter(x, y, c="red", marker=".")
    plt.title("Constellation")
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.ylim([632, 0])
    plt.show()

def drawConstellationDet(network, gts_dir, txt_file, opt_flow, label, thresh=0.5):
    # Opt flow stuff reading
    opt_flows = []
    with open(opt_flow, "r") as f:
        opt_flows = f.readlines()
        opt_flows = [c.strip().split(" ") for c in opt_flows]
        opt_flows = [(float(c[0]), float(c[1])) for c in opt_flows]

    # Parse images in chronological order
    images = []
    with open(txt_file, "r") as f:
        images = [c.strip() for c in f.readlines()]

    # Yolo params
    model = network.weights
    cfg = network.cfg
    obj = network.meta

    # Gts parsing
    gts = Parser.parse_xml_folder(gts_dir, ["mais_tige"])
    gts.mapLabels(fr_to_en)

    # Detect
    out_boxes = BoundingBoxes()
    dx, dy = 0, 0
    for i, image in enumerate(images[:500]):
        # Opt flow shit
        opt_flow = opt_flows[i]
        dx += opt_flow[0]
        dy += opt_flow[1]
        # Detection things
        img_size = image_size(image)
        detections = performDetect(image, thresh=thresh, configPath=cfg, weightPath=model, metaPath=obj, showImage=False)
        boxes = Parser.parse_yolo_darknet_detections(detections, image, img_size)
        boxes.moveBy(dx=-dx, dy=-dy)
        # Gts
        img_gts = gts.getBoundingBoxesByImageName(image)
        img_gts.moveBy(dx=-dx, dy=-dy)
        # Fill array
        out_boxes += img_gts + boxes

    # Time to plot stuff
    plt.figure()
    labels = out_boxes.getClasses()
    cmap = plt.get_cmap("gist_rainbow")

    for i, label in enumerate([label]):
        color = cmap(i * 1 / len(labels))
        boxes = out_boxes.getBoundingBoxByClass(label)
        detections = boxes.getBoundingBoxesByType(BBType.Detected)
        abs_coords = [box.getAbsoluteBoundingBox(format=BBFormat.XYC) for box in detections]
        x_values = [coord[0] for coord in abs_coords]
        y_values = [coord[1] for coord in abs_coords]
        colors = np.array([[*color[:3],  (box.getConfidence() - thresh) / (1 - thresh)] for box in detections])
        plt.scatter(x_values, y_values, c=colors, marker=".")

        groundTruths = boxes.getBoundingBoxesByType(BBType.GroundTruth)
        abs_coords = [box.getAbsoluteBoundingBox(format=BBFormat.XYC) for box in groundTruths]
        x_values = [coord[0] for coord in abs_coords]
        y_values = [coord[1] for coord in abs_coords]
        plt.scatter(x_values, y_values, c="green", marker="+")

    plt.ylim([632, -632])
    plt.legend(labels)
    plt.show()

# Obsolete?
def save_detect_to_txt(folder_path, save_dir, model, config_file, data_file, conv_back=False):
    """
    Perform detection on images in folder_path with the specified yolo
    model and saves detections in yolo format in save_dir folder.
    """
    img_gen = ImageGeneratorFromFolder(folder_path)
    create_dir(save_dir)

    for image in img_gen:
        detections = performDetect(image, thresh=0.005, configPath=config_file, weightPath=model, metaPath=data_file, showImage=False)

        save_name = os.path.join(save_dir, os.path.splitext(os.path.basename(image))[0]) + ".txt"
        print(save_name)

        (height, width) = cv.imread(image).shape[0:2]

        lines = []
        for detection in detections:
            box = detection[2]
            label = detection[0]
            if conv_back:
            # XminYminXmaxYmax abs
                (x, y, w, h) = convertBack(box[0], box[1], box[2], box[3])
                w = w - 1
                h = h - 1
            else:
                # XYWH relative
                (x, y, w, h) = box[0]/width, box[1]/height, box[2]/width, box[3]/height

            confidence = detection[1]
            lines.append("{} {} {} {} {} {}\n".format(label, confidence, x, y, w, h))

        with open(save_name, 'w') as f:
            f.writelines(lines)


# Obsolete?
def save_gt_to_txt(folder_path, save_dir, map, conv_back=False) :
    files = [os.path.join(folder_path, item) for item in os.listdir(folder_path) if os.path.splitext(item)[1] == ".txt"]

    create_dir(save_dir)

    for file in files:
        content = []
        lines = []
        with open(file, "r") as f:
            content = f.readlines()
            content = [item.strip().split() for item in content]

        for line in content:
            line[0] = map[int(line[0])]

        for cont in content:
            line = ""
            if conv_back:
                image = os.path.splitext(file)[0] + ".jpg"
                (height, width) = cv.imread(image).shape[0:2]
                (xmin, ymin, xmax, ymax) = convertBack(float(cont[1])*width, float(cont[2])*height, float(cont[3])*width, float(cont[4])*height)
                line = "{} {} {} {} {}\n".format(cont[0], xmin, ymin, xmax-1, ymax-1)
            else:
                line = "{} {} {} {} {}\n".format(cont[0], cont[1], cont[2], cont[3], cont[4])

            lines.append(line)

        save_file = os.path.join(save_dir, os.path.basename(file))

        with open(save_file, "w") as f:
            f.writelines(lines)

# Obsolete
def convert_yolo_annot_to_XYX2Y2(annotation_dir, save_dir, lab_to_name):
    """
    Convert annotation files of the specified directory in XYX2Y2 abs format and
    save new files to save_dir. lab_to_name is a dictionnary that allows
    remapping of labels.
    """
    annotations = [os.path.join(annotation_dir, item) for item in os.listdir(annotation_dir) if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]

    for (image, annotation) in zip(images, annotations):
        (img_w, img_h) = Image.open(image).size
        print('Image:      {}'.format(image))
        print('Annotation: {}'.format(annotation))
        print('Image Size: {} x {}'.format(img_w, img_h))

        with open(annotation, 'r') as f:
            content = f.readlines()
        content = [item.strip() for item in content]

        with open(os.path.join(save_dir, os.path.basename(annotation)), 'w') as fw:
            for line in content:
                line = line.split(' ')
                (label, x, y, w, h) = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                # XYX2Y2 absolute
                (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(x*img_w, y*img_h, w*img_w, h*img_h)
                fw.write('{} {} {} {} {}\n'.format(lab_to_name[label], xmin, ymin, xmax, ymax))


# Obsolete
def draw_boxes(image, annotation, save_path, color=[255, 64, 0]):
    '''
    Takes path to one image and to one yolo-style detection file, draws
    bounding boxes into and saves it in save_path
    '''
    create_dir(save_path)
    save_name        = os.path.join(save_path, os.path.basename(image))
    height, width, _ = cv.imread(image).shape

    boxes = read_detection_txt_file(annotation, (width, height))
    img = cv.imread(image)

    for box in boxes.getBoundingBoxes():
        add_bb_into_image(img, box, color=color, label=box.getClassId())

    cv.imwrite(os.path.join(save_name), img)

# Obsolete
def draw_boxes_bboxes(image, bounding_boxes, save_dir, color=[255, 64, 0]):
    '''
    Takes as input one image (numpy array) and a BoundingBoxes object
    representing the bounding boxes, draws them into and saves the image in
    save_dir.
    '''
    image = image.copy()
    for box in bounding_boxes.getBoundingBoxes():
        add_bb_into_image(image, box, color=color, label=box.getClassId())
        image_path = os.path.join(save_dir, box.getImageName())

    cv.imwrite(image_path, image)


def draw_deque_boxes(image, deq, save_path):
    '''
    Takes one numpy array representing the image, a deque with boundingBoxes
    objects. Draws boxes ans save them in save_path.
    '''
    image = image.copy()
    hsv = plt.get_cmap("cool")
    colors = hsv(np.linspace(0, 1, deq.maxlen))[..., :3]

    for i, bboxes in enumerate(reversed(deq)):
        for box in bboxes.getBoundingBoxes():
            add_bb_into_image(image, box, 255*colors[i], label=box.getClassId())

    cv.imwrite(save_path, image)

# Obsolete
def draw_boxes_folder(images_path, annotations_path, save_path):
    '''
    Takes path to a folder if images and a folder of corresponding annotations,
    draws bounding boxes and save images in save_path.
    '''
    create_dir(save_path)
    images = ImageGeneratorFromFolder(images_path)

    for image in images:
        annotation = os.path.splitext(os.path.basename(image))[0] + ".txt"
        annotation = os.path.join(annotations_path, annotation)

        draw_boxes(image, annotation, save_path)


def ImageGeneratorFromVideo(video_path, skip_frames=1, gray_scale=True, down_scale=1, ratio=None):
    '''
    Takes path to a video and yield frames.
    '''
    video = cv.VideoCapture(video_path)
    ret = True
    while ret:
        for _ in range(skip_frames):
            ret, frame = video.read()

        if down_scale > 1:
            frame = frame[::down_scale, ::down_scale]

        if gray_scale:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if ratio is not None:
            height, width = frame.shape[0:2]
            new_width = ratio * height
            to_crop   = int((width - new_width) / 2)

            frame = frame[:, to_crop:-to_crop, :]

        yield (ret, frame)

def ImageGeneratorFromFolder(folder, sorted=False):
    '''
    Generator that yields images from a specified folder. Can be sorted.
    '''
    files = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == ".jpg"]
    if sorted:
        files.sort()
    yield from files

def save_images_from_video(path_to_video, save_dir, skip_frames = 2, nb_iter=100):
    '''
    Extract frames from a video and saves them to save_path. Sampling
    frequency can be tunes as well as the number of frames to extract.
    Ratio is 4/3.
    '''
    video_gen = ImageGeneratorFromVideo(path_to_video, skip_frames=skip_frames, gray_scale=False)

    create_dir(save_dir)

    for i in range(nb_iter):
        _, frame = next(video_gen)
        height, width = frame.shape[0:2]
        new_ratio = 4/3
        new_width = new_ratio * height
        to_crop   = int((width - new_width) / 2)

        frame = frame[:, to_crop:-to_crop, :]

        frame_name = os.path.join(save_dir, "im_{}.jpg".format(i*skip_frames))
        cv.imwrite(frame_name, frame, [int(cv.IMWRITE_JPEG_QUALITY), 100])

def filter_detections(video_path, video_param, save_dir, yolo_param, k=5):
    '''
    NMS filtering and drawing. For experiment purpose.
    '''
    image_dir = os.path.join(save_dir, "images")
    annot_dir = os.path.join(save_dir, "annotations")
    draw_dir  = os.path.join(save_dir, "draw")

    create_dir(save_dir)
    create_dir(image_dir)
    create_dir(annot_dir)
    create_dir(draw_dir)

    images = ImageGeneratorFromVideo(
        video_path,
        skip_frames=video_param["skip_frames"],
        gray_scale=video_param["gray_scale"],
        down_scale=video_param["down_scale"],
        ratio=video_param["ratio"])

    boxes = deque(maxlen=k)
    _, first_image = next(images)

    image_name = os.path.join(image_dir, "im_0.jpg")
    cv.imwrite(image_name, first_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    detections = performDetect(
        image_name,
        configPath = yolo_param["cfg"],
        weightPath =yolo_param["model"],
        metaPath=yolo_param["obj"],
        showImage=False)

    bboxes = yolo_det_to_bboxes("im_0.jpg", detections)
    bboxes.keepOnlyName("leek")
    boxes.append(bboxes)
    save_bboxes_to_txt(bboxes, annot_dir)
    draw_deque_boxes(first_image, boxes, os.path.join(draw_dir, "im_0.jpg"))

    first_image = scale(first_image)
    prev_opt_flow = np.zeros_like(first_image)
    for file_nb, (_, image) in enumerate(images, start=1):
        second_image = convert_to_grayscale(image)

        optical_flow = cv.calcOpticalFlowFarneback(
            prev=first_image,
            next=second_image,
            flow=prev_opt_flow,
            pyr_scale=0.5,
            levels=4,
            winsize=32,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0)

        dx = optical_flow[..., 0]
        dy = optical_flow[..., 1]

        mean_dx = dx.sum() / dx.size
        mean_dy = dy.sum() / dy.size

        image_name = os.path.join(image_dir, "im_{}.jpg".format(file_nb))
        cv.imwrite(image_name, image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        detections = performDetect(
            image_name,
            configPath=yolo_param["cfg"],
            weightPath=yolo_param["model"],
            metaPath=yolo_param["obj"],
            showImage=False)

        bboxes = yolo_det_to_bboxes("im_{}.jpg".format(file_nb), detections)
        bboxes.keepOnlyName("leek")

        # Print stuff
        print(image_name)
        print("  Mean dx: {:.6}  Mean dy: {:.6}".format(mean_dx, mean_dy))
        print("  Detection Count: {}".format(len(bboxes.getBoundingBoxes())))

        [item.shiftBoundingBoxesBy(mean_dx, mean_dy) for item in boxes]
        boxes.append(bboxes)

        # Flatten
        boxes_to_save = []
        [boxes_to_save.extend(item.getBoundingBoxes()) for item in boxes]
        [box.setImageName("im_{}.jpg".format(file_nb)) for box in boxes_to_save]
        boxes_to_save = BoundingBoxes(boxes_to_save)

        # NMS stuff and draw
        boxes_to_save = nms(boxes_to_save)
        draw_deq = deque(maxlen=1)
        draw_deq.append(boxes_to_save)
        draw_deque_boxes(image, draw_deq, os.path.join(draw_dir, "im_{}.jpg".format(file_nb)))
        save_bboxes_to_txt(boxes_to_save, annot_dir)

        first_image = second_image
        prev_opt_flow = optical_flow
        print()


def filter_detections_2(folder, save_dir, model_param, k=5):
    '''
    NMS filtering. For experiment purpose.
    '''
    image_dir = os.path.join(save_dir, "images")
    annot_dir = os.path.join(save_dir, "annotations")
    draw_dir  = os.path.join(save_dir, "draw")

    create_dir(save_dir)
    create_dir(image_dir)
    create_dir(annot_dir)
    create_dir(draw_dir)

    boxes = deque(maxlen=k)

    # Read data file
    lines = []
    with open(os.path.join(folder, "data.txt"), "r") as f_read:
        line = f_read.readline()
        while line:
            lines.append(line.split())
            line = f_read.readline()

    # Manage first image
    image = line[0]

    detections = performDetect(
        image,
        configPath=model_param["cfg"],
        weightPath=model_param["model"],
        metaPath=model_param["obj"],
        showImage=False)

    bboxes = yolo_det_to_bboxes("im_0.jpg", detections)
    save_bboxes_to_txt(bboxes, annot_dir)
    boxes.append(bboxes)
    draw_boxes_bboxes(first_image, bboxes, draw_dir)

    # Process rest of file
    i = 1
    for line in lines[1:]:
        (image, dx, dy, t) = line[0], line[1], line[2], line[3]

        detections = performDetect(
            image,
            configPath=model_param["cfg"],
            weightPath=model_param["model"],
            metaPath=model_param["obj"],
            showImage=False)

        bboxes = yolo_det_to_bboxes("im_{}.jpg".format(i), detections)
        [item.shiftBoundingBoxesBy(dx, dy) for item in boxes]
        boxes.append(bboxes)

        boxes_to_save = []
        [boxes_to_save.extend(item.getBoundingBoxes()) for item in boxes]
        boxes_to_save = [box for box in boxes_to_save if box.getClassId() == "bean"]
        [box.setImageName("im_{}.jpg".format(i)) for box in boxes_to_save]
        boxes_to_save = BoundingBoxes(boxes_to_save)

        boxes_to_keep = nms(boxes_to_save)

        save_bboxes_to_txt(boxes_to_keep, annot_dir)
        draw_boxes_bboxes(image, boxes_to_keep, draw_dir)


def double_detector(image, yolo_1, yolo_2):
    # Unwrap models
    model_1 = yolo_1["model"]
    cfg_1 = yolo_1["cfg"]
    obj_1 = yolo_1["obj"]

    model_2 = yolo_2["model"]
    cfg_2 = yolo_2["cfg"]
    obj_2 = yolo_2["obj"]

    global altNames
    global net_2, meta_2

    # Instantiate second network if not
    if net_2 is None:
        net_2 = load_net_custom(cfg_2.encode("ascii"), model_2.encode("ascii"), 0, 1)  # batch size = 1
    if meta_2 is None:
        meta_2 = load_meta(obj_2.encode("ascii"))

    # Read image and size
    img = cv.imread(image)
    (height, width) = img.shape[0:2]

    # Perform plant detection on image
    altNames = ['maize', 'bean', 'leek', 'stem_maize', 'stem_bean', 'stem_leek']
    plant_detections = performDetect(image, thresh=0.25, configPath=cfg_1, weightPath=model_1, metaPath=obj_1, showImage=False)

    print(image)

    # Open an annotation file
    annotation = os.path.splitext(image)[0] + ".txt"
    with open(annotation, "w") as f:
        # Loop through plants
        for plant_det in plant_detections:
            (label, confidence, box) = plant_det

            if "stem" in label: continue

            # Save raw plant annotation
            (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])
            f.write("{} {} {} {} {} {}\n".format(label, confidence, xmin, ymin, xmax, ymax))

            # Extract patch for this plant
            (x, y, w, h) = clip_box_to_size(box, (width, height))
            (x, y, w, h) = (int(x), int(y), int(w), int(h))
            print(label, confidence, x, y, w, h)

            patch = cv.getRectSubPix(img, (w, h), (x, y))
            patch = cv.resize(patch, (832, 832))

            # Detect stems on this plant
            altNames = ['stem_maize', 'stem_bean', 'stem_leek']
            stem_detections = detect_image(net_2, meta_2, array_to_image(patch)[0], thresh=0.10)

            # Loop through stems
            for stem_det in stem_detections:
                (stem_label, stem_confidence, stem_box) = stem_det
                (xs, ys, ws, hs) = stem_box
                print(" |", stem_label, stem_confidence, xs, ys, ws, hs)

                # Compute coordinates in original image
                new_x = x + (xs / 832 - 0.5) * w
                new_y = y + (ys / 832 - 0.5) * h
                new_w = ws / 832 * w
                new_h = hs / 832 * h

                # Save annotation
                (xmin_s, ymin_s, xmax_s, ymax_s) = xywh_to_xyx2y2(new_x, new_y, new_w, new_h)
                f.write("{} {} {} {} {} {}\n".format(stem_label, stem_confidence, xmin_s, ymin_s, xmax_s, ymax_s))
                print("  *", new_x, new_y, new_w, new_h)

    # Perform NMS to avoid redundancy and save the result
    boxes = read_detection_txt_file(annotation, (width, height))
    boxes = nms(boxes, nms_thresh=0.4)
    save_bboxes_to_txt(boxes, os.path.split(annotation)[0])

    # Save annoted images for visualization
    draw_boxes(image, annotation, "save/double-det/")

# Global variable for the secondary network
net_2 = None
meta_2 = None
def double_detector_folder(folder, yolo_1, yolo_2):
    images = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == ".jpg"]

    # Loop through images
    for image in images:
        double_detector(image, yolo_1, yolo_2)

def _test_optical_flow(folder):
    """
    Private test function.
    """
    images = files_with_extension(folder, ".jpg")
    images.sort(key=os.path.getmtime)

    opt_flow = None
    first_image = cv.imread(images[0])

    for image in images[1:]:
        second_image = cv.imread(image)
        opt_flow, first_image = optical_flow(first_image, second_image, opt_flow)
        egi = egi_mask(second_image)
        dx, dy = mean_opt_flow(opt_flow, egi)

        print("Dx: {}, Dy: {}".format(dx, dy))


def detect_and_track_aggr(network, txt_file, optical_flow, label, conf_thresh=0.25, min_points=8, dist_thresh=7.5/100):
    """
    Tracking by aggregation.
    """
    opt_flows = read_optical_flow(optical_flow)
    images = read_image_txt_file(txt_file)
    (cfg, model, obj) = network.get_cfg_weight_meta()

    # Tracker
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_points, dist_thresh=dist_thresh)

    for i, image in enumerate(images):  # [:300]
        detections = performDetect(image, thresh=conf_thresh,
            configPath=cfg, weightPath=model, metaPath=obj, showImage=False)
        boxes = Parser.parse_yolo_darknet_detections(detections, image,
            image_size(image), [label])
        tracker.update(boxes, opt_flows[i])

    tracker.print_stats_for_tracks(tracker.get_filtered_tracks())

    return tracker

def detect_and_track_aggr_2(boxes, txt_file, optical_flow,
    conf_thresh, min_points, dist_thresh, verbose=False
):
    """
    Tracking by aggregation without recomputing the detections with
    Darknet.
    """
    images = read_image_txt_file(txt_file)
    opt_flows = OpticalFlow.read(optical_flow)
    # opt_flows = OpticalFlow.read_planes("data/opt_flow_haricot_plane.txt")

    # Tracker
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_points, dist_thresh=dist_thresh)
    iterator = tenumerate(images, desc="Tracking", unit="image", leave=False) if verbose else enumerate(images)
    for (i, image) in iterator:
        image_boxes = boxes.getBoundingBoxesByImageName(image)
        tracker.update(image_boxes, opt_flows[i])

    return tracker

def detect_and_track_aggr_visu(network, txt_file, optical_flow, label, conf_thresh=0.25, min_points=8, dist_thresh=7.5/100):
    # Opt flow stuff reading
    opt_flows = OpticalFlow.read(optical_flow)
    acc_flow = np.cumsum(opt_flows, axis=0)
    images = read_image_txt_file(txt_file)
    (cfg, model, obj) = network.get_cfg_weight_meta()

    # Tracker
    tracker = Tracker(min_confidence=conf_thresh, dist_thresh=dist_thresh, min_points=min_points)

    cmap = plt.get_cmap("plasma")
    create_dir("save/aggr_tracking_visu/")

    for i, image in enumerate(images[:1_000]):
        detections = performDetect(image, thresh=conf_thresh,
            configPath=cfg, weightPath=model, metaPath=obj, showImage=False)
        boxes = Parser.parse_yolo_darknet_detections(detections, image,
            image_size(image), [label])
        tracker.update(boxes, opt_flows[i])

        img = cv.imread(image)
        img_save = os.path.join("save/aggr_tracking_visu/", os.path.basename(image))

        for track in tracker.tracks:
            nb_hits = len(track)
            box = track.barycenter_box()
            label = box.getClassId()
            box.moveBy(dx=acc_flow[i, 0], dy=acc_flow[i, 1])
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)

            confidence = track.mean_confidence()
            color = nb_hits / 10 * 255

            cv.circle(img,
                center=(int(x), int(y)),
                radius=5,
                color=color,
                thickness=cv.FILLED)

            cv.putText(img,
                text=f"{nb_hits}",
                org=(int(x + 5), int(y - 10)),
                fontFace=cv.FONT_HERSHEY_PLAIN,
                fontScale=1, color=color, thickness=2)

            cv.putText(img,
                text="{:.2}".format(confidence),
                org=(int(x + 5), int(y + 5)),
                fontFace=cv.FONT_HERSHEY_PLAIN,
                fontScale=1, color=color, thickness=2)

        cv.imwrite(img_save, img)

def point_cloud_visu(yolo, txt_file, opt_flow, label, conf_thresh=0.25):
    # Stuff reading
    (cfg, weights, obj) = yolo.get_cfg_weight_meta()
    images = read_image_txt_file(txt_file)
    opt_flows = OpticalFlow.read(opt_flow)
    acc_flow = np.cumsum(opt_flows, axis=0)
    save_dir = "save/point_cloud_visu/"
    create_dir(save_dir)

    all_boxes = BoundingBoxes()

    for (i, image) in enumerate(images[:1_000]):
        (dx, dy) = acc_flow[i, :]
        detections = performDetect(image, thresh=conf_thresh,
            configPath=cfg, weightPath=weights, metaPath=obj, showImage=False)

        boxes = Parser.parse_yolo_darknet_detections(detections, image,
            image_size(image), [label])

        all_boxes += boxes.movedBy(-dx, -dy)

    all_boxes.erase_image_names()
    all_boxes = associate_boxes_with_image(txt_file, opt_flow, all_boxes)

    for image_name in all_boxes.getNames():
        boxes = all_boxes.getBoundingBoxesByImageName(image_name)
        img = cv.imread(image_name)
        save_img_name = os.path.join(save_dir, os.path.basename(image_name))

        for box in boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)

            cv.circle(img,
                center=(int(x), int(y)),
                radius=2,
                color=(0, 0, 255),
                thickness=cv.FILLED)

        cv.imwrite(save_img_name, img)


def track_aggr(boxes, image_list, optflow_file, image_names, conf_thresh, min_points, dist_thresh, crop_percent=5/100):
    images = read_image_txt_file(image_list)
    opt_flows = OpticalFlow.read(optflow_file)
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_points, dist_thresh=dist_thresh)

    # Python dict are ordered but safer to iterate over image_names as key than dict.items()
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    for (i, image_name) in tenumerate(images, desc="Tracking", unit="image", leave=False):
        tracker.update(boxes_by_name[image_name], opt_flows[i])

    dets = tracker.get_filtered_boxes()
    dets = associate_boxes_with_image(image_list, optflow_file, dets)
    dets = BoundingBoxes([det for det in dets
        if det.getImageName() in image_names
        and det.centerIsIn((crop_percent, crop_percent, 1.0 - crop_percent, 1.0 - crop_percent), as_percent=True)
    ])
    return dets, tracker


if __name__ == "__main__":
    train_path = "data/train/"
    val_path = "data/val/"
    test_path = "/media/deepwater/DATA/Shared/Louis/datasets/test_set/"

    bean_folder = "/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2"
    bean_img_list = "data/haricot_debug_long_2.txt"
    bean_optflow = "data/opt_flow_haricot_better.txt"

    maize_folder = "/media/deepwater/DATA/Shared/Louis/datasets/mais_debug_montoldre_2"
    maize_img_list = "data/mais_debug_long_2.txt"
    maize_optflow = "data/opt_flow_mais_better.txt"

    bean_2_folder = "/media/deepwater/DATA/Shared/Louis/datasets/tache_detection/haricot"
    bean_2_img_list = os.path.join(bean_2_folder, "image_list.txt")
    bean_2_optflow = os.path.join(bean_2_folder, "optical_flow.txt")

    maize_2_folder = "/media/deepwater/DATA/Shared/Louis/datasets/tache_detection/maize"
    maize_2_img_list = os.path.join(maize_2_folder, "image_list.txt")
    maize_2_optflow = os.path.join(maize_2_folder, "optical_flow.txt")

    # YOLO V3
    # model_path = "results/yolov3-tiny_3l_14"  # BDD 4.2
    # model_path = "results/yolo_v3_tiny_pan3_7/"  # BDD 4.2
    # model_path = "results/yolo_v3_pan_csr50_optimal_2"  # BDD 4.2
    # model_path = "results/yolo_v3_pan_csr50_optimal_3"  # BDD 6.0
    # model_path = "results/yolo_v3_tiny_pan3_8/"  # BDD 6.0

    # YOLO V4
    # model_path = "results/yolov4-tiny_1"  # BDD 4.2
    # model_path = "results/yolov4_3"  # BDD 4.2
    # model_path = "results/yolov4-tiny_3l_2"  # BDD 4.2
    # model_path = "results/yolov4-tiny_4"  # BDD 6.0
    # model_path = "results/yolov4-tiny_6"  # BDD 4.2, norm stem
    # model_path = "results/yolov4_4"  # BDD 4.2 norm stem 7.5
    # model_path = "results/yolov4-tiny_7"  # BDD 6.1 norm stems
    # model_path = "results/yolov4-tiny_8"  # BDD 7.0 norm stems
    # model_path = "results/yolov4_5"  # BDD 7.0 norm stems
    # model_path = "results/yolov4-tiny_9"  # BDD 8.0 norm stems
    # model_path = "results/yolov4_6/"  # BDD 8.0 norm stems

    # model_path = "results/yolov4-tiny_stem_classif_2"

    # model_path = "results/yolov4_7/"  # BDD 8.2 norm stems
    # model_path = "results/yolov4-tiny_10/"  # BDD 8.2 norm stems
    # model_path = "results/yolov3-tiny_3l_15/"  # BDD 8.2 norm stems
    model_path = "results/yolov4-tiny_11/"  # BDD 9.0 norm stems
    # model_path = "results/yolov4-tiny_12/"  # BDD 9.0 norm stems 5.0
    # model_path = "results/yolov4-tiny_13/"  # BDD 9.0 norm stems 2.5

    # model_path = "results/yolov4-tiny_stem/"

    yolo = YoloModelPath(model_path)

    # video_path = "/media/deepwater/Elements/Louis/2019-07-25_larrere_videos/demo_tele_4K.mp4"
    # video_param = {"skip_frames": 5, "down_scale": 2, "gray_scale": False, "ratio": 4/3}

    consort = 'Bipbip'
    save_dir = 'save/'
    labels_to_names = ['maize', 'bean', 'leek', 'stem_maize', 'stem_bean', 'stem_leek']
    label_to_number = {'maize': 0, 'bean': 1, 'leek': 2, 'stem_maize': 3, 'stem_bean': 4, 'stem_leek': 5}
    number_to_label = {0: "maize", 1: "bean", 2: "leek", 3: "stem_maize", 4: "stem_bean", 5: "stem_leek"}
    # number_to_label = {0: "stem_maize", 1: "stem_bean"}
    fr_to_en = {"mais": "maize", "haricot": "bean", "poireau": "leek", "mais_tige": "stem_maize", "haricot_tige": "stem_bean", "poireau_tige": "stem_leek"}
    en_to_fr = {"maize": "mais", "bean": "haricot", "leek": "poireau", "stem_maize": "mais_tige", "stem_bean": "haricot_tige", "stem_leek": "poireau_tige"}

    # STEM CLASSIFICATION
    # dets = performDetectOnFolder(yolo, test_path, conf_thresh=25/100, n_proc=1)
    # dets.drawAll(save_dir="stem_classif")

    # COMPUTE MAP
    detections = performDetectOnFolder(yolo, val_path, conf_thresh=25/100)
    gts = Parser.parse_yolo_gt_folder(val_path)
    gts.mapLabels(number_to_label)
    Evaluator().printF1ByClass(detections + gts, threshold=5/100, method=EvaluationMethod.Distance)

    # SAVE DARKNET DETECTION TO FILES
    # folders = [os.path.join(f"/media/deepwater/DATA/Shared/Louis/datasets/training_set/2021-03-29_larrere/row_{i+1}") for i in range(4)]

    # for folder in folders:
    #     boxes = performDetectOnFolder(yolo, folder, conf_thresh=25/100)
    #     boxes.mapLabels(en_to_fr)
    #     boxes = BoundingBoxes([box for box in boxes if box.getClassId() in {"poireau", "poireau_tige"}])
    #     boxes.save_xml()

    # FILTERING EVALUATION
    # label = "bean"
    # stem_label = "stem_maize" if label == "maize" else "stem_bean"
    # min_points = 10 if label == "maize" else 13
    # dist_thresh = (12 if label == "maize" else 6) / 100
    # folder = maize_folder if label == "maize" else bean_folder
    # folder_2 = maize_2_folder if label == "maize" else bean_2_folder
    # dets_folder = "save/stem_maize_aggr/" if label == "maize" else "save/stem_bean_aggr/"
    # dets_folder_2 = "save/stem_maize_2_aggr/" if label == "maize" else "save/stem_bean_2_aggr/"
    # # dets_folder = f"/home/deepwater/github/ObjectStructureDetector/experiments/{label}_debug_montoldre_2"
    # # dets_folder_2 = f"/home/deepwater/github/ObjectStructureDetector/experiments/{label}_2"
    # # dets_folder = f"save/yolo_stem_{label}_1"
    # # dets_folder_2 = f"save/yolo_stem_{label}_2"
    # img_list = maize_img_list if label == "maize" else bean_img_list
    # img_list_2 = maize_2_img_list if label == "maize" else bean_2_img_list
    # opt_flow = maize_optflow if label == "maize" else bean_optflow
    # opt_flow_2 = maize_2_optflow if label == "maize" else bean_2_optflow

    # gts1 = Parser.parse_xml_folder(folder, [en_to_fr[stem_label]])
    # gts1.mapLabels(fr_to_en)
    # image_names_1 = gts1.getNames()
    # gts2 = Parser.parse_json_folder(folder_2, classes={stem_label})
    # image_names_2 = gts2.getNames()
    # gts = gts1 + gts2
    # boxes = Parser.parse_yolo_det_folder(dets_folder,
    #     img_folder=folder,
    #     classes=[stem_label])
    # boxes += Parser.parse_yolo_det_folder(dets_folder_2,
    #     img_folder=folder_2,
    #     classes=[stem_label])
    # # boxes.mapLabels({label: stem_label})
    # dets, tracker = track_aggr(boxes, img_list, opt_flow, gts.getNames(),
    #     conf_thresh=25/100, min_points=min_points, dist_thresh=dist_thresh)
    # dets2, tracker2 = track_aggr(boxes, img_list_2, opt_flow_2, gts.getNames(),
    #     conf_thresh=25/100, min_points=min_points, dist_thresh=dist_thresh)
    # # tracks = associate_tracks_with_image(maize_img_list, maize_optflow, tracker)
    # # tracks = {k: v for (k, v) in tracks.items() if k in image_names_1}
    # # draw_tracked_confidence_ellipse(tracks, "save/ellipse_paper_maize_1/")
    # # tracks = associate_tracks_with_image(maize_2_img_list, maize_2_optflow, tracker2)
    # # tracks = {k: v for (k, v) in tracks.items() if k in image_names_2}
    # # draw_tracked_confidence_ellipse(tracks, "save/ellipse_paper_maize_2/")
    # Evaluator().printAPsByClass((gts + dets + dets2),
    #     thresh=5/100,
    #     method=EvaluationMethod.Distance)
    # # dets.drawAllCenters(save_dir="save/unflawed_maize_database/")
    # # gts.drawAllCenters("save/a_2")

    # WITHOUT FILTERING EVALUATION
    # gts = Parser.parse_xml_folder(folder, [en_to_fr[stem_label]])
    # gts.mapLabels(fr_to_en)
    # gts += Parser.parse_json_folder(folder_2, classes={stem_label})
    # boxes = Parser.parse_yolo_det_folder(dets_folder,
    #     img_folder=folder,
    #     classes=[stem_label])
    # boxes += Parser.parse_yolo_det_folder(dets_folder_2,
    #     img_folder=folder_2,
    #     classes=[stem_label])
    # # boxes.mapLabels({label: stem_label})
    # image_names = gts.getNames()
    # dets = BoundingBoxes([det for det in boxes
    #     if det.getImageName() in image_names
    #     and det.centerIsIn((0.05, 0.05, 0.95, 0.95), as_percent=True)
    # ])
    # Evaluator().printAPsByClass((dets + gts),
    #     thresh=5/100,
    #     method=EvaluationMethod.Distance)
    # # dets.drawAllCenters("save/tmp2")

    # GRID SEARCH TEST
    # gts = Parser.parse_xml_folder(bean_folder, [en_to_fr["stem_bean"]])
    # gts.mapLabels(fr_to_en)
    # gts += Parser.parse_json_folder(bean_2_folder, classes={"stem_bean"})
    # boxes = Parser.parse_yolo_det_folder("save/stem_bean_aggr/",
    #     img_folder=bean_folder,
    #     classes=["stem_bean"])
    # boxes += Parser.parse_yolo_det_folder("save/stem_bean_2_aggr/",
    #     img_folder=bean_2_folder,
    #     classes=["stem_bean"])
    # out = []
    # r1, r2 = range(3, 21, 3), range(1, 21)
    # pbar = tqdm(desc="Grid Search", total=len(r1)*len(r2))
    # for min_dist in r1:
    #     min_dist /= 100
    #     for min_points in r2:
    #         dets, _ = track_aggr(boxes, bean_img_list, bean_optflow, gts.getNames(),
    #             conf_thresh=25/100, min_points=min_points, dist_thresh=min_dist)
    #         dets2, _ = track_aggr(boxes, bean_2_img_list, bean_2_optflow, gts.getNames(),
    #             conf_thresh=25/100, min_points=min_points, dist_thresh=min_dist)
    #         result = Evaluator().GetPascalVOCMetrics(
    #             boxes=(gts + dets + dets2),
    #             thresh=5/100,
    #             method=EvaluationMethod.Distance)["stem_bean"]
    #         (TP, FP, accuracies) = (result["total TP"], result["total FP"], result["accuracies"])
    #         nb_tp = result["total TP"]
    #         string = f"{min_points}, {min_dist}, {TP}, {FP}, {sum(accuracies) / nb_tp}"
    #         pbar.update()
    
    #         out.append(string)
    # out = "\n".join(out)
    
    # with open("save/_aggr_grid_search_bean.csv", "w") as f:
    #     f.write(out)

    # DRAW CONSTELLATIONS
    # boxes = Parser.parse_json_folder(bean_2_folder, classes={"stem_bean"})
    # drawConstellationFlatBoxes(bean_2_img_list, boxes, bean_2_folder, bean_2_optflow, "stem_bean")

    # # boxes_std = performDetectOnFolder(yolo, maize_long_folder, conf_thresh=0.25)
    # # boxes_std = BoundingBoxes([box for box in boxes_std if box.getImageName() in gts.getNames() and box.getClassId() == "stem_maize"])
    # boxes = BoundingBoxes([box for box in boxes if box.getImageName() in gts.getNames() and box.getClassId() == "stem_maize"])
    # Evaluator().printAPsByClass((boxes + gts), thresh=5/100, method=EvaluationMethod.Distance)
    # # boxes_std.drawAllCenters(save_dir="save/not_aggr/")

    # plt.figure()
    # x_values = []
    # y_values = []
    # mean_confidences = []
    #
    # for box in boxes:
    #     (x, y, _, _) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
    #     x_values.append(x)
    #     y_values.append(y)
    #     mean_confidences.append(box.getConfidence())
    #
    # plt.scatter(x_values, y_values, c="red")
    #
    # gts = gts_in_unique_ref(maize_demo, maize_demo_folder, maize_demo_opt_flow, "stem_maize")
    # x_values = []
    # y_values = []
    # for box in gts:
    #     (x, y, _, _) = box.getAbsoluteBoundingBox(format=BBFormat.XYC)
    #     x_values.append(x)
    #     y_values.append(y)
    #
    # plt.scatter(x_values, y_values, marker="+", c="green")
    #
    # plt.ylim([1000, -200])
    # plt.show()

    # tracks = BoundingBoxes([
    #     BoundingBox(imageName="image1", classId="label", x=100, y=200, w=20, h=10, format=BBFormat.XYC),
    #     BoundingBox(imageName="image1", classId="label", x=400, y=600, w=20, h=10, format=BBFormat.XYC),
    # ])

    # detections = BoundingBoxes([
    #     BoundingBox(imageName="image1", classId="label", x=100, y=200, w=20, h=10, format=BBFormat.XYC),
    #     BoundingBox(imageName="image1", classId="label", x=400, y=600, w=20, h=10, format=BBFormat.XYC),
    # ])
    #bean
    # matches, undets, untracks = associate_detections_to_tracks(detections, tracks, min_distance=100)
    # print(matches)
    # print(undets)
    # print(untracks)

    # _test_optical_flow(image_path)

    # Untracked
    # folder = "/media/deepwater/DATA/Shared/Louis/datasets/" + "haricot_debug_montoldre_2"
    # folder_txt = "data/haricot_debug_long_2.txt"
    # gts = Parser.parse_xml_folder(folder)
    # gts.mapLabels(fr_to_en)
    # gts.stats()
    # dets = performDetectOnFolder(yolo, folder, 0.5)
    # dets = BoundingBoxes([det for det in dets if det.getImageName() in gts.getNames()])
    # Evaluator().printAPs(gts + dets)
    # Evaluator().printAPsByClass(gts + dets)
    # Evaluator().printAPsByClass(gts + dets, 3.7 / 100, EvaluationMethod.Distance)
    # dets.drawAll(save_dir="save/maize_debug_long_untracked/")

    # Tracked
    # folder = "/media/deepwater/DATA/Shared/Louis/datasets/" + "haricot_debug_montoldre_2"
    # folder_txt = "data/haricot_debug_long_2.txt"
    # gts = Parser.parse_xml_folder(folder)
    # gts.mapLabels(fr_to_en)
    # gts.stats()
    # dets, opt_flows = performDetectOnFolderAndTrack(yolo, folder_txt, 0.5, max_age=10, min_hits=1)
    # filtered_dets = dets
    # # filtered_dets = BoundingBoxes()
    # # image_names = dets.getNames()
    # # for i, name in enumerate(image_names[:-2]):
    # #     current_image_boxes = dets.getBoundingBoxesByImageName(name)
    # #     next_image_boxes = dets.getBoundingBoxesByImageName(image_names[i+1])
    # #     opt_flow = opt_flows[i+1]
    # #
    # #     next_image_boxes = next_image_boxes.copy()
    # #     [box.setImageName(name) for box in next_image_boxes]
    # #     next_image_boxes = next_image_boxes.movedBy(-opt_flow[0], -opt_flow[1])
    # #     filtered_boxes = nms(bboxes=(current_image_boxes + next_image_boxes), nms_thresh=0.5)
    # #     filtered_dets += filtered_boxes.cliped() # Should add centerIn...
    # #
    # # filtered_dets += dets.getBoundingBoxesByImageName(image_names[-1])
    #
    # filtered_dets = BoundingBoxes([det for det in filtered_dets if det.getImageName() in gts.getNames()])
    # Evaluator().printAPs(gts + filtered_dets)
    # Evaluator().printAPsByClass(gts + filtered_dets)
    # Evaluator().printAPsByClass(gts + filtered_dets, 3.7 / 100, EvaluationMethod.Distance)
    # filtered_dets.drawAll(save_dir="save/bean_debug_long_tracked_KF_aggr_1/")

    # generate_opt_flow("data/haricot_debug_long_2.txt", name="data/opt_flow_last.txt")
    # drawConstellation(maize_demo, nb_samples=100, offset=0)
    # drawConstellationFlat("data/haricot_sequential.txt", "/media/deepwater/DATA/Shared/Louis/datasets/haricot_montoldre_sequential", "data/opt_flow_haricot_sequential.txt")
    # drawConstellationDet(yolo, maize_long_folder, maize_long, maize_opt_flow, "stem_maize")

    # tracker = Sort(max_age=3, min_hits=1)
    # mult = 2
    # for i in range(6):
    #     if i != 2 and i != 3 and i != 4:
    #         dets = np.array([[10 + i * (mult + 1), 10, 20 + i * (mult + 1), 20, 1]])
    #         if i == 5:
    #             dets = np.array([[13, 10, 23, 20]])
    #     else:
    #         dets = np.array([])
    #     print(i)
    #     print("Dets:")
    #     print(dets)
    #     (print("Speed:"))
    #     if i < 2 :
    #         print("{}, {}".format(mult, 0))
    #         tracks = tracker.update(dets, (mult, 0))
    #     else:
    #         print("{}, {}".format(0, 0))
    #         tracks = tracker.update(dets, (0, 0))
    #     print("Tracks:")
    #     print(tracks)
    #     print("Self.Trackers:")
    #     [print(tracker.trackers[i].get_state()) for i in range(len(tracker.trackers))]
    #     print("\n")

    # gts.save(save_dir="save/groundTruths", type_coordinates=CoordinatesType.Absolute, format=BBFormat.XYX2Y2)
    # dets.save(save_dir="save/detections", type_coordinates=CoordinatesType.Absolute, format=BBFormat.XYX2Y2)

    # double_detector_folder("data/double_detector/test/", yolo_1, yolo_2)

    # files = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == ".txt"]
    #
    # new_lines = []
    # for file in files:
    #     with open(file, "r") as f_read:
    #         content = f_read.readlines()
    #         if content != []:
    #             new_lines.append(os.path.basename(file))
    #
    # with open("val_2.txt", "w") as f_write:
    #     for line in new_lines:
    #         line = os.path.splitext(line)[0] + ".jpg"
    #         f_write.write("data/val/" + line + "\n")

    # Create a list of image names to process
    # images = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == ".jpg"]

    # compute_mean_average_precision(
    #     folder=image_path,
    #     model=model_path,
    #     config_file=config_file,
    #     data_obj=meta_path)

    # save_images_from_video(video_path, os.path.join(save_dir, "images_from_video/"), nb_iter=100)
    # save_detect_to_txt(os.path.join(save_dir, "images_from_video/"), save_dir+'result/', model_path, config_file, meta_path)
    # draw_boxes_folder(os.path.join(save_dir, "images_from_video"), os.path.join(save_dir, "result/"), save_path=save_dir)
    # image_vid  = os.path.join(save_dir, "images_from_video")
    # save_path  = os.path.join(save_dir, "save_dir")
    # annot_path = os.path.join(save_dir, "result")

    # filter_detections(video_path, video_param, save_path, yolo_param, k=5)
    # filter_detections_2("/Volumes/KINGSTON/Bipbip_sept19/sm0128_1704")

    # save_detect_to_txt(image_path, save_dir+"detections", model_path, config_file, meta_path, True)
    # save_gt_to_txt(image_path, save_dir+"gts", labels_to_names, True)

    # with open("data/val.txt", "r") as f:
    #     content = f.readlines()
    #
    # content = [item.strip() for item in content]
    # content = [os.path.join("data/img/val", os.path.basename(item)) for item in content]
    #
    # with open("data/val.txt", "w") as f:
    #     for line in content:
    #         f.write(line + "\n")

    # crop_annotation_to_square(image_path, save_dir+'ground-truth', labels_to_names)
    # crop_detection_to_square(image_path, save_dir+'detection-results', model_path, config_file, meta_path)
