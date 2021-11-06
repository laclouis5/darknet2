"""
Python 3 wrapper for identifying objects in images

@author: Philip Kahn
@date: 20180503
"""

from ctypes import *
import random
import os

from pathlib import Path
import cv2


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
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


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


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    x, y, w, h = bbox
    h_w, h_h = w / 2, h / 2
    x_min = x - h_w
    x_max = x + h_w
    y_min = y - h_h
    y_max = y + h_h
    return x_min, y_min, x_max, y_max 

def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)

    return network, class_names, colors


def load_network_2(config_file, name_file, weights, batch_size=1):
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)

    with open(name_file) as f:
        data = f.readlines()
        class_names = [d.strip() for d in data]

    colors = class_colors(class_names)

    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), colors[label], 5)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    output = []
    for j in range(num):
        dets = detections[j]
        for idx, name in enumerate(class_names):
            p = dets.prob[idx]
            if p > 0:
                bbox = dets.bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                output.append((name, p, bbox))
    return output


# def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
#     """
#         Returns a list with highest confidence class and their bbox
#     """
#     pnum = pointer(c_int(0))
#     predict_image(network, image)
#     detections = get_network_boxes(network, image.w, image.h,
#                                    thresh, hier_thresh, None, 0, pnum, 0)
#     num = pnum[0]
#     if nms:
#         do_nms_sort(detections, num, len(class_names), nms)
#     predictions = remove_negatives(detections, class_names, num)
#     predictions = decode_detection(predictions)
#     free_detections(detections, num)
#     return sorted(predictions, key=lambda x: x[1])


# def image_detection(image_path, network, class_names, thresh):
#     width = network_width(network)
#     height = network_height(network)
#     darknet_image = make_image(width, height, 3)

#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_resized = cv2.resize(image_rgb, (width, height),
#                                interpolation=cv2.INTER_LINEAR)

#     copy_image_from_bytes(darknet_image, image_resized.tobytes())
#     detections = detect_image(network, class_names, darknet_image, thresh=thresh)
#     free_image(darknet_image)

#     return detections


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
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
                print("Flag value {} not forcing CPU mode".format(tmp))
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
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL(os.path.join(
        os.environ.get('DARKNET_PATH', './'),
        "libdarknet.so"), RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

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

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

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


## Created by Louis Lac 04-2021 ##


class YoloDetector:

    def __init__(self, cfg, names, weights):
        self.network, self.class_names, _ = load_network_2(
            str(cfg), str(names), str(weights))
        self.width = network_width(self.network)
        self.height = network_height(self.network)

    def predict(self, image, conf_threshold=0.25):
        """
        Parameters:
         - image (np.array): RGB image of shape (H, W, 3). The image will be 
         resized to network size automatically.

        Returns:
         - detections (list): Tuples of (<label>, <conf>, (<x>, <y>, <w>, <h>)) where the coordinates are expressed in the absolute referential of the input image.
        """
        assert 0.0 <= conf_threshold <= 1.0
        width, height = self.width, self.height
        class_names = self.class_names
        network = self.network
        img_h, img_w = image.shape[:2]

        image_resized = cv2.resize(image, (width, height), cv2.INTER_LINEAR)
        darknet_image = make_image(width, height, 3)
        copy_image_from_bytes(darknet_image, image_resized.tobytes())

        pnum = pointer(c_int(0))
        predict_image(network, darknet_image)
        detections = get_network_boxes(
            network, darknet_image.w, darknet_image.h, conf_threshold, .5, None, 0, pnum, 0)
        num = pnum[0]
        do_nms_sort(detections, num, len(class_names), 0.45)
        predictions = remove_negatives(detections, class_names, num)
        predictions.sort(key=lambda x: x[1], reverse=True)

        r_x, r_y = img_w / width, img_h / height
        predictions = [(l, c, (x1 * r_x, y1 * r_y, x2 * r_x, y2 * r_y)) 
            for (l, c, (x1, y1, x2, y2)) in predictions]

        free_detections(detections, num)
        free_image(darknet_image)

        return predictions

    @staticmethod
    def get_cfg_names_weights(model_dir):
        model_dir = Path(model_dir).expanduser().resolve()
        cfg = next(model_dir.glob("*.cfg"))
        weights = next(model_dir.glob("*.weights"))
        names = next(model_dir.glob("*.names"))

        return cfg, names, weights

    @staticmethod
    def from_dir(model_dir):
        cfg, names, weights = YoloDetector.get_cfg_names_weights(model_dir)
        return YoloDetector(cfg, names, weights)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    yolo_path = Path("results/yolov4-tiny_12/")  # BDD 9.0 norm stems 5.0
    net = YoloDetector.from_dir(yolo_path)
    image = str(next(Path("data/val/").glob("*.jpg")))
    img = np.array(Image.open(image))

    detections = net.predict(img, 0.25)
    for d in detections:
        print(d)