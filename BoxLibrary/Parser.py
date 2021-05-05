from .BoundingBox import BoundingBox
from .BoundingBoxes import BoundingBoxes
from .utils import *
import os
import lxml.etree as ET
import PIL
import json
from pathlib import Path


class Parser:
    
    @staticmethod
    def parse_json_directories(directories, classes=None):
        boxes = BoundingBoxes()
        for directory in directories:
            boxes += Parser.parse_json_folder(directory, classes)
        return boxes

    @staticmethod
    def parse_json_folder(folder, classes=None):
        boxes = BoundingBoxes()
        for file in Path(folder).glob("*.json"):
            boxes += Parser.parse_json_file(file, classes)
        return boxes

    @staticmethod
    def parse_json_file(file, classes=None):
        data = json.load(open(file))
        image_path = str(Path(data["image_path"]).expanduser().resolve())
        (img_w, img_h) = PIL.Image.open(image_path).size
        stem_size = min(img_w, img_h) * 7.5/100
        boxes = BoundingBoxes()

        for obj in data["objects"]:
            label = obj["label"]

            if (classes is None) or (classes and (label in classes)):
                box = obj["box"]
                x_min, y_min, x_max, y_max = float(box["x_min"]), float(box["y_min"]), float(box["x_max"]), float(box["y_max"])
                confidence = obj.get("score", None)
                bb_type = BBType.Detected if confidence else BBType.GroundTruth

                boxes.append(BoundingBox(
                    image_path, str(label),
                    x_min, y_min, x_max, y_max,
                    format=BBFormat.XYX2Y2, imgSize=(img_w, img_h),
                    bbType=bb_type, classConfidence=confidence))

            for part in obj["parts"]:
                part_label = part["kind"]
                kind = f"{part_label}_{label}"

                if (classes is None) or (classes and (kind in classes)):
                    location = part["location"]
                    x, y = float(location["x"]), float(location["y"])
                    confidence = part.get("score", None)
                    bb_type = BBType.Detected if confidence else BBType.GroundTruth

                    boxes.append(BoundingBox(
                        image_path, kind, 
                        x, y, stem_size, stem_size, 
                        format=BBFormat.XYC, imgSize=(img_w, img_h),
                        bbType=bb_type, classConfidence=confidence))

        return boxes

    @staticmethod
    def parse_xml_directories(directories, classes=None):
        boxes = BoundingBoxes()

        for directory in directories:
            boxes += Parser.parse_xml_folder(directory, classes=classes)

        return boxes

    @staticmethod
    def parse_xml_folder(folder, classes=None):
        boxes = BoundingBoxes()

        for file in Path(folder).glob("*.xml"):
            boxes += Parser.parse_xml_file(file, classes=classes)

        return boxes

    @staticmethod
    def parse_xml_file(file, classes=None):
        boxes = BoundingBoxes()
        tree = ET.parse(str(file)).getroot()

        if classes is not None:
            classes = [str(item) for item in classes]

        name = (Path(tree.find('path').text).parent / tree.find('filename').text).expanduser().resolve()
        width = tree.find('size').find('width').text
        height = tree.find('size').find('height').text

        for object in tree.findall('object'):
            class_id = object.find('name').text
            if classes and (class_id not in classes):
                continue

            xmin = float(object.find('bndbox').find('xmin').text)
            ymin = float(object.find('bndbox').find('ymin').text)
            xmax = float(object.find('bndbox').find('xmax').text)
            ymax = float(object.find('bndbox').find('ymax').text)

            box = BoundingBox(str(name), class_id, xmin, ymin, xmax, ymax, format=BBFormat.XYX2Y2, imgSize=(int(width), int(height)))
            boxes.append(box)

        return boxes

    @staticmethod
    def parse_yolo_gt_directories(directories, classes=None):
        boxes = BoundingBoxes()

        for directory in directories:
            boxes += Parser.parse_yolo_gt_folder(directory, classes=classes)

        return boxes

    @staticmethod
    def parse_yolo_gt_folder(folder, classes=None):
        boxes = BoundingBoxes()

        for file in Path(folder).glob("*.txt"):
            boxes += Parser.parse_yolo_gt_file(file, classes)

        return boxes

    @staticmethod
    def parse_yolo_det_folder(folder, img_folder, classes=None):
        boxes = BoundingBoxes()

        for file in Path(folder).glob("*.txt"):
            image_name = os.path.join(img_folder, os.path.basename(os.path.splitext(file)[0] + ".jpg"))
            boxes += Parser.parse_yolo_det_file(file, image_name, classes)

        return boxes

    @staticmethod
    def parse_yolo_gt_file(file, classes=None):
        '''
        Designed to read Yolo annotation files that are in the same folders
        as their corresponding image.
        '''
        boxes = BoundingBoxes()
        image_name = (file.with_suffix(".jpg")).expanduser().resolve()
        img_size = PIL.Image.open(image_name).size

        if classes:
            classes = [str(item) for item in classes]

        content = open(file, "r").readlines()
        content = [line.strip().split() for line in content]

        for det in content:
            (label, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4])

            if classes and label not in classes:
                continue

            box = BoundingBox(imageName=str(image_name), classId=label,x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, format=BBFormat.XYC, imgSize=img_size, bbType=BBType.GroundTruth)

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_yolo_det_file(file, image_name, classes=None):
        boxes = BoundingBoxes()
        image_name = Path(image_name).expanduser().resolve()
        img_size = PIL.Image.open(image_name).size

        if classes:
            classes = [str(item) for item in classes]

        content = open(file, "r").readlines()
        content = [line.strip().split() for line in content]

        for det in content:
            (label, confidence, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])

            if classes and label not in classes:
                continue

            box = BoundingBox(imageName=str(image_name), classId=label, x=x, y=y, w=w, h=h,
            classConfidence=confidence, typeCoordinates=CoordinatesType.Relative,
            format=BBFormat.XYC, imgSize=img_size, bbType=BBType.Detected)

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_yolo_darknet_detections(detections, image_name, img_size=None, classes=None):
        """
        Parses a detection returned by yolo detector wrapper.
        """
        boxes = BoundingBoxes()
        if classes:
            classes = [str(item) for item in classes]
        
        image_name = Path(image_name).expanduser().resolve()

        for detection in detections:
            (label, confidence, box) = detection

            if classes and (label not in classes):
                continue
            
            (x, y, w, h) = box

            box = BoundingBox(imageName=str(image_name), classId=label, x=x, y=y, w=w, h=h, classConfidence=confidence, typeCoordinates=CoordinatesType.Absolute, format=BBFormat.XYC, imgSize=img_size, bbType=BBType.Detected)
            boxes.append(box)

        return boxes

    @staticmethod
    def parse_coco(gt, det, image_path=None):
        (images, categories, img_sizes) = Parser.parse_coco_params(gt)
        gt_boxes = Parser.parse_coco_gt(gt, image_path)
        det_boxes = Parser.parse_coco_det(det, images, categories, img_sizes, image_path)

        return gt_boxes + det_boxes

    @staticmethod
    def parse_coco_gt(gt, img_path=None):
        gt_dict = json.load(open(gt, "r"))

        categories = {item["id"]: item["name"] for item in gt_dict["categories"]}
        images = {item["id"]: item["file_name"] for item in gt_dict["images"]}
        img_sizes = {item["id"]: (item["width"], item["height"]) for item in gt_dict["images"]}

        boxes = BoundingBoxes()

        for annotation in gt_dict["annotations"]:
            img_name = images[annotation["image_id"]]
            if img_path is not None:
                img_name = os.path.join(img_path, img_name)
            label = categories[annotation["category_id"]]
            (x, y, w, h) = annotation["bbox"]
            (width, height) = img_sizes[annotation["image_id"]]

            box = BoundingBox(imageName=img_name, classId=label, x=float(x), y=float(y), w=float(w), h=float(h), imgSize=(int(width), int(height)), bbType=BBType.GroundTruth, format=BBFormat.XYWH)

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_coco_det(det, images, categories, img_sizes=None, img_path=None):
        det_dict = json.load(open(det, "r"))

        boxes = BoundingBoxes()

        for detection in det_dict:
            img_name = images[detection["image_id"]]
            if img_path is not None:
                img_name = os.path.join(img_path, img_name)

            label = categories[detection["category_id"]]
            (x, y, w, h) = detection["bbox"]
            confidence = detection["score"]

            if img_sizes is None:
                img_size = None
            else:
                (width, height) = img_sizes[detection["image_id"]]
                img_size = (int(width), int(height))

            box = BoundingBox(imageName=img_name, classId=label, x=float(x), y=float(y), w=float(w), h=float(h), imgSize=img_size, bbType=BBType.Detected, format=BBFormat.XYWH, classConfidence=float(confidence))

            boxes.append(box)

        return boxes

    @staticmethod
    def parse_coco_params(gt):
        gt_dict = json.load(open(gt, "r"))

        categories = {item["id"]: item["name"] for item in gt_dict["categories"]}
        images = {item["id"]: item["file_name"] for item in gt_dict["images"]}
        img_sizes = {item["id"]: (item["width"], item["height"]) for item in gt_dict["images"]}

        return (images, categories, img_sizes)