from .utils import *
from math import sqrt
import copy
from lxml import etree


class BoundingBox:
    """
    Represents a bounding box with a classId, associated to an image name and with an optional confidence.

    Constructor:
        imageName (str):
            The image name.
        classId (str):
            The class id or label.
        x (float):
            The X upper-left coordinate of the bounding box.
        y (float):
            The Y upper-left coordinate of the bounding box.
        w (float):
            The width bounding box.
        h (float):
            The height bounding box.
        typeCoordinates (optional CoordinatesType):
            Enum (Relative or Absolute) representing if the bounding box coordinates (x,y,w,h) are absolute or relative to size of the image. Default:'Absolute'.
        imgSize (optional tuple(float, flaot)):
            2D vector (width, height) represents the size of the image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
        bbType (optional BBType):
            Enum (Groundtruth or Detection) identifies if the bounding box represents a ground truth or a detection. If it is a detection, the classConfidence has to be informed.
        classConfidence (float):
            The confidence of the detected class. If detectionType is Detection, classConfidence needs to be informed.
        format (optional BBFormat):
            Enum (XYWH, XYX2Y2 or XYC) indicating the format of the coordinates of the bounding boxes. XYWH: <left> <top> <width> <height>, XYX2Y2: <left> <top> <right> <bottom>, XYC: <xCenter> <yCenter> <width> <height>.
    """

    def __init__(self, imageName, classId, x, y, w, h, typeCoordinates=CoordinatesType.Absolute, imgSize=None, bbType=BBType.GroundTruth, classConfidence=None, format=BBFormat.XYWH):
        self._imageName = imageName
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize is None:
            raise IOError(
                "Parameter 'imgSize' is required. It is necessary to inform the image size.")
        if bbType == BBType.Detected and classConfidence is None:
            raise IOError(
                "For BBType.Detection, it is necessary to inform the classConfidence value.")
        # if classConfidence != None and (classConfidence < 0 or classConfidence > 1):
        # raise IOError('classConfidence value must be a real value between 0 and 1. Value: %f' %
        # classConfidence)

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        self._format = format

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (typeCoordinates == CoordinatesType.Relative):
            """!!! Input is (xCenter, yCenter, w, h), rel"""
            if format == BBFormat.XYC:
                (self._x, self._y, self._x2, self._y2) = convertToAbsoluteValues(imgSize, (x, y, w, h))
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
                self._width_img = imgSize[0]
                self._height_img = imgSize[1]
            else:
                raise IOError(
                    "For relative coordinates, the format must be XYWH (x, y, width, height)")
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            if format == BBFormat.XYWH:
                self._x = x
                self._y = y
                self._w = w
                self._h = h
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            elif format == BBFormat.XYX2Y2:  # format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x = x
                self._y = y
                self._x2 = w
                self._y2 = h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            elif format == BBFormat.XYC:
                self._w = w
                self._h = h
                self._x = x - self._w / 2.0
                self._y = y - self._h / 2.0
                self._x2 = x + self._w / 2.0
                self._y2 = y + self._h / 2.0
            else:
                raise IOError(f"BBFormat '{format}' not defined.")

        if imgSize is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = imgSize[0]
            self._height_img = imgSize[1]

    def getAbsoluteBoundingBox(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)
        elif format == BBFormat.XYC:
            x = (self._x + self._x2) / 2.0
            y = (self._y + self._y2) / 2.0
            return (x, y, self._w, self._h)

    def getRelativeBoundingBox(self, imgSize=None):
        """(xCenter, yCenter, w, h)"""
        if imgSize is None and (self._width_img is None or self._height_img) is None:
            raise IOError(
                "Parameter 'imgSize' is required. It is necessary to inform the image size.")
        if imgSize is not None:
            return convertToRelativeValues((imgSize[0], imgSize[1]),
                                           (self._x, self._x2, self._y, self._y2))
        else:
            return convertToRelativeValues((self._width_img, self._height_img),
                                           (self._x, self._x2, self._y, self._y2))

    def getImageName(self):
        return self._imageName

    def getConfidence(self):
        return self._classConfidence

    def getFormat(self):
        return self._format

    def getClassId(self):
        return self._classId

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates

    def getBBType(self):
        return self._bbType

    def setImageName(self, new_name):
        self._imageName = new_name

    def getArea(self):
        area = (self._w + 1) * (self._h + 1)
        return area

    def moveBy(self, dx, dy, typeCoordinates=CoordinatesType.Absolute, imgSize=None):
        """
        Moves the box by the vector (dx, dy). By default coordinates are in absolute values. If using relative coordinates you should specify an image size if not already stored in the BoundingBox object.

        Parameters:
            dx (float):
                The horizontal offset.
            dy (float):
                The vertical offset.
            typeCoordinates (optional CoordinatesType):
                The type of coordinates used. If 'Relative', imgSize should be informed either as a property of BoundingBox or as parameter to this method.
            imgSize (tuple(float, float)):
                The image size: (widht, height).

        Raises:
            IOError: imgSize should be informed when using CoordinatesType.Relative.
        """
        if typeCoordinates == CoordinatesType.Relative \
            and imgSize is None \
            and (self._width_img is None or self._height_img) is None:
            raise IOError("Parameter 'imgSize' is required. It is necessary to inform the image size.")

        if typeCoordinates == CoordinatesType.Relative:
            (img_width, img_height) = imgSize
            dx = img_width * dx
            dy = img_height * dy

        self._x += dx
        self._y += dy
        self._x2 += dx
        self._y2 += dy

    def movedBy(self, dx, dy, typeCoordinates=CoordinatesType.Absolute, imgSize=None):
        """
        <!> DOESN'T WORK, MAYBE BECAUSE OF .COPY <!>
        Moves the box by the vector (dx, dy). By default coordinates are in absolute values. If using relative coordinates you should specify an image size if not already stored in the BoundingBox object.

        Parameters:
            dx (float):
                The horizontal offset.
            dy (float):
                The vertical offset.
            typeCoordinates (optional CoordinatesType):
                The type of coordinates used. If 'Relative', imgSize should be informed either as a property of BoundingBox or as parameter to this method.
            imgSize (tuple(float, float)):
                The image size: (widht, height).

        Raises:
            IOError: imgSize should be informed when using CoordinatesType.Relative

        Returns:
            BoundingBox: The moved box.
        """
        box = self.copy()
        box.moveBy(dx, dy, typeCoordinates, imgSize)
        return box

    def clip(self, size=None):
        """
        Crops the bounding box to fit into a given rectangle. It correponds to the partition that is common with the given rectangle.

        Parameters:
            size (float, float): The width and height in absolute coordinates of the bounding frame. If size is None and an image size is specified in the BoundingBox object, the last in used.
        """
        if (self._width_img is None or self._height_img is None) and size is None:
            raise IOError("Parameter 'size' is required. It is necessary to inform the size.")

        def clippy(box, size):
            (xmin, ymin, xmax, ymax) = box.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
            if xmin < 0:
                xmin = 0
            if xmax >= size[0]:
                xmax = size[0] - 1
            if ymin < 0:
                ymin = 0
            if ymax >= size[1]:
                ymax = size[1] - 1

            box._x = xmin
            box._y = ymin
            box._x2 = xmax
            box._y2 = ymax
            box._w = xmax - xmin
            box._h = ymax - ymin

        if size is not None:
            clippy(self, size)
        else:
            clippy(self, self.getImageSize())

    def cliped(self, size=None):
        """
        Crops the bounding box to fit into a given rectangle and returns it as a new box. It correponds to the partition that is common with the given rectangle.

        Parameters:
            size (float, float): The width and height in absolute coordinates of the bounding frame. If size is None and an image size is specified in the BoundingBox object, the last in used.

        Returns:
            box (BoundingBox): The clipped box.
        """
        box = self.copy()
        box.clip(size)
        return box

    def centerIsIn(self, rect=None, as_percent=False):
        """
        Returns True if the BoundingBox center is in a given rectangle. If no rectangle is provided, the image size stored in the BoundingBox is used, if there is one.

        Parameters:
            rect [float]: The coordinnates of the rectangle with format [xMin, yMin, xMax, yMax].

        Returns:
            bool: Boolean indicading if the BoundingBox center is in the Rectangle.
        """
        (img_w, img_h) = self.getImageSize()
        if (img_w is None or img_h is None) and rect is None:
            raise IOError("Parameter 'rect' is required")

        if rect:
            if as_percent:
                rect = (rect[0]*img_w, rect[1]*img_h, rect[2]*img_w, rect[3]*img_h)
        else:
            rect = (0.0, 0.0, img_w, img_h)

        (x, y, _, _) = self.getAbsoluteBoundingBox(format=BBFormat.XYC)

        return x > rect[0] and x < rect[2] and y > rect[1] and y < rect[3]

    def iou(self, other):
        """
        Returns the Intersection over Union of two boxes.

        Parameters:
            other (BoundingBox): The second box.

        Returns:
            float: The intersection over union
        """
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        if not self.intersects(other): return 0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        intersection = (xB - xA + 1) * (yB - yA + 1)

        areaA = self.getArea()
        areaB = other.getArea()

        union = areaA + areaB - intersection

        return intersection / union

    def intersects(self, other):
        """
        Returns True if the BoundingBox intersects the other BoundingBox.

        Parameters:
            other (BoundingBox): The other box to compare with.

        Returns:
            bool: Boolean indicating if the intersection of the two boxes is not zero.
        """
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        return boxA[0] < boxB[2] and boxB[0] < boxA[2] and boxA[3] > boxB[1]  and boxB[3] > boxA[1]

    def distance(self, other):
        # Change this with format=XYC
        """
        Returns the distance from the center of this BoundingBox to the center of another BoundingBox (Euclidian distance).

        Parameters:
            other (BoundingBox): The other box.

        Returns:
            float: The distance between box centers.
        """
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        cxa = (boxA[2] + boxA[0]) / 2
        cya = (boxA[3] + boxA[1]) / 2
        cxb = (boxB[2] + boxB[0]) / 2
        cyb = (boxB[3] + boxB[1]) / 2

        vx = cxb - cxa
        vy = cyb - cya

        dist = sqrt(pow(vx, 2) + pow(vy, 2))

        return dist

    def setClassId(self, new_class_id):
        self._classId = new_class_id

    def description(self, type_coordinates=None, format=None, save_conf=True):
        """
        Returns a string representing the box: "classId <optional confidence> coord1 coord2 coord3 coord4". The coordinates depends on the chosen format.

        Parameters:
            type_coordinates (optional CoordinatesType):
                The coordinates to use, either Absolute or relative. If not specified, the value stored in the BoundingBox is used.
            format (optional BBFormat):
                The coordinates format to use: XYWH, XYC or XYX2Y2. If not specified, the value stored in the BoundingBox object is used.
        """
        type_coordinates = type_coordinates or self._typeCoordinates
        format = format or self._format
        box = self.getRelativeBoundingBox() if type_coordinates == CoordinatesType.Relative else self.getAbsoluteBoundingBox(format)

        if self._bbType == BBType.Detected and save_conf:
            return f"{self._classId} {self._classConfidence} {box[0]} {box[1]} {box[2]} {box[3]}"

        return f"{self._classId} {box[0]} {box[1]} {box[2]} {box[3]}"

    def xml_repr(self):
        obj = etree.Element("object")
        name = etree.Element("name")
        name.text = f"{self._classId}"
        obj.append(name)

        xmin, ymin, xmax, ymax = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        bbox = etree.Element("bndbox")

        x_min = etree.Element("xmin")
        x_min.text = f"{int(xmin)}"
        bbox.append(x_min)

        y_min = etree.Element("ymin")
        y_min.text = f"{int(ymin)}"
        bbox.append(y_min)

        x_max = etree.Element("xmax")
        x_max.text = f"{int(xmax)}"
        bbox.append(x_max)

        y_max = etree.Element("ymax")
        y_max.text = f"{int(ymax)}"
        bbox.append(y_max)

        obj.append(bbox) 

        return obj

    def addIntoImage(self, image, color=None, thickness=2):
        import cv2

        # Choose color if not specified
        color = color or ((127, 255, 127) if self._bbType == BBType.GroundTruth else (255, 100, 100))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 1

        x1, y1, x2, y2 = self.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Add label
        label = self._classId
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), color,
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
        return image

    def drawCenter(self, save_name=None):
        label = self.getClassId()
        (x, y, _, _) = self.getAbsoluteBoundingBox(BBFormat.XYC)
        image_name = self.getImageName()
        name = os.path.basename(image_name)
        save_name = f"center_{name}" if not save_name else save_name

        image = cv.imread(image_name)
        color = (0, 255, 0) if self.getBBType() == BBType.GroundTruth else (0, 0, 255)
        cv.circle(image, (int(x), int(y)), 5, color, thickness=cv.FILLED)
        cv.imwrite(save_name, image)

    def normalized(self, ratio=7.5/100):
        side_length = min(self._width_img, self._height_img) * ratio
        (x, y, _, _) = self.getAbsoluteBoundingBox(format=BBFormat.XYC)
        box = BoundingBox(
            imageName=self._imageName, classId=self._classId, x=x, y=y, w=side_length, h=side_length, typeCoordinates=CoordinatesType.Absolute, imgSize=(self._width_img, self._height_img), bbType=self._bbType, classConfidence=self._classConfidence, format=BBFormat.XYC
        )
        return box

    def copy(self):
        """
        The behavior of this method is obvious, why are you even reading the doc?
        """
        return copy.deepcopy(self)

    def __repr__(self):
        coords = self.getAbsoluteBoundingBox(BBFormat.XYC)
        description = "{}, {}, xMid: {:.6}, yMid: {:.6}, w: {:.6}, h: {:.6}".format(
            os.path.basename(self._imageName),
            self._classId,
            *coords)

        if self._bbType == BBType.Detected:
            description += " conf.: {:.2%}".format(self._classConfidence)

        return description
