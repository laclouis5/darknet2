# Created by Louis LAC 2019

from lxml.etree import Element, SubElement, tostring, parse
import datetime
import os
from pathlib import Path


class XMLTree:

    def __init__(self, image_path, width, height, user_name="Bipbip", date=datetime.date.today()):
        self.plant_count = 0
        self.image_path = Path(image_path)
        self.tree = Element("GEDI")

        dl_document = SubElement(self.tree, "DL_DOCUMENT")
        user = SubElement(self.tree, "USER")

        user.attrib["name"] = user_name
        user.attrib["date"] = str(date)

        dl_document.attrib["src"] = self.image_path.name
        dl_document.attrib["docTag"] = "xml"
        dl_document.attrib["width"] = str(width)
        dl_document.attrib["height"] = str(height)

    def add_mask(self, name, crop_type="PlanteInteret"):
        dl_document = self.tree.find("DL_DOCUMENT")
        mask = SubElement(dl_document, "MASQUE_ZONE")
        mask.attrib["id"] = str(self.plant_count)
        mask.attrib["type"] = crop_type
        mask.attrib["name"] = str(name)

        self.plant_count += 1

    def save(self, save_dir=""):
        path = Path(save_dir) / self.image_path.with_suffix(".xml").name
        path.write_text(tostring(self.tree, encoding='unicode', pretty_print=True))
