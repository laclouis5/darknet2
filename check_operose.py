#!/usr/bin/python3

from xml.etree.ElementTree import parse, ParseError
from pathlib import Path
import string
from argparse import ArgumentParser
from tqdm import tqdm
import re

CROP_TYPES = {"Adventice", "PlanteInteret"}
CROP_NAMES = {"mais", "haricot"}
DOC_TAG = "xml"
CONSORTIUMS = {"pead", "roseau", "bipbip", "weedelec"}
INVALID_CHAR = set(string.punctuation)
ID_RE = re.compile(r"([a-z]+)([0-9]+)")


def check(xml_dir: Path, mask_dir: Path, img_dir: Path, mask_ext: str):
	logs = []
	xml_files = list(xml_dir.glob("*.xml"))

	for xml_file in tqdm(xml_files, desc="Validating format", unit="file"):	
		try:
			root = parse(xml_file).getroot().find("DL_DOCUMENT")
			src = root.attrib["src"]
			img_name = Path(src).name
			img_file = img_dir / img_name
			doc_tag = root.attrib["docTag"]
			file_stem = xml_file.stem

			parts = file_stem.split("_")
			if len(parts) != 2:
				logs.append(f"Image stem of xml file '{xml_file}' should have two parts separated with an underscore (consortium and identifier)")
				continue
			
			consortium, identifier = parts
			if consortium not in CONSORTIUMS:
				logs.append(f"Unknown consortium '{consortium}' for xml file '{xml_file}'")

			groups = ID_RE.fullmatch(identifier).groups()
			if len(groups) != 2:
				logs.append(f"Invalid file identifier for file '{xml_file}'")
				continue

			crop_name, _ = groups
			if crop_name not in CROP_NAMES:
				logs.append(f"Invalid file identifier for file '{xml_file}'")

			if img_name != src:
				logs.append(f"Image source '{src}' in XML file '{xml_file.name}' is not a filename")

			if img_file.stem != xml_file.stem:
				logs.append(f"XML file '{xml_file.name}' does not have the same identifier as source '{img_name}'")

			if not img_file.is_file():
				logs.append(f"Image file '{img_name}' for XML file '{xml_file.name}' not found in folder '{img_dir}'")

			if doc_tag != DOC_TAG:
				logs.append(f"Attribute 'docTag' should be '{DOC_TAG}', not '{doc_tag}' for XML file '{xml_file.name}'")

			try:
				img_w, img_h = int(root.attrib["width"]), int(root.attrib["height"])
				if img_w <= 0 or img_h <= 0:
					logs.append(f"Image size should be greater than 0 for XML file '{xml_file.name}'")
			except ValueError as error:
				logs.append(f"Invalid image width or height for XML file '{xml_file.name}'")

			for mask in root.findall("MASQUE_ZONE"):
				try:
					mask_id = int(mask.attrib["id"])
				except ValueError:
					logs.append(f"'mask_id' attribute of XML file '{xml_file.name}' should be an integer, not '{mask.attrib['id']}'")
					continue

				if mask_id < 0:
					logs.append(f"'mask_id' attribute of XML file '{xml_file.name}' should be greater than or equal to 0, not {mask_id}")

				crop_type = mask.attrib["type"]
				if crop_type not in CROP_TYPES:
					logs.append(f"Label '{crop_type}' for mask with id '{mask_id}' is not a valid type for XML file '{xml_file.name}'")

				mask_file = (mask_dir / f"{img_file.stem}_{mask_id}").with_suffix(mask_ext)
				if not mask_file.is_file():
					logs.append(f"Mask file '{mask_file.name}' not found in folder '{mask_dir}'")

		except ParseError as error:
			logs.append(f"Ill-formed XML file '{xml_file.name}': {error}")

		except KeyError as error:
			logs.append(f"Attribute {error} in XML file '{xml_file.name}' not found")

	if not logs:
		print("Validation finished without error.")
	else:
		Path("logs.txt").write_text("\n".join(logs))
		print(f"Validation finished with {len(logs)} error(s). Check 'logs.txt' file for details.")


def parse_args():
	parser = ArgumentParser("Check XML files structure for the Operose Challenge")

	parser.add_argument("xml_dir", type=Path, help="Directory where XML files are stored.")
	parser.add_argument("mask_dir", type=Path, help="Directory where binary masks are stored.")
	parser.add_argument("img_dir", type=Path, help="Directory where images are stored.")
	
	parser.add_argument("--mask_ext", type=str, default=".pgm", help="The file extension of mask files.")

	args = parser.parse_args()

	args.xml_dir = args.xml_dir.expanduser().resolve()
	args.mask_dir = args.mask_dir.expanduser().resolve()
	args.img_dir = args.img_dir.expanduser().resolve()

	assert args.xml_dir.is_dir(), f"Directory '{args.xml_dir}' does not exist"
	assert args.mask_dir.is_dir(), f"Directory '{args.mask_dir}' does not exist"
	assert args.img_dir.is_dir(), f"Directory '{args.img_dir}' does not exist"
	assert args.mask_ext.startswith("."), f"File extension '{args.mask_ext}' should start with a '.'"

	return args


if __name__ == "__main__":
	args = parse_args()
	check(args.xml_dir, args.mask_dir, args.img_dir, args.mask_ext)