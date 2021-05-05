#!/usr/bin/python

import xml.etree.ElementTree as ET
import sys
import os
import os.path
import glob
from pathlib import PurePath


if(len(sys.argv) != 4):
	print("Usage : ")
	print("%s dir-xml-hyp dir-masques dir-img-sources" % sys.argv[0])
	exit(1)

for filename in os.listdir(sys.argv[1]):	
	print("Fichier traité : %s" % filename)
	try:
		tree = ET.parse(sys.argv[1]+"/"+filename)
		root = tree.getroot()

		for child in root:
			if(child.tag == "DL_DOCUMENT"):
				src_img = child.attrib["src"]
				src_img_stem = PurePath(src_img).stem
				print(sys.argv[3]+"/"+src_img_stem+".*")
				if(glob.glob(sys.argv[3]+"/"+src_img_stem+".*") == []):
					print("Erreur : le fichier source ", src_img," indiqué dans le fichier xml hypothèse ", filename," n'est pas présent dans le répertoire", sys.argv[3])				

				for subchild in child:
					if(subchild.tag == "MASQUE_ZONE"):
						id_masque = subchild.attrib["id"]
						file_masque = src_img_stem+"_"+id_masque
						if(glob.glob(sys.argv[2]+"/"+file_masque+".*") == []):
							print("Erreur : le fichier masque ", file_masque," indiqué dans le fichier xml hypothèse ", filename," n'est pas présent dans le répertoire", sys.argv[2])				

	except ET.ParseError as error:
		print("Erreur : fichier xml mal formé")
		print(error)
