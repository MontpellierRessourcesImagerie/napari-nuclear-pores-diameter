import cv2 as cv
import numpy as np
import math
import tifffile
from pathlib import Path
from scipy.ndimage import (gaussian_filter, white_tophat)
from skimage.measure import label, regionprops

class NuclearPoresAnalyzer():
    def __init__(self):
        self.image       = None
        self.pattern     = None
        self.method_name = 'TM_CCOEFF_NORMED'
        self.prominence  = 0.8
        self.convo_map   = None
        self.points      = []
        self.radius      = 10
        self.anisotropy  = 1.0
        self.padding     = 5
        self.sigma       = 2.0

    def get_points(self):
        return self.points
    
    def get_methods(self):
        return {
            'TM_CCOEFF'        : (cv.TM_CCOEFF, 1.0),
            'TM_CCOEFF_NORMED' : (cv.TM_CCOEFF_NORMED, 1.0),
            'TM_CCORR'         : (cv.TM_CCORR, 1.0),
            'TM_CCORR_NORMED'  : (cv.TM_CCORR_NORMED, 1.0),
            'TM_SQDIFF'        : (cv.TM_SQDIFF, 1.0),
            'TM_SQDIFF_NORMED' : (cv.TM_SQDIFF_NORMED, 1.0),
        }

    def get_image_input(self):
        return self.image

    def set_image_input(self, data):
        self.image = data

    def get_pattern_properties(self):
        return (self.radius, self.anisotropy, self.padding, self.sigma)

    def set_pattern_properties(self, radius, anisotropy, padding, sigma):
        self.radius = radius
        self.anisotropy = anisotropy
        self.padding = padding
        self.sigma = sigma
        print(f"Pattern properties: radius={radius}, anisotropy={anisotropy}, padding={padding}, sigma={sigma}")

    def generate_pattern(self):
        r = self.radius
        a = self.anisotropy
        p = self.padding
        h = int(math.ceil(2 * r * a))
        w = int(math.ceil(2 * r * (1/a)))
        s = max(h + 2 * p, w + 2 * p)
        canvas = np.zeros((s, s), dtype=np.float32)
        cv.ellipse(canvas, (s//2, s//2), (w//2, h//2), 0, 0, 360, 1, 1)
        canvas = gaussian_filter(canvas, sigma=self.sigma)
        canvas /= canvas.max()
        self.pattern = canvas

    def get_pattern(self):
        return self.pattern
    
    def set_pattern(self, data):
        self.pattern = data

    def get_prominence(self):
        return self.prominence
    
    def set_prominence(self, value):
        self.prominence = max(value, 0.0)
        print(f"Prominence set to: {value}")

    def get_method_name(self):
        return self.method_name
    
    def set_method_name(self, name):
        if name is None or name not in self.get_methods().keys():
            raise ValueError(f"Method name '{name}' is not valid.")
        self.method_name = name
        print(f"Method set to: {name}")

    def preprocess_image(self, img):
        img = img.astype(np.float32)
        img = white_tophat(img, size=5)
        img -= img.min()
        img /= img.max()
        return img

    def process_points(self):
        if self.image is None or self.pattern is None:
            raise ValueError("Image and pattern must be set before processing points.")
        
        method, coef = self.get_methods().get(self.method_name, (cv.TM_CCOEFF_NORMED, 1.0))
        template = self.pattern
        img = self.preprocess_image(self.image)
        res = cv.matchTemplate(img, template, method)

        self.convo_map = res * coef
        _, max_val, _, _ = cv.minMaxLoc(res)
        threshold = max_val * self.prominence
        mask = (res >= threshold).astype(np.uint8)
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        shift = np.array([self.pattern.shape[0]//2, self.pattern.shape[1]//2])
        self.points = np.array([(float(r.centroid[0]), float(r.centroid[1])) for r in regions]) + shift[::-1]

def export_points_to_csv(points, file_path):
    import csv
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Y', 'X'])
        for point in points:
            writer.writerow(point)

if __name__ == "__main__":
    folder_path = Path("/home/clement/Documents/projects/measures-orestis/2025-10-27/STED-pores-nucleaires")
    image_name  = "0v17b.tif"
    image_path  = folder_path / image_name
    img_base    = tifffile.imread(image_path)

    npa = NuclearPoresAnalyzer()
    npa.set_image_input(img_base)
    npa.set_pattern_properties(radius=3.0, anisotropy=1.0, padding=5, sigma=1.0)
    npa.generate_pattern()
    npa.process_points()
    points = npa.get_points()
    export_points_to_csv(points, folder_path / "detected_nuclear_pores.csv")