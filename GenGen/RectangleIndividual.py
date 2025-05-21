from PIL import Image, ImageDraw, ImageStat
import random
from typing import Optional, Tuple
from copy import deepcopy
import math
import numpy as np


from .AbstractIndividual import AbstractIndividual


class RectangleIndividual(AbstractIndividual):
    def __init__(self, **kwargs):
        self.MIN_SIDE = 4
        self.width: int    
        self.height: int
        self.rotation: float
        self._image = None
        super().__init__(**kwargs)

    def get_position(self):
        return self.position

    def get_canvas_size(self):
        return self.canvas_size

    @property
    def image(self):
        return self._image

    def get_transformed_bbox(self):
        x, y = self.position
        return (x, y, x + self.image.width, y + self.image.height)

    def reset_attributes(self, canvas_size):
        self.canvas_size = canvas_size
        self.center = (random.randint(0, canvas_size[0]), random.randint(0, canvas_size[1]))
        self.rotation = random.uniform(0, 360)
        canvas_area = canvas_size[0] * canvas_size[1]

        # GET WIDTH AND HEIGHT
        def sample_wh(area, r_low=0.5, r_high=2.0):
            mu = np.sqrt(area)

            # Sample a ratio r such that E(r) = E(1/r) = 1 (symmetry)
            r = np.random.uniform(r_low, r_high)
            
            w = r * mu
            h = (1 / r) * mu

            return w, h
        
        min_pourcentage = 0.001
        max_pourcentage = 0.1
        max_surface = max_pourcentage * canvas_area
        min_surface = min_pourcentage * canvas_area
        surface = random.randint(min_surface, max_surface)
        
        
        


        min_initial_side = 10 # minimum side length of 10px
        max_initial_area_coverage = 0.1 # 10% of the canvas
        min_rectangle_area = int(min_initial_side * min_initial_side)
        max_rectangle_area = int(canvas_area * max_initial_area_coverage)
        chosen_area = random.randint(min_rectangle_area, max_rectangle_area)
        float_width, float_height = sample_wh(chosen_area)

        # Rescale if either side is too small. Also convert to int
        def enforce_min_side_preserve_area(w, h, min_side):
            area = w * h
            if w >= min_side and h >= min_side:
                return int(w), int(h)

            # Scale factor to lift the smaller side to min_side
            if w < min_side:
                w = min_side
                h = area / w
            if h < min_side:
                h = min_side
                w = area / h

            return int(round(w)), int(round(h))
        
        self.width, self.height = enforce_min_side_preserve_area(float_width, float_height, self.MIN_SIDE)
        
        self.apply_transformations()

    def apply_transformations(self):
        base = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 255))
        self._image = base.rotate(self.rotation, expand=True)
        self.position = (int(self.center[0] - self._image.width // 2), int(self.center[1] - self._image.height // 2))

    def mutate(self):
        self.rotation += random.uniform(-30, 30)
        self.width = max(self.MIN_SIDE, int(self.width * random.uniform(0.8, 1.2)))
        self.height = max(self.MIN_SIDE, int(self.height * random.uniform(0.8, 1.2)))
        self.center = (self.center[0] + random.randint(-10, 10), self.center[1] + random.randint(-10, 10))
        self.apply_transformations()

    def reproduce(self):
        child = deepcopy(self)
        child.children_count = 0
        child.genealogy = self.genealogy + [self.children_count]
        self.children_count += 1
        child.mutate()
        return child

    def recolor_to_region(self, region_img: Image.Image):
        mean_color = tuple(map(int, ImageStat.Stat(region_img.convert("RGB")).mean))
        base = Image.new("RGBA", (self.width, self.height), (*mean_color, 255))
        self._image = base.rotate(self.rotation, expand=True)

    def __str__(self):
        return f"RectangleIndividual(name={self.name}, size=({self.width}, {self.height}), rotation={self.rotation:.2f}, position={self.position}\nGenealogy={self.genealogy})"
    
