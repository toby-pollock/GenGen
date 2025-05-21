from PIL import Image, ImageDraw, ImageStat
import random
from typing import Optional, Tuple
from copy import deepcopy
import math
import numpy as np

from .AbstractIndividual import AbstractIndividual


class CircleIndividual(AbstractIndividual):
    def __init__(self, **kwargs):
        self.MAX_INITIAL_AREA_COVERAGE = 0.5
        self.MIN_DIAMETER = 4
        self.diameter: int
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
        return (x, y, x + self.diameter, y + self.diameter)

    def reset_attributes(self, canvas_size):
        self.canvas_size = canvas_size
        self.center = (random.randint(0, canvas_size[0]), random.randint(0, canvas_size[1]))
        canvas_area = canvas_size[0] * canvas_size[1]
        
        min_initial_diameter = 10 # minimum diameter of 10px
        max_initial_area_coverage = 0.05 # 5% of the canvas
        min_circle_area = int(math.pi * min_initial_diameter * min_initial_diameter / 4) # minimum area of a circle with diameter 10px
        max_circle_area = int(canvas_area * max_initial_area_coverage)
        chosen_area = random.randint(min_circle_area, max_circle_area)
        self.diameter = int(math.sqrt(chosen_area / 3.14159) * 2)

        self.apply_transformations()

    def apply_transformations(self):
        self._image = Image.new("RGBA", (self.diameter, self.diameter), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self._image)
        draw.ellipse([0, 0, self.diameter, self.diameter], fill=(255, 255, 255, 255))
        self.position = (int(self.center[0] - self.diameter // 2), int(self.center[1] - self.diameter // 2))

    def mutate(self):
        self.diameter = max(self.MIN_DIAMETER, int(self.diameter * random.uniform(0.8, 1.2)))
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
        draw = ImageDraw.Draw(self._image)
        draw.ellipse([0, 0, self.diameter, self.diameter], fill=(*mean_color, 255))

    def __str__(self):
        return f"CircleIndividual(name={self.name}, diameter={self.diameter}, position={self.position}\nGenealogy={self.genealogy})"
