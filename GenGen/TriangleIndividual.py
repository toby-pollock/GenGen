from PIL import Image, ImageDraw, ImageStat
import random
from typing import Optional, Tuple
from copy import deepcopy
import math
import numpy as np

from .AbstractIndividual import AbstractIndividual

class TriangleIndividual(AbstractIndividual):
    def __init__(self, **kwargs):
        self.MAX_INITIAL_AREA_COVERAGE = 0.5
        self.MIN_SIDE = 4
        self.points: tuple[int, int, int]
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
        return (x, y, x + self._image.width, y + self._image.height)

    def reset_attributes(self, canvas_size):
        self.canvas_size = canvas_size
        self.center = (random.randint(0, canvas_size[0]), random.randint(0, canvas_size[1]))
        
        # Reset the points of the triangle
        canvas_area = canvas_size[0] * canvas_size[1]

        # GET WIDTH AND HEIGHT
        def sample_wh(area, r_low=0.5, r_high=2.0):
            mu = np.sqrt(area)

            # Sample a ratio r such that E(r) = E(1/r) = 1 (symmetry)
            r = np.random.uniform(r_low, r_high)
            
            w = r * mu
            h = (1 / r) * mu

            return w, h

        min_initial_side = 10 # minimum side length of 10px
        max_initial_area_coverage = 0.05 # 3% of the canvas
        min_triangle_area = int(math.sqrt(3) * min_initial_side * min_initial_side / 4) # minimum area of an equilateral triangle with side length 10px
        max_triangle_area = int(canvas_area * max_initial_area_coverage)
        chosen_area = random.randint(min_triangle_area, max_triangle_area)
        float_bounding_box_width, float_bounding_box_height = sample_wh(chosen_area)

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


        bounding_box_width, bounding_box_height = enforce_min_side_preserve_area(float_bounding_box_width, float_bounding_box_height, self.MIN_SIDE)
        a1 = (random.randint(0, bounding_box_width), random.randint(0, bounding_box_height))
        b1 = (random.randint(0, bounding_box_width), random.randint(0, bounding_box_height))
        c1 = (random.randint(0, bounding_box_width), random.randint(0, bounding_box_height))
        center = (
            (a1[0] + b1[0] + c1[0]) // 3,
            (a1[1] + b1[1] + c1[1]) // 3
        )
        a2 = (self.center[0] + (a1[0] - center[0]), self.center[1] + (a1[1] - center[1]))
        b2 = (self.center[0] + (b1[0] - center[0]), self.center[1] + (b1[1] - center[1]))
        c2 = (self.center[0] + (c1[0] - center[0]), self.center[1] + (c1[1] - center[1]))
        self.points = (a2, b2, c2)

        self.apply_transformations()

    def apply_transformations(self):
        min_x = min(p[0] for p in self.points)
        max_x = max(p[0] for p in self.points)
        min_y = min(p[1] for p in self.points)
        max_y = max(p[1] for p in self.points)
        width = max_x - min_x
        height = max_y - min_y

        self._image = Image.new("RGBA", (width + 1, height + 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self._image)
        adjusted_points = [(x - min_x, y - min_y) for (x, y) in self.points]
        draw.polygon(adjusted_points, fill=(255, 255, 255, 255))
        if not hasattr(self, 'center') or self.center == (0, 0):
            self.center = (min_x + width // 2, min_y + height // 2)
        self.position = (
            int(self.center[0] - width // 2),
            int(self.center[1] - height // 2)
        )

    def mutate(self):
        self.points = [(x + random.randint(-5, 5), y + random.randint(-5, 5)) for (x, y) in self.points]
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

        # Recreate the blank transparent image
        min_x = min(p[0] for p in self.points)
        max_x = max(p[0] for p in self.points)
        min_y = min(p[1] for p in self.points)
        max_y = max(p[1] for p in self.points)
        width = max_x - min_x
        height = max_y - min_y

        self._image = Image.new("RGBA", (width + 1, height + 1), (0, 0, 0, 0))
        adjusted_points = [(x - min_x, y - min_y) for (x, y) in self.points]

        draw = ImageDraw.Draw(self._image)
        draw.polygon(adjusted_points, fill=(*mean_color, 255))


    def __str__(self):
        return f"TriangleIndividual(name={self.name}, points={self.points}, position={self.position}\nGenealogy={self.genealogy})"
