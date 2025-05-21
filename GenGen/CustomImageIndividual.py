from PIL import Image, ImageDraw, ImageStat, ImageEnhance
import random
# ...existing code...
from typing import Optional, Tuple, Union
from copy import deepcopy
from colorsys import rgb_to_hls, hls_to_rgb
import numpy as np

from .AbstractIndividual import AbstractIndividual

class CustomImageIndividual(AbstractIndividual):
    def __init__(self, image: Union[str, Image.Image], recoloring_method="overwrite", **kwargs):
        if isinstance(image, str):
            base_image = Image.open(image).convert("RGBA")
        else:
            base_image = image.copy()

        self.base_image = base_image
        self._image = self.base_image.copy()
        self.scale: float
        self.rotation: float
        self.recoloring_method = recoloring_method
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
        self.rotation = random.uniform(0, 360)

        canvas_area = canvas_size[0] * canvas_size[1]
        min_initial_side = 10  # minimum side length of 10px
        max_initial_area_coverage = 0.05  # 5% of the canvas)
        smallest_side = min(self.base_image.size)
        bigger_side = max(self.base_image.size)
        bigger_side_scaled_down = bigger_side * (min_initial_side / smallest_side)
        min_image_area = int(min_initial_side * bigger_side_scaled_down)
        max_image_area = int(canvas_area * max_initial_area_coverage)
        chosen_area = random.randint(min_image_area, max_image_area)

        self.scale = chosen_area / (self.base_image.width * self.base_image.height)
        self.apply_transformations()

    def apply_transformations(self):
        scaled_width = min(max(10, int(self.base_image.width * self.scale)), 128)
        scaled_height = min(max(10, int(self.base_image.height * self.scale)), 128)
        img = self.base_image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
        img = img.rotate(self.rotation, expand=True, resample=Image.Resampling.BICUBIC)
        self._image = img
        self.position = (
            int(self.center[0] - self._image.width / 2),
            int(self.center[1] - self._image.height / 2)
        )

    def mutate(self):
        self.center = (
            self.center[0] + random.randint(-20, 20),
            self.center[1] + random.randint(-20, 20)
        )
        self.rotation += random.uniform(-60, 60)
        self.scale *= random.uniform(0.6, 1.4)
        self.apply_transformations()

    def reproduce(self):
        child = deepcopy(self)
        child.children_count = 0
        child.genealogy = self.genealogy + [self.children_count]
        self.children_count += 1
        child.mutate()
        return child

    def recolor_to_exact_mean(self, region_img: Image.Image):
        # Step 1: Compute mean color of the target region
        stat = ImageStat.Stat(region_img.convert("RGB"))
        mean_color = np.array(stat.mean, dtype=np.uint8)  # (3,)

        # Step 2: Convert current image to RGBA NumPy array
        img_rgba = self._image.convert("RGBA")
        img_np = np.array(img_rgba)  # Shape: (H, W, 4)

        # Step 3: Replace R, G, B channels where alpha > 0
        mask = img_np[..., 3] > 0  # Alpha channel
        img_np[mask, 0:3] = mean_color  # Apply mean color to RGB

        # Step 4: Convert back to PIL image
        self._image = Image.fromarray(img_np, mode="RGBA")


    def recolor_grayscale_tint(self, region_img: Image.Image, min_impact: float = 0.5):
        """
        Converts image to high-contrast grayscale and tints based on brightness,
        ensuring even dark pixels receive some tint.

        Args:
            region_img: Target region to tint toward.
            contrast_factor: Strength of contrast enhancement.
            min_impact: Minimum tint level for all pixels (0–1).
                        0 = normal, 0.3 = dark pixels get 30% tint, 1 = full tint everywhere.
        """

        # Step 1: Compute target mean color
        stat = ImageStat.Stat(region_img.convert("RGB"))
        mean_color = np.array(stat.mean, dtype=np.float32)  # (3,)

        # Step 2: Prepare grayscale + contrast
        img_rgba = self._image.convert("RGBA")
        _, _, _, a = img_rgba.split()
        gray = img_rgba.convert("L")

        # Step 3: To NumPy
        gray_np = np.asarray(gray, dtype=np.float32) / 255.0  # (H, W)
        alpha_np = np.asarray(a, dtype=np.uint8)[..., None]   # (H, W, 1)

        # Step 4: Adjust brightness so even darks have impact
        effective_brightness = gray_np * (1.0 - min_impact) + min_impact  # (H, W)

        # Step 5: Multiply by mean color
        tinted_rgb = (effective_brightness[..., None] * mean_color).astype(np.uint8)  # (H, W, 3)

        # Step 6: Stack with alpha and return
        rgba_np = np.concatenate([tinted_rgb, alpha_np], axis=2)
        self._image = Image.fromarray(rgba_np, mode="RGBA")


    def recolor_to_region(self, region_img: Image.Image):
        if self.recoloring_method == 'overwrite':
            self.recolor_to_exact_mean(region_img)
        elif self.recoloring_method == 'grayscale_tint':
            self.recolor_grayscale_tint(region_img)
        else:
            raise ValueError(f"Unknown recoloring method: {self.recoloring_method}")

    def __str__(self):
        return (
            f"CustomImageIndividual(name={self.name}, center={self.center}, "
            f"scale={self.scale:.2f}, rotation={self.rotation:.2f}, "
            f"size=({self._image.width}, {self._image.height}), "
            f"genealogy={self.genealogy})"
        )
    
