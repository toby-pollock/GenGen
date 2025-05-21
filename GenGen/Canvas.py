### Canvas.py (refactored)
from PIL import Image, ImageStat
from .Individual import Individual

class Canvas:
    def __init__(self, size, target_image: Image.Image):
        self.size = size
        self.target_image = target_image

        # Compute mean RGB color from the target image
        stat = ImageStat.Stat(target_image.convert("RGB"))
        mean_color = tuple(map(int, stat.mean))

        # self.image = Image.new("RGB", size, mean_color)
        self.image = Image.new("RGB", size, (0, 0, 0))
        self.subimageCounter = 0

    def apply_individual(self, individual: Individual):
        try:
            self.image.paste(individual.image, individual.position, individual.image.convert("RGBA"))
        except ValueError:
            self.image.paste(individual.image, individual.position)
        self.subimageCounter += 1
        print(f"Canvas now has {self.subimageCounter} subimages.")