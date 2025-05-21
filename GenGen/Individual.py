from PIL import Image, ImageEnhance, ImageStat
import random
from typing import Optional
from colorsys import rgb_to_hls, hls_to_rgb

class Individual:
    def __init__(self, base_image: Image.Image, canvas_size: tuple, name: Optional[str] = "Unnamed", genealogy=None):
        self.name = name
        self.base_image = base_image.copy()
        self.image = self.base_image.copy()
        self.canvas_size = canvas_size

        self.rotation = 0
        self.scale = 1
        self.center = (0, 0)  # This is the canvas-centered coordinate
        self.position = (0, 0)  # This is the top-left corner derived from center
        self.children_count = 0
        self.genealogy = genealogy if genealogy is not None else []

        self.reset_random_attributes(canvas_size)

    def __str__(self):
        # get color of the pixel in the center of the image
        center_pixel = self.image.getpixel((self.image.width // 2, self.image.height // 2))
        # convert to hex
        center_pixel_hex = f"0x{center_pixel[0]:02X}{center_pixel[1]:02X}{center_pixel[2]:02X}"
        return f"Individual(name={self.name}, genealogy={self.genealogy}\nposition={self.position}, rotation={self.rotation}, scale={self.scale}\nwidth={self.image.width}, height={self.image.height})\nRGB: {center_pixel_hex}\n"

    def apply_transformations(self):
        # Apply a clamped scale. Min 0.4 and max 3.0
        # self.scale = max(0.4, min(self.scale, 3.0))
        scaled_width = min(max(10, int(self.base_image.width * self.scale)), 128)
        scaled_height = min(max(10, int(self.base_image.height * self.scale)), 128)

        self.image = self.image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)

        # Apply rotation
        self.image = self.image.rotate(self.rotation, expand=True, resample=Image.Resampling.BICUBIC)


        # Update top-left position to keep image centered at self.center
        self.position = (
            int(self.center[0] - self.image.width / 2),
            int(self.center[1] - self.image.height / 2)
        )

        # Clamp position to ensure partial canvas overlap
        canvas_width, canvas_height = self.canvas_size
        img_w, img_h = self.image.size

        min_x = -img_w + 1
        max_x = canvas_width - 1
        min_y = -img_h + 1
        max_y = canvas_height - 1

        clamped_x = max(min_x, min(self.position[0], max_x))
        clamped_y = max(min_y, min(self.position[1], max_y))

        self.position = (clamped_x, clamped_y)


    def reset_random_attributes(self, canvas_size: tuple):
        self.rotation = random.uniform(0, 360)
        self.scale = random.uniform(0.1, 2.0)

        # Random center point within canvas
        self.center = (
            random.randint(0, canvas_size[0] - 1),
            random.randint(0, canvas_size[1] - 1)
        )

        self.apply_transformations()

    def mutate(self):
        # Jitter center slightly
        self.center = (
            self.center[0] + random.randint(-20, 20),
            self.center[1] + random.randint(-20, 20)
        )

        self.rotation += random.uniform(-60, 60)
        self.scale *= random.uniform(0.6, 1.4)

        self.apply_transformations()

    def get_transformed_bbox(self):
        x, y = self.position
        return (x, y, x + self.image.width, y + self.image.height)


    def match_color_to_region(self, region_img: Image.Image, tint_strength: float = 0.8):
        """
        Tints the image toward the region's average color, scaled by pixel alpha (transparency).
        
        Args:
            region_img: The region of the target image to match.
            tint_strength: A float between 0 (no tint) and 1 (full tint on opaque pixels).
        """
        stat = ImageStat.Stat(region_img)
        mean_color = tuple(map(int, stat.mean))  # (R, G, B)

        img = self.image.convert("RGBA")
        pixels = img.load()
        width, height = img.size

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                alpha_factor = (a / 255.0) * tint_strength
                new_r = int((1 - alpha_factor) * r + alpha_factor * mean_color[0])
                new_g = int((1 - alpha_factor) * g + alpha_factor * mean_color[1])
                new_b = int((1 - alpha_factor) * b + alpha_factor * mean_color[2])
                pixels[x, y] = (new_r, new_g, new_b, a)

        self.image = img




    def recolor_to_exact_mean(self, region_img: Image.Image):
        # Compute mean RGB
        stat = ImageStat.Stat(region_img.convert("RGB"))
        mean_color = tuple(map(int, stat.mean))  # (R, G, B)

        # Ensure image is in RGBA mode
        self.image = self.image.convert("RGBA")

        pixels = self.image.load()
        width, height = self.image.size

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a > 0:
                    pixels[x, y] = (*mean_color, a)

    def match_color_to_region_by_luminance(self, region_img: Image.Image, tint_strength: float = 1.0):
        """
        Tints bright (whiter) pixels more than dark (blacker) ones based on their luminance.
        
        Args:
            region_img: Target region whose average color will be blended in.
            tint_strength: Max tinting strength for pure white pixels (0-1).
        """
        stat = ImageStat.Stat(region_img)
        mean_color = tuple(map(int, stat.mean))  # (R, G, B)

        img = self.image.convert("RGBA")
        pixels = img.load()
        width, height = img.size

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]

                # Compute luminance using the Rec. 709 formula
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                blend_factor = luminance * tint_strength  # Stronger tint for lighter pixels

                new_r = int((1 - blend_factor) * r + blend_factor * mean_color[0])
                new_g = int((1 - blend_factor) * g + blend_factor * mean_color[1])
                new_b = int((1 - blend_factor) * b + blend_factor * mean_color[2])

                pixels[x, y] = (new_r, new_g, new_b, a)

        self.image = img

    def recolor_preserve_luminance(self, region_img: Image.Image):
        """
        Shifts hue toward target_color while preserving luminance and saturation.
        """
        stat = ImageStat.Stat(region_img)
        mean_color = tuple(map(int, stat.mean))
        target_color = tuple(map(int, mean_color))

        img = self.image.convert("RGBA")
        pixels = img.load()
        width, height = img.size

        # Convert target to HLS
        r_t, g_t, b_t = [v / 255.0 for v in target_color]
        h_target, _, _ = rgb_to_hls(r_t, g_t, b_t)

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                h, l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
                r_new, g_new, b_new = hls_to_rgb(h_target, l, s)
                pixels[x, y] = (int(r_new * 255), int(g_new * 255), int(b_new * 255), a)

        self.image = img
    

