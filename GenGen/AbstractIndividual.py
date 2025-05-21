from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
import random
from typing import Optional, Tuple
from copy import deepcopy

class AbstractIndividual(ABC):
    def __init__(self, canvas_size: Tuple[int, int] = None, name: Optional[str] = "Unnamed", genealogy=None, replication_factor: int = 1):
        self.name = name
        self.canvas_size = canvas_size
        self.center = (0, 0)
        self.position = (0, 0)
        self.children_count = 0
        self.genealogy = genealogy if genealogy is not None else []
        self.replication_factor = replication_factor


    @abstractmethod
    def get_position(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_canvas_size(self) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def image(self) -> Image.Image:
        pass

    @abstractmethod
    def get_transformed_bbox(self) -> Tuple[int, int, int, int]:
        pass

    @abstractmethod
    def reset_attributes(self, canvas_size: Tuple[int, int]) -> None:
        pass

    @abstractmethod
    def apply_transformations(self) -> None:
        pass

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def reproduce(self):
        pass

    @abstractmethod
    def recolor_to_region(self, region_img: Image.Image):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass