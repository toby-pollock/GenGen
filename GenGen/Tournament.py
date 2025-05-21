from PIL.Image import Image
import numpy as np
from copy import deepcopy
from .AbstractIndividual import AbstractIndividual
from .Canvas import Canvas

scored_individual = tuple[AbstractIndividual, float]

class Tournament:
    def __init__(self,
                 base_population: list[AbstractIndividual],
                 target_image: Image,
                 canvas: Canvas,
                 mutation_rate=0.1,
                 elite=True):
        self.base_population = base_population
        self.population = []
        self.target_image = target_image
        self.canvas = canvas
        self.mutation_rate = mutation_rate
        self.elite = elite
        self.survivor_ratio = 0.25
        self.reinitialise()

    def reinitialise(self):
        self.population = []
        for ind in self.base_population:
            for _ in range(ind.replication_factor):
                clone = deepcopy(ind)
                clone.reset_attributes(self.canvas.size)
                self.apply_target_region_color(clone)
                self.population.append(clone)

    def apply_target_region_color(self, individual: AbstractIndividual) -> None:
        x1, y1, x2, y2 = individual.get_transformed_bbox()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.target_image.width, x2), min(self.target_image.height, y2)
        if x2 > x1 and y2 > y1:
            region = self.target_image.crop((x1, y1, x2, y2)).convert("RGB")
            individual.recolor_to_region(region)

    def compute_fitness(self, individual: AbstractIndividual) -> float:
        box = individual.get_transformed_bbox()
        x1, y1, x2, y2 = map(int, box)

        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(self.canvas.size[0], x2)
        y2_clamped = min(self.canvas.size[1], y2)

        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            return -float('inf')  # Completely off-canvas

        region = (x1_clamped, y1_clamped, x2_clamped, y2_clamped)

        canvas_crop = self.canvas.image.crop(region)
        target_crop = self.target_image.crop(region)

        temp_crop = canvas_crop.copy()
        paste_x = x1_clamped - x1
        paste_y = y1_clamped - y1
        try:
            temp_crop.paste(individual.image, (paste_x, paste_y), individual.image.convert("RGBA"))
        except ValueError:
            temp_crop.paste(individual.image, (paste_x, paste_y))

        before = np.asarray(canvas_crop, dtype=np.float32)
        after = np.asarray(temp_crop, dtype=np.float32)
        target = np.asarray(target_crop, dtype=np.float32)

        difference_before = np.abs(before - target)
        difference_after = np.abs(after - target)

        # get sum of differences between before and after
        fitness = np.sum(difference_before - difference_after)
        return fitness


    def evaluate_fitnesses(self):
        return [(ind, self.compute_fitness(ind)) for ind in self.population]

    def select_best(self, scored_population):
        return max(scored_population, key=lambda item: item[1])[0]

    def reproduce(self, parent: AbstractIndividual) -> AbstractIndividual:
        child = deepcopy(parent)
        child.children_count = 0
        child.mutate()
        child.genealogy = parent.genealogy + [parent.children_count]
        parent.children_count += 1
        return child

    def new_generation(self, scored_population):
        scored_population.sort(key=lambda x: x[1], reverse=True)
        survivors = [ind for ind, _ in scored_population[:max(1, int(len(scored_population) * self.survivor_ratio))]]
        new_population = [deepcopy(survivors[0])] if self.elite else []

        for survivor in survivors:
            for _ in range(4):
                child = survivor.reproduce()
                self.apply_target_region_color(child)
                new_population.append(child)

        self.population = new_population

    def step(self) -> AbstractIndividual:
        scored = self.evaluate_fitnesses()
        best = self.select_best(scored)
        self.new_generation(scored)
        return best
