import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .Canvas import Canvas
from .Tournament import Tournament
from copy import deepcopy


class GeneticImageGenerator:
    def __init__(self,
                 population,
                 target_image_path,
                 generations=15,
                 tournament_size=502,
                 enable_display=True,
                 save_timelapse=True,
                 output_name="generated_image",
                 output_dir="./"):
        self.population = population
        self.generations = generations
        self.tournament_size = tournament_size
        self.enable_display = enable_display
        self.save_timelapse = save_timelapse
        self.output_name = output_name
        self.output_dir = output_dir

        self.target_image = Image.open(target_image_path).convert("RGB")
        self.canvas_size = self.target_image.size
        self.target_image = self.target_image.resize(self.canvas_size)
        self.canvas = Canvas(self.canvas_size, self.target_image)

        self.tournament = Tournament(
            base_population=self.population,
            target_image=self.target_image,
            canvas=self.canvas,
        )

        # Create output directory for this run
        self.run_output_dir = os.path.join(output_dir, output_name)
        os.makedirs(self.run_output_dir, exist_ok=True)

        if save_timelapse:
            self.timelapse_dir = os.path.join(self.run_output_dir, "timelapse_frames")
            os.makedirs(self.timelapse_dir, exist_ok=True)

        self.final_image_path = os.path.join(self.run_output_dir, f"{output_name}.png")

    def generate(self):
        if self.enable_display:
            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(np.array(self.canvas.image))
            plt.title("Evolution Progress")
            plt.axis("off")

        for t in range(self.tournament_size):
            print(f"\n=== Tournament {t + 1}/{self.tournament_size} ===")
            best = None
            for _ in range(self.generations):
                best = self.tournament.step()
            best = deepcopy(best)

            fitness = self.tournament.compute_fitness(best)
            print(f"Best individual: {best}\nFitness: {fitness}")

            if fitness > 1:
                self.canvas.apply_individual(best)

                if self.enable_display:
                    image_display.set_data(np.array(self.canvas.image))
                    plt.draw()
                    plt.pause(0.001)

                if self.save_timelapse:
                    frame_path = os.path.join(self.timelapse_dir, f"frame_{t + 1:04d}.png")
                    self.canvas.image.save(frame_path)
            else:
                print("No valid individual found.")
                t -= 1

            self.tournament.reinitialise()

        self.canvas.image.save(self.final_image_path)

        if self.enable_display:
            plt.ioff()
            plt.show()
