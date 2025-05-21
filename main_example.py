import GenGen as gg

population_custom = [
    gg.CustomImageIndividual(
        image="../ressources/initial_populations/fastfood/burger.png",
        recoloring_method="grayscale_tint",
        replication_factor=32
    ),
    gg.CustomImageIndividual(
        image="ressources/initial_populations/fastfood/fries.png",
        recoloring_method="grayscale_tint",
        replication_factor=32
    ),
]

# === Setup and Run Genetic Generator ===
generator = gg.GeneticImageGenerator(
    population=population_custom,
    target_image_path="ressources/target_images/totoro-xs.jpg",
    generations=7,
    tournament_size=1000,
    output_name="totoro-1000-fastfood",
    output_dir="output",
    enable_display=True,
    save_timelapse=True
)

generator.generate()
