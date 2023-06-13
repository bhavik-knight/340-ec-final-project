import numpy
import imageio
import gari
import pygad
import matplotlib.pyplot

"""
Program :Project.py
Description:This program attempts to regenerate an input image using GA class in pygad Module .
Reference :https://pygad.readthedocs.io/en/latest/
Student Name :Bhavik Bhagat | Greeshma Raju
Student Email:x2020coq@stfx.ca | x2020bgu@stfx.ca
"""

# Reading target image to be reproduced using Genetic Algorithm (GA).
target_im = imageio.imread("input/orange.jpg")
target_im = numpy.asarray(target_im / 255, dtype=numpy.float)

# Target image after enconding. Value encoding is used.
target_chromosome = gari.img2chromosome(target_im)


def fitness_fun(solution, solution_idx):
    """
    Calculating the fitness value for a solution in the population.
    The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes.

    solution: Current solution in the population to calculate its fitness.
    solution_idx: Index of the solution within the population.
    """

    fitness = numpy.sum(numpy.abs(target_chromosome - solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = numpy.sum(target_chromosome) - fitness
    return fitness


def callback(ga_instance):
    print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    if ga_instance.generations_completed % 500 == 0:
        matplotlib.pyplot.imsave(
            "solution_" + str(ga_instance.generations_completed) + ".png",
            gari.chromosome2img(ga_instance.best_solution()[0], target_im.shape),
        )


ga_instance = pygad.GA(
    num_generations=20000,
    num_parents_mating=10,
    fitness_func=fitness_fun,
    sol_per_pop=20,
    num_genes=target_im.size,
    init_range_low=0.0,
    init_range_high=1.0,
    mutation_percent_genes=0.01,
    mutation_type="random",
    mutation_by_replacement=True,
    random_mutation_min_val=0.0,
    random_mutation_max_val=1.0,
    callback_generation=callback,
)

ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_result()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(
    "Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness
    )
)
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print(
        "Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation
        )
    )

result = gari.chromosome2img(solution, target_im.shape)
matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title("PyGAD & GARI for Reproducing Images")
matplotlib.pyplot.show()
