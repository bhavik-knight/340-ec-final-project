"""
Program :Project.py
Description:This program attempts to regenerate an input image using GA class in pygad Module .
Reference :https://pygad.readthedocs.io/en/latest/
Student Name :Bhavik Bhagat | Greeshma Raju
Student Email:x2020coq@stfx.ca | x2020bgu@stfx.ca
"""

import numpy as np
import pygad
import numpy
import imageio
import gari
import functools
import operator
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Image we want to generate using GA
    target_im = get_data("flower.jpg")

    # fitness function for our images
    def fitness_fun(solution, solution_idx):
        target_chromosome = get_chromosome(target_im)
        fitness = numpy.sum(numpy.abs(target_chromosome - solution))
        return fitness

    # tunable parameters
    crpb = 0  # crossover prob
    mtpb = 0.01  # mutation prob
    gen = 20  # number of generations
    # pop_size_list = (10, 50, 100)
    # num_parents = (5, 10)
    num_parents = (10, 20, 100)

    # The modifications
    # different kinds of mutation strategies
    # mutations_strategy = ("random", "inversion", "swap", "scram
    mutations_strategy = ("random", "swap", "inversion", "scramble")

    # different kinds of crossover strategies
    # crossover_strategy = ("single_point", "two_points", "uniform")

    # different kinds of selection strategies
    # selection_strategy = ("sss", "rws", "sus", "rank", "random", "tournament")

    parameters = {
        "crossover_prob": crpb,
        "mutation_prob": mtpb,
        "num_generations": gen,
        "mutation": mutations_strategy,
        "parent_num": num_parents,
    }

    def on_generation(ga_instance):
        print("Generation = {gen}".format(gen=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

        if ga_instance.generations_completed % 500 == 0:
            """""
            plt.imsave(
                "solution_" + str(ga_instance.generations_completed) + ".png",
                gari.chromosome2img(ga_instance.best_solution()[0], target_im.shape),
            )
            """ ""

    # counter for total figures generated
    count = 1
    for mutation_type in mutations_strategy:
        for num in num_parents:
            parameters.update({"mutation": mutation_type, "parent_num": num})

            # just print something on screen
            print(f"Iteration: {count}, {parameters}")

            # create an instance of the GA
            ga_instance = pygad.GA(
                fitness_func=fitness_fun,
                num_genes=target_im.size,
                sol_per_pop=20,
                num_parents_mating=num,
                num_generations=gen,
                mutation_type=mutation_type,
                mutation_probability=mtpb,
                init_range_low=0.0,
                init_range_high=1.0,
                mutation_percent_genes=mtpb,
                random_mutation_min_val=0.0,
                random_mutation_max_val=1.0,
                on_generation=on_generation,
            )

            # run GA for the given parameters, and plot fitness
            ga_instance.run()
            ga_instance.plot_fitness(
                title=parameters,
                save_dir=f"output/fitness/{count}.{mutation_type}_{mtpb}.jpg",
            )

            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Index of the best solution : {solution_idx}")
            if ga_instance.best_solution_generation != -1:
                print(
                    f"Best fitness value reached after "
                    f"{ga_instance.best_solution_generation} generations."
                )

            generated_img = chromosome2img(solution, target_im.shape)
            plot_and_save_generated_img(
                generated_img,
                f"{count}.{mutation_type}_{mtpb}",
                solution_fitness,
                parameters,
            )
            count += 1
    return None


def get_data(filename):
    data = None
    current = Path(__file__).parent.resolve()
    path = current.joinpath(f"input/{filename}")
    try:
        data = imageio.imread(path)
    except IOError:
        exit(f"Sorry! Couldn't find the file: {filename}")
    return data


def plot_and_save_generated_img(array, filename, fitness, parameters=None):
    plt.imshow(array)
    plt.title(f"{fitness}\n{parameters.values()}")
    # plt.show()

    # create a directory to save output in this working directory
    current = Path(__file__).parent.resolve()
    generated_img = current.joinpath(f"output/results/{filename}.jpg")
    plt.savefig(generated_img)
    return None


def get_chromosome(filename):
    target_im = numpy.asarray(filename / 255, dtype=float)
    target_chromosome = gari.img2chromosome(target_im)
    return target_chromosome


# convert image into chromosome required for Genetic Algorithm
def img2chromosome(img_arr):
    return numpy.reshape(
        a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape))
    )


# convert chromosome back to the image
def chromosome2img(vector, shape):
    if len(vector) != functools.reduce(operator.mul, shape):
        raise ValueError(
            f"A vector of length {len(vector)} into an array of shape {shape}."
        )
    return numpy.reshape(a=vector, newshape=shape)


if __name__ == "__main__":
    main()
