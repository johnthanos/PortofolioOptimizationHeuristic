
import numpy
import genetic_algorithm
import csv
import datetime
import time
import os

"""
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 100
num_parents_mating = 20
lamda = 1
alpha = 2
T = 145
starting_time = 0
Kappa = 10
cash = 1000000
gamma_list = [0.0000, 0.0025, 0.0050, 0.0075, 0.0100]

mutation_chance = 0.9
# crossover_chance = 0.2
sigma = 1 / 6
num_generations = 150
threshold = 0.00001

dataset = 8
now = datetime.datetime.now()
message = ""
# file_list = os.listdir("stocks")
file_list = ["indtrack1.csv", "indtrack2.csv", "indtrack3.csv", "indtrack4.csv", "indtrack5.csv"]
for file in file_list:
    with open("stocks/{}".format(file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        stock_values = list(csv_reader)
        stock_values = [list(map(float, i[:-1])) for i in stock_values]
    # Number of the weights we are looking to optimize. num of stocks
    num_weights = len(stock_values) + 1
    print(num_weights)

    # Defining the population size.
    # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    pop_size = (sol_per_pop, num_weights)
    epsilon = [0.01]*num_weights
    delta = [1]*num_weights
    for gamma in gamma_list:
        for j in range(5):
            # define starting portfolio, as equal quantities of kappa stocks
            current_portofolio = [0] * num_weights
            for i in range(1, Kappa + 1):
                current_portofolio[i] = 1 / Kappa
            fitness_list = list()
            index = genetic_algorithm.index_value(stock_values, Kappa)
            # Creating the initial population.
            new_population = numpy.random.uniform(low=0, high=1.0, size=pop_size)
            fit_population, unfit_population = genetic_algorithm.split_population(stock_values, new_population, index, T,
                                                                                  lamda, alpha, epsilon, delta, gamma,
                                                                                  Kappa)
            print(fit_population)
            start = time.time()
            for generation in range(num_generations):
                print("Generation : ", generation)
                # Measing the fitness of each chromosome in the population.
                # fitness = genetic_algorithm.pop_fitness(stock_values, new_population, T, lamda, alpha, epsilon, delta,
                # gamma)

                # Selecting the best parents in the population for mating.
                parents1 = genetic_algorithm.select_mating_pool(stock_values, fit_population, index, num_parents_mating, T,
                                                                lamda, alpha, epsilon, delta, gamma)
                parents2 = genetic_algorithm.select_mating_pool_unfit(stock_values, unfit_population, num_parents_mating, T,
                                                                      current_portofolio, starting_time, epsilon, delta,
                                                                      gamma)

                # Generating next generation using crossover.
                offspring_crossover_fit, offspring_crossover_unfit = genetic_algorithm.crossover(
                    parents1, parents2, index, stock_values, T, lamda, alpha, epsilon, delta, gamma, current_portofolio,
                    starting_time,  Kappa)

                # Adding some variations to the offspring using mutation.
                offspring_mutation_fit = genetic_algorithm.mutation(offspring_crossover_fit, sigma, mutation_chance, Kappa)
                offspring_mutation_unfit = genetic_algorithm.reverse_mutation(offspring_crossover_unfit, sigma,
                                                                              mutation_chance)

                if generation % 5 == 0 and generation > 0:
                    sigma = max(0, sigma * float((numpy.exp(genetic_algorithm.gaussian_mutation_calmped(0, 1)))))

                # Creating the new population based on the parents and offspring.
                fit_population = genetic_algorithm.new_populations_fit(fit_population, offspring_mutation_fit, stock_values,
                                                                       index, T, current_portofolio, starting_time, lamda,
                                                                       alpha, epsilon, delta, gamma)

                unfit_population = genetic_algorithm.new_populations_unfit(unfit_population, offspring_mutation_unfit,
                                                                           stock_values, index, T, current_portofolio,
                                                                           starting_time, lamda, alpha, epsilon, delta,
                                                                           gamma)

                end = time.time()
                print(datetime.timedelta(seconds=end - start))

                # Convergence
                fitness = genetic_algorithm.pop_fitness(stock_values, fit_population, index, T, lamda, alpha, epsilon,
                                                        delta,
                                                        gamma)
                print(numpy.min(fitness))
                print(numpy.average(fitness))
                message = "No convergence {}".format(generation)
                if numpy.average(fitness) - numpy.min(fitness) < threshold:
                    print("Convergence achieved")
                    message = "Convergence achieved {}".format(generation)
                    break

            # Getting the best solution after iterating finishing all generations.
            # At first, the fitness is calculated for each solution in the final generation.
            fitness = genetic_algorithm.pop_fitness(stock_values, fit_population, index, T, lamda, alpha, epsilon, delta,
                                                    gamma)
            fitness290 = genetic_algorithm.pop_fitness(stock_values, fit_population, index, 290, lamda, alpha, epsilon,
                                                       delta, gamma, starting_time=145)

            unfitness = genetic_algorithm.pop_unfitness(stock_values, fit_population, T, current_portofolio, starting_time,
                                                        epsilon, delta, gamma)
            # Then return the index of that solution corresponding to the best fitness.
            best_match_idx = numpy.where(fitness == numpy.min(fitness))

            print("Best solution : ", fit_population[best_match_idx[0][0]])
            print("Best solution fitness : ", fitness[best_match_idx[0][0]])
            print("Best solution outofsample error :  {}".format(fitness290[best_match_idx[0][0]]))

            print("Best solution unfitness : ", unfitness[best_match_idx[0][0]])

            with open("results145/{}_{}_{}_{}_{}.txt".format(now.month, now.day, now.hour, gamma, file.split(".")[0]),
                      "a") as outfile:
                outfile.write("Gamma value: {}".format(gamma))
                outfile.write("Repetition {}\n".format(j+1))
                outfile.write(message)
                outfile.write("\nBest solution : {}\n".format(fit_population[best_match_idx[0][0]]))
                outfile.write("Best solution insample error :  {}\n".format(fitness[best_match_idx[0][0]]))
                outfile.write("Best solution outofsample error :  {}\n".format(fitness290[best_match_idx[0][0]]))

                outfile.write("Best solution unfitness :  {}\n".format(unfitness[best_match_idx[0][0]]))
