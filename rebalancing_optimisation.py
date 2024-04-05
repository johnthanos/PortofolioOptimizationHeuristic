
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
T = [150, 170, 190, 210, 230, 250, 270]
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
        for i in range(5):
            # define starting portfolio, as equal quantities of kappa stocks
            current_portofolio_list = []
            current_portofolio = [0] * num_weights
            for j in range(1, Kappa + 1):
                current_portofolio[j] = 1 / Kappa
            fitness_list = list()
            index = genetic_algorithm.index_value(stock_values, Kappa)
            # Creating the initial population.
            new_population = numpy.random.uniform(low=0, high=1.0, size=pop_size)
            fit_population, unfit_population = genetic_algorithm.split_population(stock_values, new_population, index, T[0],
                                                                                  lamda, alpha, epsilon, delta, gamma,
                                                                                  Kappa)
            current_portofolio_list.append(current_portofolio)
            print(fit_population)
            start = time.time()
            for t in T:
                for generation in range(num_generations):
                    print("Generation : ", generation)
                    # Measing the fitness of each chromosome in the population.
                    # fitness = genetic_algorithm.pop_fitness(stock_values, new_population, T, lamda, alpha, epsilon, delta,
                    # gamma)

                    # Selecting the best parents in the population for mating.
                    parents1 = genetic_algorithm.select_mating_pool(stock_values, fit_population, index,
                                                                    num_parents_mating, t,
                                                                    lamda, alpha, epsilon, delta, gamma)
                    parents2 = genetic_algorithm.select_mating_pool_unfit(stock_values, unfit_population,
                                                                          num_parents_mating, t, current_portofolio,
                                                                          starting_time, epsilon, delta, gamma)

                    # Generating next generation using crossover.
                    offspring_crossover_fit, offspring_crossover_unfit = genetic_algorithm.crossover(
                        parents1, parents2, index, stock_values, t, lamda, alpha, epsilon, delta, gamma,
                        current_portofolio, starting_time,  Kappa)

                    # Adding some variations to the offspring using mutation.
                    offspring_mutation_fit = genetic_algorithm.mutation(offspring_crossover_fit, sigma, mutation_chance,
                                                                        Kappa)
                    offspring_mutation_unfit = genetic_algorithm.reverse_mutation(offspring_crossover_unfit, sigma,
                                                                                  mutation_chance)

                    if generation % 5 == 0 and generation > 0:
                        sigma = max(0, sigma * float((numpy.exp(genetic_algorithm.gaussian_mutation_calmped(0, 1)))))

                    # Creating the new population based on the parents and offspring.
                    fit_population = genetic_algorithm.new_populations_fit(fit_population, offspring_mutation_fit,
                                                                           stock_values,
                                                                           index, t, current_portofolio, starting_time,
                                                                           lamda,
                                                                           alpha, epsilon, delta, gamma)

                    unfit_population = genetic_algorithm.new_populations_unfit(
                        unfit_population, offspring_mutation_unfit, stock_values, index, t, current_portofolio,
                        starting_time, lamda, alpha, epsilon, delta, gamma)

                    end = time.time()
                    print(datetime.timedelta(seconds=end - start))

                    # Convergence
                    fitness = genetic_algorithm.pop_fitness(stock_values, fit_population, index, t, lamda, alpha,
                                                            epsilon,
                                                            delta,
                                                            gamma)
                    print(numpy.min(fitness))
                    print(numpy.average(fitness))
                    message = "No convergence {}".format(generation)
                    if numpy.average(fitness) - numpy.min(fitness) < threshold:
                        print("Convergence achieved")
                        message = "Convergence achieved {}".format(generation)
                        break
                best_match_idx = numpy.where(fitness == numpy.min(fitness))
                current_portofolio = fit_population[best_match_idx[0][0]]
                current_portofolio_list.append(current_portofolio)

            # Getting the best solution after iterating finishing all generations.
            # At first, the fitness is calculated for each solution in the final generation.
            T_test = [0, 150, 170, 190, 210, 230, 250, 270, 290]
            total_error = []
            for _, portofolio in enumerate(current_portofolio_list):
                fitness = genetic_algorithm.pop_fitness(stock_values, [portofolio], index, T_test[i+1], lamda, alpha, epsilon,
                                                        delta,
                                                        gamma, starting_time=T_test[i])
                total_error.append(numpy.min(fitness))
                print(T_test[i+1])

            # Then return the index of that solution corresponding to the best fitness.
            # best_match_idx = numpy.where(fitness == numpy.min(fitness))

            print("Best solution : ", current_portofolio_list[-1])
            print("Total tracking error : ", total_error)

            with open("results_rebalancing/{}_{}_{}_{}_{}.txt".format(now.month, now.day, now.hour, gamma,
                                                                      file.split(".")[0]),
                      "a") as outfile:
                outfile.write("Gamma value: {}\n".format(gamma))
                outfile.write("Repetition {}\n".format(i+1))
                outfile.write(message)
                outfile.write("\nBest solution : {}\n".format(current_portofolio_list[-1]))
                outfile.write("Total rebalancing tracking error :  {}\n".format(sum(total_error)))
