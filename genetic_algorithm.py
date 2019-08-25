
import numpy


def cal_pop_fitness(equation_inputs, individual, index, tau, lamda, alpha):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    e = [0] * tau
    for t in range(1, tau):
        sum1 = [0] * len(individual)
        sum2 = [0] * len(individual)
        for i, ind in enumerate(individual):
            if i == 0:
                continue
            sum1[i-1] = float(ind) * equation_inputs[i-1][t] / equation_inputs[i-1][tau]
            sum2[i-1] = float(ind) * equation_inputs[i-1][t-1] / equation_inputs[i-1][tau]
        new_return = numpy.log(numpy.sum(sum1)/numpy.sum(sum2))
        index_return = numpy.log(index[t]/index[t-1])
        e[t] = new_return - index_return
    gain = numpy.sum(e)/tau
    error = numpy.power(numpy.sum(numpy.power(numpy.abs(e), alpha)), 1/alpha)/tau

    fitness = lamda*error - (1-lamda)*gain

    return fitness


def cal_pop_unfitness(equation_inputs, individual, tau, curent_portofolio, starting_time):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    cost_sum = 0
    for i, y in enumerate(individual):
        if i > 0:
            cost_sum += 0.01*abs(curent_portofolio[i-1]/equation_inputs[i-1][starting_time] - y/equation_inputs[i-1][tau])
    unfitness = abs(individual[0] - cost_sum)

    return unfitness


def pop_fitness(equation_inputs, pop, index, tau, lamda, alpha, epsilon, delta, gamma):
    fitness = []
    for individual in pop:
        variables = transform_chromosome_to_variables(individual, epsilon, delta, gamma)
        fitness.append(cal_pop_fitness(equation_inputs, variables, index, tau, lamda, alpha))

    return fitness


def pop_unfitness(equation_inputs, pop, tau, curent_portofolio, starting_time, epsilon, delta, gamma):
    unfitness = []
    for individual in pop:
        variables = transform_chromosome_to_variables(individual, epsilon, delta, gamma)
        unfitness.append(cal_pop_unfitness(equation_inputs, variables, tau, curent_portofolio, starting_time))

    return unfitness


def fitness_sum(equation_inputs, pop, index, tau, lamda, alpha, epsilon, delta, gamma):
    fitness = sum(pop_fitness(equation_inputs, pop, index, tau, lamda, alpha, epsilon, delta, gamma))

    return fitness


def unfitness_sum(equation_inputs, pop, tau, curent_portofolio, starting_time, epsilon, delta, gamma):
    unfitness = sum(pop_unfitness(equation_inputs, pop, tau, curent_portofolio, starting_time, epsilon,
                                  delta, gamma))

    return unfitness


def select_mating_pool(equation_inputs, pop, index, num_parents, tau, lamda, alpha, epsilon, delta, gamma):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next
    # generation. Rhoulette wheel
    # f_eval = fitness_sum(equation_inputs, pop, tau, lamda, alpha, epsilon, delta, gamma)
    population = list(pop)
    fitness = pop_fitness(equation_inputs, population, index, tau, lamda, alpha, epsilon, delta, gamma)
    f_eval = sum(fitness)
    graded = numpy.true_divide(fitness, f_eval*1.0)
    graded = [[graded[i], population[i]] for i in range(len(graded))]
    graded = [list(x) for x in sorted(graded)]

    parents = []
    past = 0
    for i in range(len(graded)):
        graded[i][0] += past
        past = graded[i][0]
    for i in range(num_parents):
        prob = numpy.random.random_sample()
        for x in graded:
            if x[0] > prob:
                parents.append(x[1])
                break
    return parents


def select_mating_pool_unfit(equation_inputs, pop, num_parents, tau, cp, st, epsilon, delta, gamma):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next
    # generation. Rhoulette wheel
    population = list(pop)
    unfitness = pop_unfitness(equation_inputs, population, tau, cp, st, epsilon, delta, gamma)
    f_eval = sum(unfitness)
    graded = numpy.true_divide(unfitness, f_eval * 1.0)
    graded = [[graded[i], population[i]] for i in range(len(graded))]
    graded = [list(x) for x in sorted(graded)]

    parents = []
    past = 0
    for i in range(len(graded)):
        graded[i][0] += past
        past = graded[i][0]
    for i in range(num_parents):
        prob = numpy.random.random_sample()
        for x in graded:
            if x[0] > prob:
                parents.append(x[1])
                break
    return parents


def crossover(parents_pop1, parents_pop2, index, equation_inputs, tau, lamda, alpha, epsilon, delta, gamma,
              curent_portofolio, starting_time, kappa):
    new_population = []
    new_reverse_population = []
    fitness = pop_fitness(equation_inputs, parents_pop1, index, tau, lamda, alpha, epsilon, delta, gamma)
    offspring = [list()]*10

    for k in range(0, len(parents_pop1), 2):
        # Index of the first pop parent to mate.
        offspring[0] = parents_pop1[k]
        offspring[1] = parents_pop1[k+1]
        # Index of the second pop parent to mate.
        offspring[2] = parents_pop2[k]
        offspring[3] = parents_pop2[k + 1]
        # crossover positions
        pos1, pos2 = numpy.sort(numpy.random.randint(len(parents_pop1[0]), size=2))
        # fit offspring
        offspring[4] = parents_pop1[k][:pos1] + parents_pop1[k+1][pos1:pos2] + parents_pop1[k][pos2:]
        offspring[5] = parents_pop1[k+1][:pos1] + parents_pop1[k][pos1:pos2] + parents_pop1[k+1][pos2:]
        # reverse offspring
        offspring[6] = parents_pop2[k][:pos1] + parents_pop2[k + 1][pos1:pos2] + parents_pop2[k][pos2:]
        offspring[7] = parents_pop2[k + 1][:pos1] + parents_pop2[k][pos1:pos2] + parents_pop2[k + 1][pos2:]

        # crossbreed offspring, find fittest parents
        if fitness[k] > fitness[k+1]:
            pm = parents_pop1[k]
        else:
            pm = parents_pop1[k+1]
        dist1 = numpy.linalg.norm(numpy.array(pm) - numpy.array(parents_pop2[k]))
        dist2 = numpy.linalg.norm(numpy.array(pm) - numpy.array(parents_pop2[k+1]))
        if dist1 < dist2:
            pr = parents_pop2[k]
        else:
            pr = parents_pop2[k+1]

        offspring[8] = pm[:pos1] + pr[pos1:pos2] + pm[pos2:]
        offspring[9] = pr[:pos1] + pm[pos1:pos2] + pr[pos2:]

        offspring = [check_selected_stocks(x, kappa) for x in offspring]

        # find best offsprings
        fit = pop_fitness(equation_inputs, offspring, index, tau, lamda, alpha, epsilon, delta, gamma)
        fit = [[fit[i], offspring[i]] for i in range(len(fit))]
        fit = [list(x) for x in sorted(fit)]

        new_population.append(fit[0][1])
        new_population.append(fit[1][1])

        unfit = pop_unfitness(equation_inputs, offspring, tau, curent_portofolio, starting_time, epsilon, delta, gamma)
        unfit = [[unfit[i], offspring[i]] for i in range(len(unfit))]
        unfit = [list(x) for x in sorted(unfit, reverse=True)]

        new_reverse_population.append(unfit[0][1])
        new_reverse_population.append(unfit[1][1])

    return new_population, new_reverse_population


def check_selected_stocks(individual, kappa):

    gene = individual[1:]
    while numpy.count_nonzero(gene) > kappa:
        j = numpy.array(numpy.nonzero(gene))[0]
        index = numpy.random.choice(j)
        gene[index] = 0

    while numpy.count_nonzero(gene) < kappa:
        j = numpy.array(numpy.nonzero(numpy.array(gene) == 0))[0]
        index = numpy.random.choice(j)
        gene[index] = numpy.random.random_sample()
    individual = list([individual[0]] + gene)

    return individual


def mutation(offspring_crossover, stddev, chance, kappa):
    # Mutation changes a single gene in each offspring randomly.
    for idx1, chromosome in enumerate(offspring_crossover):
        for idx2, gene in enumerate(chromosome):
            if numpy.random.random_sample() < chance:
                offspring_crossover[idx1][idx2] = gaussian_mutation(offspring_crossover[idx1][idx2], stddev)
        offspring_crossover[idx1] = check_selected_stocks(offspring_crossover[idx1], kappa)

    return offspring_crossover


def reverse_mutation(offspring_crossover, stddev, chance):
    # Mutation changes a single gene in each offspring randomly.
    for idx1, chromosome in enumerate(offspring_crossover):
        for idx2, gene in enumerate(chromosome):
            if numpy.random.random_sample() < chance:
                offspring_crossover[idx1][idx2] = gaussian_mutation(offspring_crossover[idx1][idx2], stddev)

    return offspring_crossover


def gaussian_mutation(mean, stddev):
    # The random value to be added to the gene.
    x1 = numpy.random.random_sample()
    x2 = numpy.random.random_sample()
    random_value = numpy.sqrt(-2.0 * numpy.log(x1)) * numpy.cos(2.0 * numpy.pi * x2)
    new_value = mean + random_value * stddev
    if new_value > 1:
        return 1
    elif new_value < 0:
        return 0
    return new_value


def split_population(stocks_values, initial_population, index, T, lamda, alpha, epsilon, delta, gamma, kappa):

    fitness_list = pop_fitness(stocks_values, initial_population, index, T, lamda, alpha, epsilon, delta, gamma)
    fitness_list, initial_population = (list(t) for t in zip(*sorted(zip(fitness_list, initial_population))))

    fit_population = (list(t) for t in initial_population[:int(len(initial_population)/2)])
    unfit_population = (list(t) for t in initial_population[int(len(initial_population)/2):])

    fit_population = [check_selected_stocks(x, kappa) for x in list(fit_population)]
    return list(fit_population), list(unfit_population)


def transform_chromosome_to_variables(chrom, epsilon, delta, gamma):

    y = [0]*len(chrom)
    epsilon[0] = 0
    delta[0] = gamma
    sum_epsilon = 0
    sum_s = 0
    sum_delta = 0
    q = []
    delta_flag = False
    for i, s in enumerate(chrom):
        if s > 0:
            sum_epsilon += epsilon[i]
            sum_s += s
            sum_delta += delta[i]
    for i, s in enumerate(chrom):
        if s > 0 or i == 0:
            y[i] = epsilon[i] + s*(1 - sum_epsilon)/sum_s
            q.append(i)
        if y[i] > delta[i]:
            delta_flag = True
    if delta_flag:
        for i, s in enumerate(chrom):
            if y[i] > delta[i]:
                y[i] = delta[i]
            elif s > 0 or i == 0:
                y[i] = epsilon[i] + s * (1 - (sum_epsilon + sum_delta)) / sum_s

    return y


def index_value(stock_values, kappa):
    index = [0]*len(stock_values[0])

    for t in range(len(stock_values[0])):
        for i in range(len(stock_values) - kappa + 1, len(stock_values)):
            index[t] += stock_values[i][t]

    return index


def new_populations_fit(population, offspring_pop, stock_values, index, T, current_portofolio,
                        starting_time, lamda, alpha, epsilon, delta, gamma):

    fitness = pop_fitness(stock_values, population, index, T, lamda, alpha, epsilon, delta,
                          gamma)
    unfitness = pop_unfitness(stock_values, population, T, current_portofolio, starting_time,
                              epsilon, delta, gamma)

    offspring_fitness = pop_fitness(stock_values, offspring_pop, index, T, lamda, alpha, epsilon, delta,
                                    gamma)
    offspring_unfitness = pop_unfitness(stock_values, offspring_pop, T, current_portofolio, starting_time,
                                        epsilon, delta, gamma)

    for i, offspring in enumerate(offspring_pop):
        flag = True
        for j, individual in enumerate(population):

            if fitness[j] > offspring_fitness[i] and unfitness[j] > offspring_unfitness[i]:
                population.pop(j)
                fitness.pop(j)
                unfitness.pop(j)
                flag = False
                break
        if flag:
            flag = True
            for j, individual in enumerate(population):
                if fitness[j] < offspring_fitness[i] and unfitness[j] > offspring_unfitness[i]:
                    population.pop(j)
                    fitness.pop(j)
                    unfitness.pop(j)
                    flag = False
                    break
        if flag:
            flag = True
            for j, individual in enumerate(population):
                if fitness[j] > offspring_fitness[i] and unfitness[j] < offspring_unfitness[i]:
                    population.pop(j)
                    fitness.pop(j)
                    unfitness.pop(j)
                    flag = False
                    break
        if flag:
            best_match_idx = numpy.where(unfitness == numpy.max(unfitness))
            population.pop(best_match_idx[0][0])
            fitness.pop(best_match_idx[0][0])
            unfitness.pop(best_match_idx[0][0])

    population += offspring_pop

    return population


def new_populations_unfit(population, offspring_pop, stock_values, index, T, current_portofolio,
                          starting_time, lamda, alpha, epsilon, delta, gamma):
    fitness = pop_fitness(stock_values, population, index, T, lamda, alpha, epsilon, delta,
                          gamma)
    unfitness = pop_unfitness(stock_values, population, T, current_portofolio, starting_time,
                              epsilon, delta, gamma)

    offspring_fitness = pop_fitness(stock_values, offspring_pop, index, T, lamda, alpha, epsilon, delta,
                                    gamma)
    offspring_unfitness = pop_unfitness(stock_values, offspring_pop, T, current_portofolio, starting_time,
                                        epsilon, delta, gamma)

    for i, offspring in enumerate(offspring_pop):
        flag = True
        for j, individual in enumerate(population):

            if fitness[j] < offspring_fitness[i] and unfitness[j] < offspring_unfitness[i]:
                population.pop(j)
                fitness.pop(j)
                unfitness.pop(j)
                flag = False
                break
        if flag:
            flag = True
            for j, individual in enumerate(population):
                if fitness[j] > offspring_fitness[i] and unfitness[j] < offspring_unfitness[i]:
                    population.pop(j)
                    fitness.pop(j)
                    unfitness.pop(j)
                    flag = False
                    break
        if flag:
            flag = True
            for j, individual in enumerate(population):
                if fitness[j] < offspring_fitness[i] and unfitness[j] > offspring_unfitness[i]:
                    population.pop(j)
                    fitness.pop(j)
                    unfitness.pop(j)
                    flag = False
                    break
        if flag:
            best_match_idx = numpy.where(unfitness == numpy.min(unfitness))
            population.pop(best_match_idx[0][0])
            fitness.pop(best_match_idx[0][0])
            unfitness.pop(best_match_idx[0][0])

    population += offspring_pop

    return population
