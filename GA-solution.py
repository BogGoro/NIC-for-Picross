import random
import numpy as np

row_constraints = [[1], [3], [5], [3], [1]]
col_constraints = [[1], [3], [5], [3], [1]]

grid_size = len(row_constraints) 
population_size = 300
generations = 100
mutation_rate = 0.1

def create_individual():
    return [random.randint(0, 1) for _ in range(grid_size**2)]

def evaluate(individual):
    grid = np.array(individual).reshape((grid_size, grid_size))
    score = 0
    
    for i, row in enumerate(grid):
        blocks = [sum(g) for g in np.split(row, np.where(row == 0)[0]) if sum(g) > 0]
        if blocks == row_constraints[i]:
            score += 1
    
    for j, col in enumerate(grid.T):
        blocks = [sum(g) for g in np.split(col, np.where(col == 0)[0]) if sum(g) > 0]
        if blocks == col_constraints[j]:
            score += 1
    
    return score

def select(population):
    population.sort(key=evaluate, reverse=True)
    return population[:population_size // 2]

def crossover(parent1, parent2):
    point = random.randint(0, grid_size**2 - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]

def solve_picross():
    population = [create_individual() for _ in range(population_size)]
    
    for generation in range(generations):
        population = select(population)
        if evaluate(population[0]) == grid_size * 2:
            break
        next_generation = population.copy()
        
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            next_generation.extend([child1, child2])
        
        population = next_generation[:population_size]
        
        best_individual = max(population, key=evaluate)
    
    best_solution = np.array(best_individual).reshape((grid_size, grid_size))
    print("Best solution:")
    print(best_solution)

solve_picross()
