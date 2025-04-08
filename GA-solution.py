import random
import numpy as np
from numpy.typing import NDArray
import time
import json
from utils import check, save_result

# Set global vars for the solver
mutation_rate = 0.01
generations = 100
population_size = 100

# Load puzzles
with open("small.txt") as f:
    small = [json.loads(line) for line in f]
with open("medium.txt") as f:
    medium = [json.loads(line) for line in f]
with open("large.txt") as f:
    large = [json.loads(line) for line in f]

results = {"small": [], "medium": [], "large": []}

def create_individual() -> list[int]:
    """Generate a random individual (binary grid representation)."""
    return [random.randint(0, 1) for _ in range(grid_size**2)]

def evaluate(individual: list[int]) -> int:
    """Evaluate the fitness of an individual based on row and column constraints."""
    grid = np.array(individual).reshape((grid_size, grid_size))
    score = 0  # Higher score means a better match to constraints

    # Evaluate row constraints
    for i, row in enumerate(grid):
        blocks = [sum(g) for g in np.split(row, np.where(row == 0)[0]) if sum(g) > 0]
        if blocks == row_constraints[i]:  
            score += 1  

    # Evaluate column constraints
    for j, col in enumerate(grid.T):  # Transpose to iterate over columns
        blocks = [sum(g) for g in np.split(col, np.where(col == 0)[0]) if sum(g) > 0]
        if blocks == col_constraints[j]:  
            score += 1  

    return score  # Higher score means better constraint satisfaction

def select(population: list[list[int]]) -> list[list[int]]:
    """Select the top-performing individuals for reproduction."""
    population.sort(key=evaluate, reverse=True)  # Sort by fitness (descending order)
    return population[:population_size // 2]  # Retain the top 50% of individuals

def crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    """Perform single-point crossover between two parents."""
    point = random.randint(0, grid_size**2 - 1)  # Choose a crossover point
    child1 = parent1[:point] + parent2[point:]  
    child2 = parent2[:point] + parent1[point:]  
    return child1, child2

def mutate(individual: list[int]) -> None:
    """Apply mutation to an individual with a certain probability."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:  
            individual[i] = 1 - individual[i]  # Flip the bit (0 → 1, 1 → 0)

def genetic_algorithm() -> NDArray[np.long]:
    """Solve the Picross puzzle using a genetic algorithm."""
    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        population = select(population)  # Select the fittest individuals

        # If an individual satisfies all constraints, terminate early
        if evaluate(population[0]) == grid_size * 2:
            break

        next_generation = population.copy()

        # Generate offspring until the population size is restored
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population, 2)  # Select two parents
            child1, child2 = crossover(parent1, parent2)  # Perform crossover
            mutate(child1)  # Apply mutation
            mutate(child2)  
            next_generation.extend([child1, child2])  # Add offspring to the population
        
        population = next_generation[:population_size]  # Ensure population size remains constant
        
        best_individual = max(population, key=evaluate)  # Track the best solution found

    # Display the best solution found
    best_solution = np.array(best_individual).reshape((grid_size, grid_size))
    
    return best_solution

# Run the genetic algorithm to solve the Picross puzzle
if __name__ == "__main__":
    
    for size_name, puzzles in [("small", small), ("medium", medium), ("large", large)]:
        for puzzle in puzzles:
            row_constraints = puzzle["row"]
            col_constraints = puzzle["col"]
            grid_size = len(row_constraints)

            start = time.time()
            grid = genetic_algorithm()
            end = time.time()
            print(end - start)
            correct = check(grid, row_constraints, col_constraints)
            results[size_name].append({"solved": correct, "time": end - start})

    # Save results
    save_result(results, "GA-results.json")
