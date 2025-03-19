import random
import numpy as np

row_constraints = [[1], [3], [5], [3], [1]]
col_constraints = [[1], [3], [5], [3], [1]]

grid_size = len(row_constraints)
initial_temperature = 10.0
cooling_rate = 0.999
iterations = 10000

def evaluate(grid):
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

def mutate(grid):
    new_grid = grid.copy()
    i, j = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    new_grid[i, j] = 1 - new_grid[i, j]
    return new_grid

def simulated_annealing():
    current_grid = np.random.randint(0, 2, (grid_size, grid_size))
    current_score = evaluate(current_grid)
    temperature = initial_temperature
    best_grid, best_score = current_grid, current_score
    
    for step in range(iterations):
        new_grid = mutate(current_grid)
        new_score = evaluate(new_grid)
        
        if new_score > current_score:
            current_grid, current_score = new_grid, new_score
        else:
            delta = new_score - current_score
            if random.random() < np.exp(delta / temperature):
                current_grid, current_score = new_grid, new_score
        
        if new_score > best_score:
            best_grid, best_score = new_grid, new_score
        
        temperature *= cooling_rate
        if best_score == grid_size * 2:
            break
    
    print("Best solution:")
    print(best_grid)
    
    return best_grid

solved_grid = simulated_annealing()