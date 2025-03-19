import random
import numpy as np
from numpy.typing import NDArray

# Define row and column constraints for the Picross puzzle
row_constraints = [[1], [3], [5], [3], [1]]
col_constraints = [[1], [3], [5], [3], [1]]

# Define problem parameters
grid_size = len(row_constraints)  # Assuming a square grid
initial_temperature = 10.0  # Starting temperature for simulated annealing
cooling_rate = 0.999  # Rate at which temperature decreases
iterations = 10000  # Number of iterations for annealing process

def evaluate(grid: NDArray[np.long]) -> int:
    """
    Calculate the fitness score of a grid based on row and column constraints.
    A higher score indicates a better match to the constraints.
    """
    score = 0  # Initialize score
    
    # Evaluate row constraints
    for i, row in enumerate(grid):
        # Find contiguous blocks of 1s in the row
        blocks = [sum(g) for g in np.split(row, np.where(row == 0)[0]) if sum(g) > 0]
        if blocks == row_constraints[i]:  
            score += 1  

    # Evaluate column constraints
    for j, col in enumerate(grid.T):  # Transpose to iterate over columns
        blocks = [sum(g) for g in np.split(col, np.where(col == 0)[0]) if sum(g) > 0]
        if blocks == col_constraints[j]:  
            score += 1  

    return score  # Higher score means better constraint satisfaction

def mutate(grid: NDArray[np.long]) -> NDArray[np.long]:
    """
    Create a new grid by flipping a random cell (mutating the grid).
    """
    new_grid = grid.copy()  # Copy the current grid
    i, j = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)  # Select a random cell
    new_grid[i, j] = 1 - new_grid[i, j]  # Flip the bit (0 → 1, 1 → 0)
    return new_grid

def simulated_annealing() -> NDArray[np.long]:
    """
    Solve the Picross puzzle using the simulated annealing algorithm.
    """
    # Initialize a random grid with binary values (0 or 1)
    current_grid = np.random.randint(0, 2, (grid_size, grid_size))
    current_score = evaluate(current_grid)  # Compute its initial fitness score
    temperature = initial_temperature  # Set the initial temperature
    
    # Keep track of the best solution found
    best_grid, best_score = current_grid, current_score

    for step in range(iterations):
        # Generate a new candidate solution by mutating the current grid
        new_grid = mutate(current_grid)
        new_score = evaluate(new_grid)

        # If the new grid is better, accept it unconditionally
        if new_score > current_score:
            current_grid, current_score = new_grid, new_score
        else:
            # Otherwise, accept with a probability determined by the temperature
            delta = new_score - current_score
            if random.random() < np.exp(delta / temperature):  
                current_grid, current_score = new_grid, new_score

        # Update the best solution if the new one is the best so far
        if new_score > best_score:
            best_grid, best_score = new_grid, new_score

        # Reduce the temperature over time
        temperature *= cooling_rate

        # Stop early if a perfect solution is found
        if best_score == grid_size * 2:
            break

    # Display the best solution found
    print("Best solution:")
    print(best_grid)

    return best_grid

# Run the simulated annealing algorithm to solve the Picross puzzle
solved_grid = simulated_annealing()
