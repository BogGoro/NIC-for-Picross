import numpy as np
from numpy.typing import NDArray
import json

def check(grid: np.ndarray, row_constraints, col_constraints) -> bool:
    def get_blocks(line) -> list[int]:
        return [sum(g) for g in np.split(line, np.where(line == 0)[0]) if sum(g) > 0]

    for i, row in enumerate(grid):
        if get_blocks(row) != row_constraints[i]:
            return False
    for j, col in enumerate(grid.T):
        if get_blocks(col) != col_constraints[j]:
            return False
    return True

def get_blocks(line: np.ndarray) -> list[int]:
    blocks = []
    count = 0
    for cell in line:
        if cell == 1:
            count += 1
        elif count:
            blocks.append(count)
            count = 0
    if count:
        blocks.append(count)
    return blocks

def puzzle_accuracy(grid: NDArray[np.int_], row_constraints: list[list[int]], col_constraints: list[list[int]]) -> float:
    """Proportion of rows and columns that match their constraints."""
    grid_size = grid.shape[0]
    correct_rows = sum(get_blocks(grid[i]) == row_constraints[i] for i in range(grid_size))
    correct_cols = sum(get_blocks(grid[:, j]) == col_constraints[j] for j in range(grid_size))
    return (correct_rows + correct_cols) / (2 * grid_size)

def constraints_accuracy(grid: NDArray[np.int_], row_constraints: list[list[int]], col_constraints: list[list[int]]) -> float:
    """Proportion of individual constraint elements (numbers in row/col specs) that match."""
    matched = 0
    total = 0

    for i in range(grid.shape[0]):
        actual = get_blocks(grid[i])
        expected = row_constraints[i]
        total += len(expected)
        matched += sum(1 for a, b in zip(actual, expected) if a == b)

    for j in range(grid.shape[1]):
        actual = get_blocks(grid[:, j])
        expected = col_constraints[j]
        total += len(expected)
        matched += sum(1 for a, b in zip(actual, expected) if a == b)

    if total == 0:
        return 1.0  # no constraints to match
    return matched / total


def save_result(results, filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
