import numpy as np
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

def save_result(results, filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
