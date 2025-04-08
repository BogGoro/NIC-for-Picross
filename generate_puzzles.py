import numpy as np
import json
from typing import List, Tuple
from itertools import product, groupby

def extract_constraints(grid: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    def get_line_constraints(line):
        constraints = []
        count = 0
        for val in line:
            if val == 1:
                count += 1
            elif count > 0:
                constraints.append(count)
                count = 0
        if count > 0:
            constraints.append(count)
        return constraints or [0]

    row_constraints = [get_line_constraints(row) for row in grid]
    col_constraints = [get_line_constraints(col) for col in grid.T]
    return row_constraints, col_constraints

def generate_puzzles(size: int, count: int) -> List[dict]:
    puzzles = []
    while len(puzzles) < count:
        solution = np.random.randint(0, 2, (size, size))
        row_cons, col_cons = extract_constraints(solution)
        puzzles.append({
            "row": row_cons,
            "col": col_cons,
            "solution": solution.tolist()
        })
    return puzzles

def save_to_file(puzzles: List[dict], filename: str):
    with open(filename, 'w') as f:
        for puzzle in puzzles:
            f.write(json.dumps(puzzle) + '\n')

if __name__ == "__main__":
    save_to_file(generate_puzzles(5, 10), "small.txt")
    save_to_file(generate_puzzles(10, 10), "medium.txt")
    save_to_file(generate_puzzles(15, 10), "large.txt")
