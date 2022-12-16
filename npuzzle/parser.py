
from npuzzle import heuristics
from npuzzle import solved_states


def read_file(path):
    with open(path, "r") as f:
        data = f.read().splitlines()
    data = [line.strip().split("#")[0] for line in data]  # remove comments
    data = [line for line in data if len(line) > 0]  # remove empty lines
    puzzle = []
    for line in data:
        row = []
        for x in line.split(" "):
            if len(x) > 0:
                if not x.isdigit():
                    print("parser: invalid input, must be all numeric")
                    return None
                row.append(int(x))
        puzzle.append(row)
    size = puzzle[0][0]
    v = is_valid_input(puzzle)
    if v != "ok":
        print("parser: invalid input,", v)
        return None
    puzzle1d = []  # convert 2d matrix into list
    for row in puzzle:
        for item in row:
            puzzle1d.append(item)
    return (tuple(puzzle1d), size)
