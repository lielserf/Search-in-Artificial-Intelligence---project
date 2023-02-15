import os
import sys
from itertools import count
import pandas as pd
from time import perf_counter
from npuzzle.search import a_star_search
from npuzzle.is_solvable import is_solvable
from npuzzle import parser
from npuzzle import heuristics
from npuzzle import solved_states


def write_to_csv(size, p, heurist, num_evaluated, time_per_node, num_of_ties, num_steps, overall_time, sec_ties, space_, num_ex, num_thr_tie):
    dic = {"size": [size], "priority": [p], "heuristic_func": [heurist], "#expirement": [num_ex], "num_evaluated": [num_evaluated], "time_per_node": [time_per_node],
           "num_of_ties": [num_of_ties], "num_steps": [num_steps], "overall_time": [overall_time], "number_second_tie": [sec_ties], "number_third_tie": [num_thr_tie], "nodes in memory": [space_]}
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(os.getcwd() + '/result_puzzle.csv', header=False, mode='a')


def main(zero_location, heuristic_function, priority, num_exp):
    directory = os.getcwd() + '/puzzles'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            puzzle, size = parser.read_file(f)

        TRANSITION_COST = 1
        HEURISTIC = heuristics.KV[heuristic_function]
        solved = solved_states.KV[zero_location](size)

        if not is_solvable(puzzle, solved, size):
            print("This puzzle is not solvable")
            sys.exit(0)

        t_start = perf_counter()
        res = a_star_search(puzzle, solved, size, HEURISTIC, TRANSITION_COST, priority)
        t_delta = perf_counter() - t_start

        success, steps, complexity = res
        num_evaluated = complexity["time"]
        num_of_ties = complexity["count_of_tie"]
        num_second_tie = complexity["second_tie"]
        num_thr_tie = complexity['third_tie']
        time_per_node = t_delta / max(num_evaluated, 1)
        print(f"evaluated nodes: {num_evaluated}")
        print(f"second(s) per node: {time_per_node:.8f}")
        print(f"number of ties: {num_of_ties:.8f}")

        if not success:
            print("solution not found")
        print("space complexity:", complexity["space"], "nodes in memory")
        print("-" * 40)
        write_to_csv(size, priority, heuristic_function, num_evaluated, time_per_node, num_of_ties, len(steps), t_delta, num_second_tie, complexity["space"], num_exp, num_thr_tie)

zero_location_lst = ["zero_first", "zero_last", "snail"]
heuristic_function_lst = ["linear_conflicts", "manhattan"]
# priority_lst = ["LIFO", "FIFO", "H-LIFO", "H-FIFO", "H-G", "G-H", "RANDOM", "H-RAND", "DEPTH-L", "DEPTH-F"]
priority_lst = ["H-DEPTH-L", "H-DEPTH-F"]

for k in range(len(heuristic_function_lst)):
    for j in range(len(priority_lst)):
        for ex in range(3):
            main(zero_location_lst[1], heuristic_function_lst[k], priority_lst[j], ex)

# main(zero_location_lst[1], heuristic_function_lst[0], priority_lst[0])
