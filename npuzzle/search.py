from itertools import count
from heapq import heappush, heappop
from collections import deque
from math import inf
import time
import random
EMPTY_TILE = 0


def clone_and_swap(puzzle, i, j):
    clone = list(puzzle)
    clone[i], clone[j] = clone[j], clone[i]
    return tuple(clone)


# UDLR
def possible_moves(puzzle, size):
    res = []
    i = puzzle.index(EMPTY_TILE)
    if i - size >= 0:
        res.append(clone_and_swap(puzzle, i, i - size))
    if i + size < len(puzzle):
        res.append(clone_and_swap(puzzle, i, i + size))
    if i % size > 0:
        res.append(clone_and_swap(puzzle, i, i - 1))
    if i % size + 1 < size:
        res.append(clone_and_swap(puzzle, i, i + 1))
    return res


def lifo(queue):
    return max(queue, key=lambda tup: tup[5])


def fifo(queue):
    return min(queue, key=lambda tup: tup[5])


def h_random(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[7])[7]
    nodes_with_best_h = [val for val in queue if val[7] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        node = random.choice(nodes_with_best_h)
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def random_select(queue):
    return random.choice(queue)


def g_h(queue, num_sec_tie):
    best_g = min(queue, key=lambda tup: tup[3])[3]
    nodes_with_best_g = [val for val in queue if val[3] == best_g]
    if len(nodes_with_best_g) > 1:
        num_sec_tie += 1
        best_h = min([val[7] for val in nodes_with_best_g])
        nodes_with_best_h = [val for val in nodes_with_best_g if val[7] == best_h]
        node = nodes_with_best_h[0]
    else:
        node = nodes_with_best_g[0]
    return node, num_sec_tie


def h_g(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[7])[7]
    nodes_with_best_h = [val for val in queue if val[7] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        best_g = min([val[3] for val in nodes_with_best_h])
        nodes_with_best_g = [val for val in nodes_with_best_h if val[3] == best_g]
        node = nodes_with_best_g[0]
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def h_fifo(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[7])[7]
    nodes_with_best_h = [val for val in queue if val[7] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        node = min(nodes_with_best_h, key=lambda tup: tup[5])
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def h_lifo(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[7])[7]
    nodes_with_best_h = [val for val in queue if val[7] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        node = max(nodes_with_best_h, key=lambda tup: tup[5])
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def depth_last(queue, num_sec_tie):
    best_depth = max([val[6] for val in queue])
    nodes_with_best_depth = [val for val in queue if val[6] == best_depth]
    if len(nodes_with_best_depth) > 1:
        num_sec_tie += 1
        node = random.choice(nodes_with_best_depth)
    else:
        node = nodes_with_best_depth[0]
    return node, num_sec_tie


def depth_first(queue, num_sec_tie):
    best_depth = min([val[6] for val in queue])
    nodes_with_best_depth = [val for val in queue if val[6] == best_depth]
    if len(nodes_with_best_depth) > 1:
        num_sec_tie += 1
        node = random.choice(nodes_with_best_depth)
    else:
        node = nodes_with_best_depth[0]
    return node, num_sec_tie


def heappop_adj(queue, priority, num_tie, sec_tie):
    best_f = min(queue, key=lambda tup: tup[0])[0]
    nodes_with_best_f = [val for val in queue if val[0] == best_f]
    if len(nodes_with_best_f) > 1:
        num_tie += 1
        if priority == "LIFO":
            node = lifo(nodes_with_best_f)
        elif priority == "FIFO":
            node = fifo(nodes_with_best_f)
        elif priority == "H-LIFO":
            node, sec_tie = h_lifo(nodes_with_best_f, sec_tie)
        elif priority == "H-FIFO":
            node, sec_tie = h_fifo(nodes_with_best_f, sec_tie)
        elif priority == "H-G":
            node, sec_tie = h_g(nodes_with_best_f, sec_tie)
        elif priority == "G-H":
            node, sec_tie = g_h(nodes_with_best_f, sec_tie)
        elif priority == "RANDOM":
            node = random_select(nodes_with_best_f)
        elif priority == "H-RAND":
            node, sec_tie = h_random(nodes_with_best_f, sec_tie)
        elif priority == "DEPTH-L":
            node, sec_tie = depth_last(nodes_with_best_f, sec_tie)
        elif priority == "DEPTH-F":
            node, sec_tie = depth_first(nodes_with_best_f, sec_tie)

    else:
        node = nodes_with_best_f[0]
    queue.remove(node)
    return queue, node, num_tie, sec_tie


def a_star_search(puzzle, solved, size, HEURISTIC, TRANSITION_COST, priority):
    c = count()
    queue = [(0, next(c), puzzle, 0, None, time.time(), -1, 0)]  # node = (f(n), node_idx, curr_puzz, g(n), parent_idx(n), timestamp, depth, h(n))
    open_set = {}
    closed_set = {}
    last_f, last_h, last_depth, count_tie, second_tie = 0, 0, 0, 0, 0
    while queue:
        queue, pop_node, count_tie, second_tie = heappop_adj(queue, priority, count_tie, second_tie)
        f, n_idx, node, node_g, parent, _, _, _ = pop_node
        if node == solved:
            steps = [node]
            while parent is not None:
                steps.append(parent)
                parent = closed_set[parent]
            steps.reverse()
            return (True, steps, {"space": len(open_set), "time": len(closed_set), "count_of_tie": count_tie, "second_tie": second_tie})
        if node in closed_set:
            continue
        closed_set[node] = parent
        tentative_g = node_g + TRANSITION_COST
        moves = possible_moves(node, size)
        for m in moves:
            if m in closed_set:
                continue
            if m in open_set:
                move_g, move_h = open_set[m]
                if move_g <= tentative_g:
                    continue
            else:
                move_h = HEURISTIC(m, solved, size)
            f_curr = tentative_g + move_h
            h_curr = move_h
            if last_f == f_curr and h_curr == last_h:
                depth = last_depth + 1
            else:
                depth = 0
            last_depth = depth
            last_h = move_h
            last_f = f_curr
            open_set[m] = tentative_g, move_h
            heappush(queue, (move_h + tentative_g, next(c), m, tentative_g, node, time.time(), depth, move_h))
    return (False, [], {"space": len(open_set), "time": len(closed_set), "count of tie": count_tie, "second_tie": second_tie})
