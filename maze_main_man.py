import time
import random
import numpy as np
from collections import defaultdict
import pandas as pd
from time import perf_counter
import math
import os


class PrimsMaze:
    def __init__(self, size=25, show_maze=False):
        self.size = (size // 2) * 2 + 1
        self.show_maze = show_maze
        self.walls_list = []
        self.grid = np.full((self.size, self.size), -50, dtype=int)
        for i in range(size//2+1):
            for j in range(size//2+1):
                self.grid[i*2, j*2] = -1
        self.maze = np.zeros((self.size, self.size), dtype=bool)
        # print(self.grid)

    def is_valid(self, curr, dx, dy):
        x, y = curr
        if 0 <= x + dx < self.size and 0 <= y + dy < self.size:
            return True
        return False

    def add_neighbors(self, curr):
        nearby = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dx, dy in nearby:
            if self.is_valid(curr, dx, dy):
                self.walls_list.append((curr[0]+dx, curr[1]+dy))

    def create_maze(self, start):
        start = ((start[0]//2)*2, (start[1]//2)*2)
        self.grid[start[0], start[1]] = 1
        #self.grid[0, ::2] = self.grid[-1, ::2] = 1
        #self.grid[::2, 0] = self.grid[::2, -1] = 1
        self.add_neighbors(start)
        while len(self.walls_list):
            ind = np.random.randint(0, len(self.walls_list))
            wall_x, wall_y = self.walls_list[ind]
            if self.is_valid((wall_x, wall_y), -1, 0) and self.is_valid((wall_x, wall_y), 1, 0):
                top = wall_x-1, wall_y
                bottom = wall_x+1, wall_y
                if (self.grid[top] == 1 and self.grid[bottom] == -1):
                    self.grid[wall_x, wall_y] = 1
                    self.grid[bottom] = 1
                    self.add_neighbors(bottom)
                elif (self.grid[top] == -1 and self.grid[bottom] == 1):
                    self.grid[wall_x, wall_y] = 1
                    self.grid[top] = 1
                    self.add_neighbors(top)
                self.walls_list.remove((wall_x, wall_y))
            if self.is_valid((wall_x, wall_y), 0, 1) and self.is_valid((wall_x, wall_y), 0, -1):
                left = wall_x, wall_y-1
                right = wall_x, wall_y+1
                if (self.grid[left] == 1 and self.grid[right] == -1):
                    self.grid[wall_x, wall_y] = 1
                    self.grid[right] = 1
                    self.add_neighbors(right)
                elif (self.grid[left] == -1 and self.grid[right] == 1):
                    self.grid[wall_x, wall_y] = 1
                    self.grid[left] = 1
                    self.add_neighbors(left)
                self.walls_list.remove((wall_x, wall_y))

            '''
            '''
        #     if self.show_maze:
        #         img = self.grid                 # Display maze while building
        #         plt.figure(1)
        #         plt.clf()
        #         plt.imshow(img)
        #         plt.title('Maze')
        #         plt.pause(0.005)
        #         #plt.pause(5)
        #
        # plt.pause(5)

        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 1:
                    self.maze[row, col] = True

        # print(self.maze.dtype)
        return self.maze

def reconstruct_path(came_from, current):
    final_path = [current]
    while current in came_from:
        current = came_from[current]
        final_path.append(current)
    return final_path


def heauristic(cell, goal):
    x1, y1 = cell
    x2, y2 = goal

    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return dist


def lifo(queue):
    return max(queue, key=lambda tup: tup[3])


def fifo(queue):
    return min(queue, key=lambda tup: tup[3])


def h_random(queue, num_sec_tie):
    best_h = min([val[4] for val in queue])
    nodes_with_best_h = [val for val in queue if val[4] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        node = random.choice(nodes_with_best_h)
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def random_select(queue):
    return random.choice(queue)


def g_h(queue, num_sec_tie):
    best_g = min(queue, key=lambda tup: tup[2])[2]
    nodes_with_best_g = [val for val in queue if val[2] == best_g]
    if len(nodes_with_best_g) > 1:
        num_sec_tie += 1
        best_h = min([val[4] for val in nodes_with_best_g])
        nodes_with_best_h = [val for val in nodes_with_best_g if val[4] == best_h]
        node = nodes_with_best_h[0]
    else:
        node = nodes_with_best_g[0]
    return node, num_sec_tie


def h_g(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[4])[4]
    nodes_with_best_h = [val for val in queue if val[4] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        best_g = min([val[2] for val in nodes_with_best_h])
        nodes_with_best_g = [val for val in nodes_with_best_h if val[2] == best_g]
        node = nodes_with_best_g[0]
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def h_fifo(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[4])[4]
    nodes_with_best_h = [val for val in queue if val[4] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        node = min(nodes_with_best_h, key=lambda tup: tup[3])
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def h_lifo(queue, num_sec_tie):
    best_h = min(queue, key=lambda tup: tup[4])[4]
    nodes_with_best_h = [val for val in queue if val[4] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        node = max(nodes_with_best_h, key=lambda tup: tup[3])
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie


def depth_last(queue, num_sec_tie, num_thr_tie):
    best_h = min(queue, key=lambda tup: tup[4])[4]
    nodes_with_best_h = [val for val in queue if val[4] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        best_depth = max([val[5] for val in queue])
        nodes_with_best_depth = [val for val in queue if val[5] == best_depth]
        if len(nodes_with_best_depth) > 1:
            num_thr_tie += 1
            node = random.choice(nodes_with_best_depth)
        else:
            node = nodes_with_best_depth[0]
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie, num_thr_tie


def depth_first(queue, num_sec_tie, num_thr_tie):
    best_h = min(queue, key=lambda tup: tup[4])[4]
    nodes_with_best_h = [val for val in queue if val[4] == best_h]
    if len(nodes_with_best_h) > 1:
        num_sec_tie += 1
        best_depth = min([val[5] for val in queue])
        nodes_with_best_depth = [val for val in queue if val[5] == best_depth]
        if len(nodes_with_best_depth) > 1:
            num_thr_tie += 1
            node = random.choice(nodes_with_best_depth)
        else:
            node = nodes_with_best_depth[0]
    else:
        node = nodes_with_best_h[0]
    return node, num_sec_tie, num_thr_tie


def heappop_adj(open_set, fscore, gscore, priority, num_tie, times, sec_tie, depth_dic, num_thr_tie):
    nodes = []  # node = (f(n), node_loc, g(n), timestamp, h(n), depth)
    for n in open_set:
        f_n = fscore[n]
        g_n = gscore[n]
        time_n = times[n]
        h_n = f_n - g_n
        dep = depth_dic[n]
        nodes.append((f_n, n, g_n, time_n, h_n, dep))
    best_f = min(nodes, key=lambda tup: tup[0])[0]
    nodes_with_best_f = [val for val in nodes if val[0] == best_f]
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
        elif priority == "H-DEPTH-L":
            node, sec_tie, num_thr_tie = depth_last(nodes_with_best_f, sec_tie, num_thr_tie)
        elif priority == "H-DEPTH-F":
            node, sec_tie, num_thr_tie = depth_first(nodes_with_best_f, sec_tie, num_thr_tie)

    else:
        node = nodes_with_best_f[0]
    node = node[1]
    open_set.remove(node)
    return open_set, node, num_tie, sec_tie, num_thr_tie

def manhattan(cell, goal):
    x1, y1 = cell
    x2, y2 = goal
    dist = abs(x1 - x2) + abs(y1 - y2)
    return dist

def A_star(graph, start, goal, priority):

    closed_set = []  # nodes already evaluated

    open_set = [start]  # nodes discovered but not yet evaluated

    came_from = {}  # most efficient path to reach from
    time_nodes = {}
    time_nodes[start] = time.time()
    gscore = {}  # cost to get to that node from start

    for key in graph:
        gscore[key] = math.inf  # intialize cost for every node to inf

    gscore[start] = 0
    depth_dict = {}
    depth_dict[start] = -1
    fscore = {}  # cost to get to goal from start node via that node

    for key in graph:
        fscore[key] = math.inf

    fscore[start] = manhattan(start, goal)  # cost for start is only h(x)
    last_f, last_h, last_depth, count_tie, second_tie, counter_space, num_thr_tie = 0, 0, 0, 0, 0, 0, 0
    while open_set:
        open_set, min_node, count_tie, second_tie, num_thr_tie = heappop_adj(open_set, fscore, gscore, priority, count_tie, time_nodes, second_tie, depth_dict, num_thr_tie)

        current = min_node  # set that node to current
        if current == goal:
            return reconstruct_path(came_from, current), counter_space, len(closed_set), count_tie, second_tie, num_thr_tie
        closed_set.append(current)  # add it to set of evaluated nodes

        for neighbor in graph[current]:  # check neighbors of current node
            if neighbor in closed_set:  # ignore neighbor node if its already evaluated
                continue
            if neighbor not in open_set:  # else add it to set of nodes to be evaluated
                open_set.append(neighbor)
                counter_space += 1
                time_nodes[neighbor] = time.time()
            # dist from start to neighbor through current
            tentative_gscore = gscore[current] + 1

            # not a better path to reach neighbor
            if tentative_gscore >= gscore[neighbor]:
                continue
            came_from[neighbor] = current  # record the best path untill now
            gscore[neighbor] = tentative_gscore
            curr_h_score = manhattan(neighbor, goal)
            curr_f_score = gscore[neighbor] + curr_h_score
            fscore[neighbor] = curr_f_score
            if last_f == curr_f_score and curr_h_score == last_h:
                depth = last_depth + 1
            else:
                depth = 0
            last_depth = depth
            last_h = curr_h_score
            last_f = curr_f_score
            depth_dict[neighbor] = depth
    return False


def mat2graph(mat):
    rows = len(mat)
    cols = len(mat[0])
    graph = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if mat[x][y] == True:
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if 0 <= x+dx < rows and 0 <= y+dy < cols and mat[x+dx][y+dy] == True:
                        graph[(x, y)].append((x+dx, y+dy))
    return graph


def write_to_csv(size, p, num_evaluated, time_per_node, num_of_ties, num_steps, overall_time, sec_ties, inx, space_, num_thr_tie):
    dic = {"size": [size], "priority": [p], "expirement_number": [inx], "num_evaluated": [num_evaluated], "time_per_node": [time_per_node],
           "num_of_ties": [num_of_ties], "num_steps": [num_steps], "overall_time": [overall_time], "number_second_tie": [sec_ties], "number_third_tie": [num_thr_tie],  "nodes in memory": space_}
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(os.getcwd() + '/result_maze_man.csv', header=False, mode='a')


def main(size, priority, exp_num):
    start = (0, 0)                        # start <= (size, size)
    destination = (size, size)
    obj = PrimsMaze(size)
    mat = obj.create_maze(start).tolist()

    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(mat, interpolation='nearest')
    # plt.xticks([]), plt.yticks([])

    graph = mat2graph(mat)

    start = (0, 0)
    destination = (size-1, size-1)
    try:
        t_start = perf_counter()
        shortest_route, space, num_expand, num_ties, num_sec_ties, num_thr_tie = A_star(graph, start, destination, priority)
        t_delta = perf_counter() - t_start
    except:
        return
    time_per_node = t_delta / max(num_expand, 1)
    #print(shortest_route)
    for x, y in shortest_route:
        mat[x][y] = -1
    write_to_csv(size, priority, num_expand, time_per_node, num_ties, len(shortest_route), t_delta,
                 num_sec_ties, exp_num, space, num_thr_tie)
    # plt.subplot(1, 2, 2)
    # plt.imshow(mat)
    # plt.show()

#priority_lst = ["LIFO", "FIFO", "H-LIFO", "H-FIFO", "H-G", "G-H", "RANDOM", "H-RAND", "DEPTH-L", "DEPTH-F"]
priority_lst = ["H-DEPTH-L", "H-DEPTH-F"]
maze_sizes = [25, 125, 225, 325]
for si in range(len(maze_sizes)):
    print("-"*25, maze_sizes[si], "-"*25)
    for pi in range(len(priority_lst)):
        print("# ", priority_lst[pi])
        for ex in range(10):
            main(maze_sizes[si], priority_lst[pi], ex)
