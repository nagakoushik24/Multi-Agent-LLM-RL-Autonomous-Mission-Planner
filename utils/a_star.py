# utils/a_star.py
import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    """
    grid: 2D array where 0 free, 1 obstacle
    start/goal: (r,c)
    returns list of coords from start->goal inclusive or [] if no path
    """
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    while open_set:
        _, g, current, parent = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = [current]
            while parent:
                path.append(parent)
                parent = came_from.get(parent)
            return list(reversed(path))
        if current in came_from:
            continue
        came_from[current] = parent
        r,c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr,nc] != 1:
                tentative = g + 1
                neigh = (nr,nc)
                if tentative < gscore.get(neigh, 1e9):
                    gscore[neigh] = tentative
                    heapq.heappush(open_set, (tentative + heuristic(neigh, goal), tentative, neigh, current))
    return []
