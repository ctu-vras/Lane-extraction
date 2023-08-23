from collections import defaultdict

import numpy as np
from pytorch3d.ops import knn_points


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # undirected graph

    def DFSUtil(self, v, visited):
        visited.add(v)
        app_array = [v]
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                app_array.extend(self.DFSUtil(neighbour, visited))
        return app_array

    def DFS(self):
        visited = set()
        connected_lines = []
        for i in range(len(self.graph)):
            if i not in visited:
                connected_lines.append(self.DFSUtil(i, visited))
        return connected_lines

    def create_graph(self, padded_centers_array, padded_vectors_directions, outreach_mask):
        num_centers = padded_centers_array.shape[0]
        vectors_number = padded_vectors_directions.shape[0]
        for i in range(vectors_number):
            if outreach_mask[i].item() is False:
                continue
            mask = np.ones(num_centers, dtype=bool)
            mask[i] = False
            new_centers_array = padded_centers_array[mask].clone().detach()
            new_centers_array = new_centers_array.reshape(1, -1, 2)
            centers_array_i = (padded_centers_array[i] + padded_vectors_directions[i]).reshape(1, -1, 2)
            _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
            idx = idx[0][0][0].item()
            if idx >= i:
                idx += 1
            self.addEdge(i, idx)
