import heapq
import math
from collections import defaultdict
from typing import Self

import numpy as np


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


class AirportGraph:
    def __init__(self):
        self.n_intersection_nodes: int = 0
        self.n_taxiway_nodes: int = 0
        self.adj_list: dict[int, set[int]] = defaultdict(set)
        self.taxiway_lens: list[float] = []

    def get_other_taxiway_int(self, cur_intersection: int, taxiway: int) -> int:
        return list(self.adj_list[taxiway] - {cur_intersection})[0]

    def shortest_path(self, s: int, t: int, restricted_nodes: set[int] = None) -> tuple[list[int], float]:
        if restricted_nodes is None:
            restricted_nodes = set()

        visited = {s}
        heap = []
        heapq.heappush(heap, (0, [s]))

        while len(heap) > 0:
            cur_dist, cur_path = heapq.heappop(heap)
            cur_int = cur_path[-1]

            shortest_taxiway, shortest_taxiway_len, shortest_taxiway_other_int = None, 999999999, None
            for taxiway_neighbor in self.adj_list[cur_int]:
                other_int = self.get_other_taxiway_int(cur_int, taxiway_neighbor)
                if taxiway_neighbor in restricted_nodes or other_int in restricted_nodes or other_int in visited:
                    continue

                taxiway_len = self.taxiway_lens[taxiway_neighbor - self.n_intersection_nodes]
                if taxiway_len < shortest_taxiway_len:
                    shortest_taxiway = taxiway_neighbor
                    shortest_taxiway_len = taxiway_len
                    shortest_taxiway_other_int = other_int

            if shortest_taxiway is not None:
                new_dist = cur_dist + shortest_taxiway_len
                new_path = cur_path + [shortest_taxiway, shortest_taxiway_other_int]
                if shortest_taxiway_other_int == t:
                    return new_path, new_dist

                visited.add(shortest_taxiway_other_int)
                heapq.heappush(heap, (new_dist, new_path))

        return [], -1.0

    @classmethod
    def KDCA(cls) -> 'Self':
        kdca = AirportGraph()
        inodes: list[tuple[int, int]] = [
            (681, 175),   # 0
            (383, 196),   # 1
            (488, 226),   # 2
            (257, 395),   # 3
            (113, 468),   # 4
            (526, 489),   # 5
            (307, 531),   # 6
            (166, 588),   # 7
            (734, 792),   # 8
            (556, 803),   # 9
            (373, 818),   # 10
            (219, 826),   # 11
            (564, 977),   # 12
            (750, 1006),  # 13
            (764, 1132),  # 14
            (172, 1233),  # 15
        ]
        kdca.n_intersection_nodes = len(inodes)

        taxiways: list[tuple[int, int]] = [
            (0, 5), (0, 8),
            (1, 2), (1, 3),
            (2, 5), (2, 6),
            (3, 4), (3, 6),
            (4, 7),
            (5, 9), (5, 10),
            (6, 10),
            (7, 11),
            (8, 9), (8, 13),
            (9, 10), (9, 12),
            (10, 11), (10, 15),  # (10, 12) to be manually added
            # (11, 15) to be manually added
            (12, 13),  # (12, 14) to be manually added
            (13, 14),
        ]
        extra_tnodes = [
            ((453, 932), 10, 12),
            ((54, 1173), 11, 15),
            ((590, 1115), 12, 14),
        ]

        cur_tnode = 0
        for a, b in taxiways:
            tnode_id = cur_tnode + kdca.n_intersection_nodes
            kdca.adj_list[a].add(tnode_id)
            kdca.adj_list[b].add(tnode_id)
            kdca.adj_list[tnode_id].update([a, b])
            kdca.taxiway_lens.append(dist(inodes[a], inodes[b]))
            cur_tnode += 1

        for tnode, a, b in extra_tnodes:
            tnode_id = cur_tnode + kdca.n_intersection_nodes
            kdca.adj_list[a].add(tnode_id)
            kdca.adj_list[b].add(tnode_id)
            kdca.adj_list[tnode_id].update([a, b])
            kdca.taxiway_lens.append(dist(inodes[a], tnode) + dist(tnode, inodes[b]))
            cur_tnode += 1

        kdca.n_taxiway_nodes = cur_tnode

        return kdca
