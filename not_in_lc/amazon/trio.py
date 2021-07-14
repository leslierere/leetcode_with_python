class Solution:
    def minTrioDegree(self, n: int, edges: List[List[int]]) -> int:
        neighbors = collections.defaultdict(set)
        for node1, node2 in edges:
            neighbors[node1].add(node2)
            neighbors[node2].add(node1)

        visited = [0 for i in range(n + 1)]
        min_degree = float("inf")
        trios = set()

        for i in range(1, n + 1):
            if not visited[i]:
                self.dfs(i, visited, neighbors, min_degree, trios)

        for node1, node2, node3 in trios:
            min_degree = min(min_degree, len(neighbors[node1] + len(neighbors[node2]) + len(neighbors[node3]) - 6))

        if min_degree == float("inf"):
            return -1
        return min_degree

    def dfs(self, node, visited, neighbors, min_degree, trios):
        if visited[node]:
            return

        visited[node] = True
        for neighbor in neighbors[node]:
            common_nbrs = neighbors[node] & neighbors[neighbor]
            for other in common_nbrs:
                trios.add(tuple(sorted([node, neighbor, other])))
            self.dfs(neighbor, visited, neighbors, min_degree, trios)

