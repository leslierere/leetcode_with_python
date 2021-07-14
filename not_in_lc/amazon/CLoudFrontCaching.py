import collections
class Solution:
    def costEvaluation(self, n, connections):
        neighbors = collections.defaultdict(set)
        for house1, house2 in connections:
            neighbors[house1].add(house2)
            neighbors[house2].add(house1)

        visited = [False for i in range(n)]
        cost = 0
        for i in range(n):
            total = [0]
            if not visited[i]:
                self.dfs(i, neighbors, visited, total)
                nodes_count = total[0]
                if nodes_count == 1:
                    cost += 1
                else:
                    cost += int(nodes_count**0.5) + 1

        return cost


    def dfs(self, node, neighbors, visited, total):
        if visited[node]:
            return

        total[0] += 1
        visited[node] = True
        while neighbors[node]:
            neighbor = neighbors[node].pop()
            neighbors[neighbor].remove(node)
            self.dfs(neighbor, neighbors, visited, total)


if __name__ == '__main__':
    n = 4
    connections = [[0, 2], [1, 2]]
    solution = Solution()
    print(solution.costEvaluation(n, connections)) # should be 3
    n = 10
    connections = [[2, 6], [3, 5], [0, 1], [2, 9], [5, 6]]
    print(solution.costEvaluation(n, connections)) # should be 8