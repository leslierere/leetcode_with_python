import collections

class Solution:
    def findProvincesNum(self, isConnected):
        neighbors = collections.defaultdict(set)
        n = len(isConnected)
        for node1 in range(n):
            for node2 in range(node1+1, n):
                if isConnected[node1][node2]:
                    neighbors[node1].add(node2)
                    neighbors[node2].add(node1)

        visited = [False for i in range(n)]
        provinces = 0
        for i in range(n):
            if not visited[i]:
                provinces += 1
                self.dfs(i, visited, neighbors)

        return provinces

    def dfs(self, node, visited, neighbors):
        if visited[node]:
            return

        visited[node] = True
        for neighbor in neighbors:
            self.dfs(neighbor, visited, neighbors)


if __name__ == '__main__':
    solution = Solution()
    matrix = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    print(solution.findProvincesNum(matrix)) # should be 2
