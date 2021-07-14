import collections

class Solution:
    def shortestRoute(self, island):
        min_len = len(island)*len(island[0])
        for i in range(len(island)):
            for j in range(len(island[0])):
                if island[i][j] == "S":
                    steps = self.helper(island, i, j)
                    if steps != -1:
                        min_len = min(min_len, steps)

        return min_len

    def helper(self, island, start_i, start_j):
        queue = collections.deque()
        walked = set()
        walked.add((start_i, start_j))
        queue.append((start_i, start_j))
        rows = len(island)
        cols = len(island[0])
        steps = 0

        while queue:
            length = len(queue)
            cur_walked = set()
            for _ in range(length):
                i, j = queue.popleft()
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if i + di < rows and i+di >= 0 and j + dj < cols and j+dj >= 0 and (i+di, j+dj) not in walked and island[i+di][j+dj] != "D":
                        if island[i+di][j+dj] == "X":
                            return steps + 1
                        cur_walked.add((i + di, j + dj))
                        queue.append((i+di, j+dj))
            walked = walked|cur_walked
            steps += 1
        return -1

if __name__ == '__main__':
    solution = Solution()
    island = [['S', 'O', 'O', 'S', 'S'], ['D', 'O', 'D', 'O', 'D'], ['O', 'O', 'O', 'O', 'X'], ['X', 'D', 'D', 'O', 'O'], ['X', 'D', 'D', 'D', 'O']]
    print(solution.shortestRoute(island)) # should be 3