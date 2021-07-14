import collections
class Solution:
    def shortestRoute(self, island):
        queue = collections.deque()
        queue.append((0, 0))
        walked = set()
        walked.add((0, 0))
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
                        queue.append((i+di, j+dj))
                        cur_walked.add((i+di, j+dj))
            walked = walked|cur_walked
            steps += 1

        return -1

if __name__ == '__main__':
    solution = Solution()
    island = [['O', 'O', 'O', 'O'], ['D', 'O', 'D', 'O'], ['O', 'O', 'O', 'O'], ['X', 'D', 'D', 'O']]
    print(solution.shortestRoute(island)) # should be 5