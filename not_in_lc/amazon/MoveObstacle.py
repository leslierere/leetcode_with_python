import collections

class Solution:
    def move(self, building):
        queue = collections.deque()
        queue.append((0, 0))
        steps = 0
        rows = len(building)
        cols = len(building[0])

        # -1 means visited
        while queue:
            length = len(queue)
            steps += 1
            for _ in range(length):
                i, j = queue.popleft()
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if i + di < rows and i + di >= 0 and j + dj < cols and j + dj >= 0 and building[i+di][j+dj] not in [-1, 0]:
                        if building[i+di][j+dj] == 9:
                            self.recover(building)
                            return steps
                        building[i + di][j + dj] = -1
                        queue.append((i+di, j+dj))

        self.recover(building)
        return -1

    def recover(self, building):
        for i in range(len(building)):
            for j in range(len(building[0])):
                if building[i][j] == -1:
                    building[i][j] = 1

if __name__ == '__main__':
    solution = Solution()
    building = [[1, 1, 1], [1, 0, 1], [1, 1, 9]]
    print(solution.move(building)) # should be 4
    building = [[1,1,1], [1,0,0],[1,0,9]]
    print(solution.move(building))  # should be -1