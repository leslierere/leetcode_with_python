class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        di, dj = 0, 1

        i, j = (0, 0)
        walked = set()

        for instruction in instructions:
            if instruction == "L":
                if di == 0 and dj == 1:
                    di, dj = 1, 0
                elif di == 1 and dj == 0:
                    di, dj = 0, -1
                elif di == 0 and dj == -1:
                    di, dj = -1, 0
                else:
                    di, dj = 0, 1
            elif instruction == "R":
                if di == 0 and dj == 1:
                    di, dj = -1, 0
                elif di == -1 and dj == 0:
                    di, dj = 0, -1
                elif di == 0 and dj == -1:
                    di, dj = 1, 0
                else:
                    di, dj = 0, 1
            else:
                i, j = i + di, j + dj

        return (i == 0 and j == 0) or di != 0 or dj != 1

        return False