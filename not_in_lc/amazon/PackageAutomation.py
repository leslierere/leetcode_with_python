class Solution:
    def package(self, numGroups, arr):
        return min(numGroups, max(arr))

if __name__ == '__main__':
    solution = Solution()
    numGroups = 4
    arr = [3, 1, 3, 4]
    print(solution.package(numGroups, arr)==4)
    arr = [1, 3, 2, 2]
    print(solution.package(numGroups, arr) == 3)