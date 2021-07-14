class Solution:
    def package(self, numGroups, arr):
        arr.sort()
        arr[0] = 1

        for i in range(1, len(arr)):
            arr[i] = min(arr[i-1]+1, arr[i])

        return arr[-1]

if __name__ == '__main__':
    solution = Solution()
    numGroups = 4
    arr = [3, 1, 3, 4]
    print(solution.package(numGroups, arr)==4)
    arr = [1, 3, 2, 2]
    print(solution.package(numGroups, arr) == 3)