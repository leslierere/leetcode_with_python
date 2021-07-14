class Solution:
    def countSwap(self, array):
        times = 0

        for i in range(1, len(array)):
            for j in range(i):
                if array[i] < array[j]:
                    array[i], array[j] = array[j], array[i]
                    times += 1

        return times

if __name__ == '__main__':
    solution = Solution()
    input = [5, 4, 1, 2]
    print(solution.countSwap(input)) # should be 5