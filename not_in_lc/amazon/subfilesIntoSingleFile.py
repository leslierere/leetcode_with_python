import heapq
class Solution:
    def minTime(self, numOfSubFiles, files):
        heapq.heapify(files)
        total_time = 0
        for i in range(numOfSubFiles-1):
            file1 = heapq.heappop(files)
            file2 = heapq.heappop(files)
            total_time += file1 + file2
            heapq.heappush(files, file1+file2)

        return total_time

if __name__ == '__main__':
    solution = Solution()
    print(solution.minTime(4, [4,8,6,12])) # should be 58