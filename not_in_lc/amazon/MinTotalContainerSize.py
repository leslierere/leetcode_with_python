class Solution:
    def minContainerSize(self, boxSizes, days):
        if len(boxSizes) < days:
            return -1
        dp = [[0 for i in range(len(boxSizes))] for j in range(days)]
        dp[0][0] = boxSizes[0]
        for j in range(1, len(dp[0])):
            dp[0][j] = max(dp[0][j-1], boxSizes[j])

        for i in range(1, len(dp)): # i = 1
            day = i+1 # day = 2
            for end_box in range(0, day - 1):
                dp[i][end_box] = boxSizes[end_box]
            for end_box in range(day-1, len(dp[0])):
                dp[i][end_box] = dp[i-1][end_box-1] + boxSizes[end_box]
                for start_box in range(day-1, end_box+1):
                    right_max = max(boxSizes[start_box:end_box+1])
                    dp[i][end_box] = min(dp[i][end_box], dp[i-1][start_box-1] + right_max)


        return dp[-1][-1]

if __name__ == '__main__':
    solution = Solution()
    p = [10, 2, 20, 5, 15, 10, 1]
    d = 3
    print(solution.minContainerSize(p, d)) # should be 31
