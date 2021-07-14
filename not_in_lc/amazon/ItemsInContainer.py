class Solution:
    def noItems(self, s:str, startIndices, endIndices):
        dp = [[0 for i in range(len(s))] for j in range(len(s))]

        first_idx = s.find("|")
        if first_idx == -1:
            return [0 for i in range(len(startIndices))]
        second_idx = s.find("|", first_idx+1)
        if second_idx == -1:
            return [0 for i in range(len(startIndices))]

        dp[first_idx][second_idx] = second_idx - first_idx - 1

        for start in range(len(s)-2):
            for end in range(start+1, len(s)):
                if s[end] == "|":
                    for j in range(end-1, start-1, -1): # store before
                        if s[j] == "|":
                            dp[start][end] = dp[start][j] + end - j -1
                            break
                else:
                    dp[start][end] = dp[start][end-1]

        res = []
        for i in range(len(startIndices)):
            start = startIndices[i] - 1
            end = endIndices[i] - 1
            res.append(dp[start][end])

        return res

if __name__ == '__main__':
    solution = Solution()
    s = '|**|*|*'
    startIndices = [1, 1]
    endIndices = [5, 6]
    print(solution.noItems(s, startIndices, endIndices)) # should be [2,3].


