class Solution:
    def cutOff(self, cutOffRank, num, scores):
        if cutOffRank <= 0:
            return 0
        scores.sort(reverse=True)
        if scores[0] == 0:
            return 0

        rank = 1
        for i in range(1, num):
            cur_score = scores[i]
            if cur_score != scores[i-1]:
                rank = i + 1
            if rank > cutOffRank or cur_score == 0:
                return i

        return num

if __name__ == '__main__':
    solution = Solution()
    cutOffRank = 3
    num = 4
    scores = [100, 50, 50, 25]
    print(solution.cutOff(cutOffRank, num, scores)==3) # should be 3
    scores = [100, 100, 100, 100]
    print(solution.cutOff(cutOffRank, num, scores) == 4)
    scores = [100, 50, 50, 25]
    print(solution.cutOff(1, num, scores) == 1)