class Solution:
    def droppedRequests(self, requestTime):
        max_seconds = requestTime[-1]
        count = [0 for i in range(max_seconds+1)]

        last_ten = 0
        last_sixty = 0
        dropped = 0
        for second in requestTime:
            count[second] += 1

        for second in range(1, max_seconds+1):
            if second - 10 > 0:
                last_ten -= count[second - 10]
            if second - 60 > 0:
                last_sixty -= count[second - 60]
            if count[second] == 0:
                continue
            requests = count[second]
            requests -= max(requests - 3, 0)
            requests -= max(last_ten + requests - 20, 0)
            requests -= max(last_sixty + requests - 60, 0)
            dropped += count[second] - requests
            last_sixty += count[second]
            last_ten += count[second]


        return dropped

if __name__ == '__main__':
    solution = Solution()
    requestTime = [1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,7,11,11,11,11]
    print(solution.droppedRequests(requestTime)) # should be 7