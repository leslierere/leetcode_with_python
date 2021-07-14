class Solution:
    def earliestTime(self, n, openTimes, deliveryTimeCost):
        deliveryTimeCost.sort(reverse = True)
        complete_time = 0
        slot = 0
        for time in openTimes:

            complete_time = max(complete_time, time + deliveryTimeCost[slot*4])
            slot += 1

        return complete_time


if __name__ == '__main__':
    solution = Solution()
    n = 2
    openTimes = [8, 10]
    deliveryTimeCost = [2,2,3,1,8,7,4,5]
    print(solution.earliestTime(n, openTimes, deliveryTimeCost)) # should be 16