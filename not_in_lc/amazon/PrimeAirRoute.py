# https://leetcode.com/discuss/interview-question/1232603/Prime-Air-Route-or-Amazon-OA-or-Someone-Please-Help
class Solution:
    def findRoute(self, maxTravelDist, forwardRouteList, returnRouteList):
        res = []
        max_sum = 0
        forwardRouteList.sort(key = lambda x: x[1])
        returnRouteList.sort(key = lambda x: x[1])
        # sort 2 lists, and 2 pointers
        pt1 = 0
        pt2 = len(returnRouteList) - 1

        while pt1 < len(forwardRouteList) and pt2 >= 0:
            if forwardRouteList[pt1][1] + returnRouteList[pt2][1] > maxTravelDist:
                pt2 -= 1
            else:
                if forwardRouteList[pt1][1] + returnRouteList[pt2][1] >= max_sum:
                    if forwardRouteList[pt1][1] + returnRouteList[pt2][1] > max_sum:
                        res = []
                        max_sum = forwardRouteList[pt1][1] + returnRouteList[pt2][1]

                    res.append([forwardRouteList[pt1][0], returnRouteList[pt2][0]])
                    index = pt2 - 1
                    while index >= 0 and returnRouteList[index][1] == returnRouteList[index+1][1]:
                        res.append([forwardRouteList[pt1][0], returnRouteList[index][0]])
                        index -= 1
                    pt1 += 1

        return res


if __name__ == '__main__':
    solution = Solution()
    maxTravelDist = 7000
    forwardRouteList = [[1, 2000], [2, 4000], [3, 6000]]
    returnRouteList = [[1, 2000]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList)) #[[2,1]]
    maxTravelDist = 10000
    forwardRouteList = [[1, 3000], [2, 5000], [3, 7000], [4, 10000]]
    returnRouteList = [[1, 2000], [2, 3000], [3, 4000], [4, 5000]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[2, 4], [3, 2]]
    maxTravelDist = 10000
    forwardRouteList = [[1, 3000], [2, 5000], [3, 7000], [4, 5000]]
    returnRouteList = [[1, 2000], [2, 3000], [3, 4000], [4, 5000]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[2, 4], [3, 2], [4, 4]]
    maxTravelDist = 10000
    forwardRouteList = [[1, 3000], [2, 5000], [3, 7000], [4, 10000]]
    returnRouteList = [[1, 5000], [2, 3000], [3, 4000], [4, 5000]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[2, 4], [3, 2], [2, 1]]
    maxTravelDist = 7
    forwardRouteList = [[1, 2], [2, 4], [3, 6]]
    returnRouteList = [[1, 2]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[2, 1]]
    maxTravelDist = 10
    forwardRouteList = [[1, 3], [2, 5], [3, 7], [4, 10]]
    returnRouteList = [[1, 2], [2, 3], [3, 4], [4, 5]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[2, 4], [3, 2]]
    maxTravelDist = 20
    forwardRouteList = [[1, 8], [2, 7], [3, 14]]
    returnRouteList = [[1, 5], [2, 10], [3, 14]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[3, 1]]
    maxTravelDist = 20
    forwardRouteList = [[1, 8], [2, 15], [3, 9]]
    returnRouteList = [[1, 8], [2, 11], [3, 12]]
    print(solution.findRoute(maxTravelDist, forwardRouteList, returnRouteList))  # [[1, 3], [3, 2]]

