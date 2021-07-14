import collections
class Solution:
    def parse(self, logData, threshold):
        user_count = collections.defaultdict(int)

        for data in logData:
            id1, id2, transaction = data.split(" ")
            if id1 == id2:
                user_count[id1] += 1
            else:
                user_count[id1] += 1
                user_count[id2] += 1

        return sorted([int(id) for id in user_count if user_count[id]>=threshold], reverse=True)

if __name__ == '__main__':
    solution = Solution()
    logData = ["345366 89921 45", "029323 38239 23", "029323 38239 77", "345366 38239 23", "029323 345366 13", "38239 38239 23"]
    print(solution.parse(logData, 3)) # should be [345366 , 38239, 029323]

