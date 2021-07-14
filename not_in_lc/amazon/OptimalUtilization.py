class Solution:
    def findPair(self, a, b, target):
        a.sort(key=lambda x: x[1])
        b.sort(key=lambda x: x[1])
        res = []
        max_sum = float("-inf")

        for idx_a, num_a in a:
            if num_a > target:
                break
            for idx_b, num_b in b:
                if num_a + num_b > target:
                    break
                if num_a + num_b == max_sum:
                    res.append([idx_a, idx_b])
                elif num_a + num_b > max_sum:
                    res = [[idx_a, idx_b]]
                    max_sum = num_a + num_b

        return res

if __name__ == '__main__':
    a = [[1, 2], [2, 4], [3, 6]]
    b = [[1, 2]]
    target = 7
    solution = Solution()
    print(solution.findPair(a, b, target)) # should be [[2, 1]]
