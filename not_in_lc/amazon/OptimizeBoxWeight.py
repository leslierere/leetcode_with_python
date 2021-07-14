class Solution:
    # from the back to front
    def optimize(self, nums):
        if len(nums) == 1:
            return nums
        nums.sort()
        subsum = [0 for _ in range(len(nums))]
        subsum[0] = nums[0]

        for i in range(1, len(nums)):
            subsum[i] = subsum[i-1] + nums[i]
        total_sum = subsum[-1]
        for i in range(len(subsum)):
            if subsum[i]*2 >= total_sum:
                return nums[i:]

if __name__ == '__main__':
    nums1 = [4, 5, 2, 3, 1, 2]
    solution1 = Solution()
    print(solution1.optimize(nums1))#[4, 5]
    nums2 = [10, 5, 3, 1, 20]
    print(solution1.optimize(nums2))#[20]
    nums3 = [1, 2, 3, 5, 8]
    print(solution1.optimize(nums3))#[5, 8]
    nums4 = [2, 2, 2, 2]
    print(solution1.optimize(nums4))  # [2, 2, 2]