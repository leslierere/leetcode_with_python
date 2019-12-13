def maxProduct(nums):
	acc = 1
	large = 1
	small = 1
	for i in nums:
		acc*=i
		large = max(large, acc)


if __name__ == '__main__':
	ls = [-2,2,3,4]