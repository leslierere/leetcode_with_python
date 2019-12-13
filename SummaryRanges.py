def summaryRanges(nums):
	res = []
	for i in nums:
		if not res or res[-1][-1]+1!=i:
			res+=[],

		res[-1][1:]=i,
	
	return ["->".join(map(str, x)) for x in res]


def main():
	nums = [0,1,2,4,5,7]
	ls = summaryRanges(nums)
	# print(ls)

if __name__ =='__main__':
	main()