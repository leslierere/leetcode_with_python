def helper(heights):
    res = 0
    for i in range(1, len(heights) - 1):
        if heights[i] > heights[i - 1] and heights[i] > heights[i + 1]:
            res += 1

    return res


if __name__ == '__main__':
    cases = int(input())
    # peaks = []
    for i in range(1, cases + 1):
        mountains = int(input())
        heights = [int(s) for s in input().split(" ")]
        peak=helper(heights)
        print("Case #%s: %s" % (i, peak))
