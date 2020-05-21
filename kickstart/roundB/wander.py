def helper(W, H, L, U, R, D):
    dp = [0.5**i for i in range(W)]
    res = 0

    for i in range(1, H):
        nextRow = [0]*W

        for j in range(0, W):
            if j==0:
                nextRow[j] = dp[0] * 0.5
            else:
                nextRow[j] = nextRow[j-1]*0.5 + dp[j]*0.5
            if i==U-1 and j==L-1:
                res+=nextRow



        dp = nextRow


    return res


if __name__ == '__main__':
    cases = int(input())
    # peaks = []
    for i in range(1, cases + 1):
        W, H, L, U, R, D = [int(s) for s in input().split(" ")]

        p = helper(W, H, L, U, R, D)
        print("Case #%s: %.4f" % (i, p))
