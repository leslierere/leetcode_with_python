def helper(diffs, addNo):
    if not addNo or isSmooth(diffs):
        return max(diffs)

    maxIndex, curMax = 0, 0
    for index, diff in enumerate(diffs):
        if diff>curMax:
            maxIndex, curMax = index, diff
    insert1 = curMax//2
    insert2 = curMax-insert1
    newDiffs = diffs[:maxIndex]+[insert1, insert2] + diffs[maxIndex+1:]
    return helper(newDiffs, addNo-1)



def isSmooth(diffs):
    return sum(diffs)==len(diffs)



if __name__ == '__main__':
    tests = int(input())
    for i in range(1, tests + 1):
        sessionNo, addNo = [int(s) for s in input().split(" ")]

        sessions = [int(s) for s in input().split(" ")]
        diffs = [sessions[i]-sessions[i-1] for i in range(1, len(sessions))]
        if not diffs:
            print("Case #%s: %s" % (i, 0))
        else:

            print("Case #%s: %s" % (i, helper(diffs, addNo)))