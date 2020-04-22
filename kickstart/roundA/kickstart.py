tests = int(input())


def helper(skills, studentNo, candiNo):
    skills.sort()
    # studentNo<candiNo?
    res = temp = sum([skills[candiNo - 1] - i for i in skills[:candiNo - 1]])
    # print('skills: ', skills)
    for i in range(candiNo, studentNo):
        # print('i: ', i)
        # print(temp)

        temp -= skills[i - 1] - skills[i - candiNo]
        # print(temp)
        temp += (skills[i] - skills[i - 1]) * (candiNo - 1)
        # print(temp)
        if temp < res:
            res = temp

    return res


if __name__ == '__main__':
    for i in range(1, tests + 1):
        studentNo, candiNo = [int(s) for s in input().split(" ")]
        skills = [int(s) for s in input().split(" ")]
        # helper(skills, studentNo, candiNo)
        print("Case #%s: %s" % (i, helper(skills, studentNo, candiNo)))

