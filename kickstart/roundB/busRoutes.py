def helper(day, schedule):
    bus = len(schedule) - 1

    while day>0:
        day = (day//schedule[bus])*schedule[bus]
        while day%schedule[bus]==0:
            bus -= 1
            if bus==-1:
                return day


if __name__ == '__main__':
    cases = int(input())

    for i in range(1, cases + 1):
        buses, day = [int(s) for s in input().split(" ")]
        schedule = [int(s) for s in input().split(" ")]
        start = helper(day, schedule)
        print("Case #%s: %s" % (i, start))
