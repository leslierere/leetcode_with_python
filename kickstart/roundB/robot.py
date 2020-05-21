from collections import Counter


def helper(program):
    stack = []
    east = 0
    south = 0

    for i in program:
        if i.isdigit():
            stack.append(int(i))
        elif i == '(':
            stack.append("")
        elif i.isalpha():
            if len(stack)==0:
                if i == 'S':
                    south += 1
                elif i == 'N':
                    south -= 1
                elif i == 'E':
                    east += 1
                else:
                    east -= 1
            else:
                stack[-1] += i
        else:
            string = stack.pop()
            number = stack.pop()
            if len(stack) == 0:
                counter = Counter(string)
                for key, val in counter.items():
                    if key == 'S':
                        south += number * val
                    elif key == 'N':
                        south -= number * val
                    elif key == 'E':
                        east += number * val
                    else:
                        east -= number * val
            else:
                stack[-1]+=number*string

    w = (east+1)%10**9
    h = (south+1)%10**9

    if w==0:
        w = 10**9
    if h==0:
        h = 10**9
    return w,h




if __name__ == '__main__':
    cases = int(input())

    for i in range(1, cases + 1):
        program = input()

        w, h = helper(program)
        print("Case #%s: %s %s" % (i, w, h))
