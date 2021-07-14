class Solution:
    def evalRPN(self, tokens) -> int:
        stack = []

        for token in tokens:
            if token == "+":
                stack.append(stack.pop() + stack.pop())
            elif token == "-":
                second = stack.pop()
                first = stack.pop()
                stack.append(first - second)
            elif token == "*":
                stack.append(stack.pop() * stack.pop())
            elif token == "/":
                second = stack.pop()
                first = stack.pop()
                if first * second < 0:
                    stack.append(-(abs(first) // abs(second)))
                else:
                    stack.append(first // second)
            else:
                stack.append(int(token))

        return stack[-1]