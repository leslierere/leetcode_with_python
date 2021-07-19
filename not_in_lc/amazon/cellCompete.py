class Solution:
    def cellCompete(self, states, days):
        # WRITE YOUR CODE HERE
        # length is 1
        temp = [0 for i in range(len(states))]
        for day in range(days):
            if states[1] == 1:
                temp[0] = 1
            else:
                temp[0] = 0
            if states[-2] == 1:
                temp[-1] = 1
            else:
                temp[-1] = 0
            for i in range(1, len(states) - 1):
                temp[i] = 1 if states[i - 1] ^ states[i + 1] else 0

            states, temp = temp, states

        return states


if __name__ == '__main__':
    arr = [1,1,1,0,1,1,1,1]
    days = 2
    solution = Solution()
    print(solution.cellCompete(arr, days)) # should be [0,0,0,0,1,1,0]
    arr = [1, 1]
    print(solution.cellCompete(arr, 3)) # should be [1, 1]
