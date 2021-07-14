import collections
class Solution:
    def getTimes(self, time, direction):
        n = len(time)

        exit_queue = collections.deque()
        enter_queue = collections.deque()

        for i in range(n):
            if direction[i] == 1:
                exit_queue.append((time[i], i))
            else:
                enter_queue.append((time[i], i))

        last_action = 1
        res = [0 for i in range(n)]

        cur_time = 0
        while exit_queue and enter_queue:
            exit_time = exit_queue[0][0]
            enter_time = enter_queue[0][0]

            if exit_time <= cur_time and enter_time <= cur_time:
                if last_action == 1:
                    exit_time, idx = exit_queue.popleft()
                    res[idx] = cur_time
                else:
                    enter_time, idx = enter_queue.popleft()
                    res[idx] = cur_time
            elif exit_time <= cur_time:
                exit_time, idx = exit_queue.popleft()
                res[idx] = cur_time
                last_action = 1
            elif enter_time <= cur_time:
                enter_time, idx = enter_queue.popleft()
                res[idx] = cur_time
                last_action = 0
            else:
                last_action = 1

            cur_time += 1

        while exit_queue:
            exit_time = exit_queue[0][0]
            if exit_time <= cur_time:
                exit_time, idx = exit_queue.popleft()
                res[idx] = cur_time
            cur_time += 1

        while enter_queue:
            enter_time = enter_queue[0][0]
            if enter_time <= cur_time:
                enter_time, idx = enter_queue.popleft()
                res[idx] = cur_time
            cur_time += 1

        return res

if __name__ == '__main__':
    solution = Solution()
    time = [0, 0, 1, 5]
    direction = [0, 1, 1, 0]
    print(solution.getTimes(time, direction)) # should be [2, 0, 1, 5]





