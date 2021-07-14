import collections
import heapq
class Solution:
    def getTimes(self, time, direction):
        n = len(time)
        time_dic = dict()

        for index in range(n):
            start = time[index]
            turn = direction[index]
            if start not in time_dic:
                time_dic[start] = []
                heapq.heapify(time_dic[start])
            heapq.heappush(time_dic[start], (turn, index))

        max_time = time[-1]
        res = [0 for i in range(n)]
        last_action = 1
        i = 0
        while len(time_dic) != 0:

            if i not in time_dic:
                last_action = 1
            else:
                cur_heap = time_dic[i]
                if len(cur_heap) == 1:
                    turn, index = heapq.heappop(cur_heap)
                    res[index] = i
                    last_action = turn
                else:
                    if i+1 not in time_dic:
                        time_dic[i+1] = []
                        heapq.heapify(time_dic[i+1])
                    if last_action == 1: # exit or not used last time
                        while len(cur_heap)!=0:
                            turn, index = heapq.heappop(cur_heap)
                            if turn == 1:
                                res[index] = i
                                last_action = 1
                                while len(cur_heap) != 0:
                                    turn, index = heapq.heappop(cur_heap)
                                    heapq.heappush(time_dic[i+1], (turn, index))
                            else:
                                heapq.heappush(time_dic[i + 1], (turn, index))
                    else:
                        while len(cur_heap) != 0:
                            turn, index = heapq.heappop(cur_heap)
                            if turn == 0:
                                res[index] = i
                                last_action = 0
                                while len(cur_heap) != 0:
                                    turn, index = heapq.heappop(cur_heap)
                                    heapq.heappush(time_dic[i+1], (turn, index))
                            else:
                                res[index] = i
                                last_action = 1
                                while len(cur_heap) != 0:
                                    turn, index = heapq.heappop(cur_heap)
                                    heapq.heappush(time_dic[i + 1], (turn, index))

                time_dic.pop(i)
            i += 1


        return res


if __name__ == '__main__':
    solution = Solution()
    time = [0, 0, 1, 5]
    direction = [0, 1, 1, 0]
    print(solution.getTimes(time, direction)) # should be [2, 0, 1, 5]





