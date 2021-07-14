class Solution:
    def findMusic(self, rideDuration, songDuration):
        rideDuration -= 30
        max_len = -1
        seen = dict()
        res = [-1, -1]
        for i, song in enumerate(songDuration):
            if rideDuration-song in seen:
                temp_max = max(rideDuration-song, song)
                if temp_max > max_len:
                    max_len = temp_max
                    res = [seen[rideDuration-song], i]
            seen[song] = i

        return res

if __name__ == '__main__':
    solution = Solution()
    rideDuration = 90
    songDuration = [1, 10, 25, 35, 60]
    print(solution.findMusic(rideDuration, songDuration)) # should be [2, 3]
    songDuration = [1, 10, 25, 35, 50]
    print(solution.findMusic(rideDuration, songDuration))  # should be [1, 4]
    rideDuration = 30
    songDuration = [0, 0]
    print(solution.findMusic(rideDuration, songDuration))  # should be [0, 1]