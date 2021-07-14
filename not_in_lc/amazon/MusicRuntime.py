class Solution:
    def findMusic(self, rideDuration, songDuration):
        rideDuration -= 30
        index_map = dict()
        for i in range(len(songDuration)):
            song = songDuration[i]
            index_map[song] = i

        songDuration.sort()
        left = 0
        right = len(songDuration) - 1

        while left < right:
            total_len = songDuration[left] + songDuration[right]
            if total_len < rideDuration:
                left += 1
            elif total_len > rideDuration:
                right -= 1
            else:
                return [index_map[songDuration[left]], index_map[songDuration[right]]]

        return [-1, -1]

if __name__ == '__main__':
    solution = Solution()
    rideDuration = 90
    songDuration = [1, 10, 25, 35, 60]
    print(solution.findMusic(rideDuration, songDuration)) # should be [2, 3]