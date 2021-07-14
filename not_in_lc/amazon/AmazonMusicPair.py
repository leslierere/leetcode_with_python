class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        number = 0
        modular_count = collections.defaultdict(int)
        for song in time:
            remain = song % 60
            modular_count[remain] += 1

        for length in range(1, 30):
            if length in modular_count and 60 - length in modular_count:
                number += modular_count[length] * modular_count[60 - length]
        if 0 in modular_count and modular_count[0] > 1:
            number += (modular_count[0] * (modular_count[0] - 1)) // 2
        if 30 in modular_count and modular_count[30] > 1:
            number += (modular_count[30] * (modular_count[30] - 1)) // 2

        return number