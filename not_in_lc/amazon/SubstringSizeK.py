import collections

class Solution:
    def uniqueSubstringSizeK(self, s, k):
        counter = collections.Counter(s[:k-1])
        result = set()

        for i in range(k-1, len(s)):
            char = s[i]
            counter[char] += 1
            if len(counter) == k:
                result.add(s[i-k+1:i+1])
            first_char = s[i-k+1]
            if counter[first_char] == 1:
                counter.pop(first_char)
            else:
                counter[first_char] -= 1

        return list(result)


if __name__ == '__main__':
    solution = Solution()
    s = "abcabc"
    k = 3
    print(solution.uniqueSubstringSizeK(s, k)) # should be ["abc", "bca", "cab"]
    s = "abacab"
    k = 3
    print(solution.uniqueSubstringSizeK(s, k)) # ["bac", "cab"]
    s = "awaglknagawunagwkwagl"
    k = 4
    print(solution.uniqueSubstringSizeK(s, k)) # ["wagl", "aglk", "glkn", "lkna", "knag", "gawu", "awun", "wuna", "unag", "nagw", "agwk", "kwag"]