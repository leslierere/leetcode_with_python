class Solution:
    def breakPalindrome(self, s):
        if len(s)<2:
            return ""
        mid = len(s)//2 # len = 3 -> mid = 1, len = 4 -> mid = 2

        for i in range(mid):
            char = s[i]
            if char!="a":
                return s[:i] + "a" + s[i+1:]


        return s[:-1] + "b"


if __name__ == '__main__':
    solution = Solution()
    print(solution.breakPalindrome("abccba")) # should be "aaccba"
    print(solution.breakPalindrome("aaaa"))