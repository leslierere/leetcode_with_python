##basics

8.5

#### 28. Implement strStr()

https://leetcode.com/problems/implement-strstr/





#### 14. Longest Common Prefix

https://leetcode.com/problems/longest-common-prefix/



#### 58. Length of Last Word

https://leetcode.com/problems/length-of-last-word/





#### 8.8

#### 383. Ransom Note

https://leetcode.com/problems/ransom-note/





#### 344. Reverse String

https://leetcode.com/problems/reverse-string/





#### 151. Reverse Words in a String

https://leetcode.com/problems/reverse-words-in-a-string/

- solution

  尝试O(1) space-not done
  
  

#### 186. Reverse Words in a String II

https://leetcode.com/problems/reverse-words-in-a-string-ii/



#### 345. Reverse Vowels of a String

https://leetcode.com/problems/reverse-vowels-of-a-string/



#### 293. Flip Game-easy

https://leetcode.com/problems/flip-game/

8.13

####  294. Flip Game II

https://leetcode.com/problems/flip-game-ii/

* Solution-minimax&backtracking

  ***worth thinking and doing***



8.21

### 49. Group Anagrams

https://leetcode.com/problems/group-anagrams/



### 249. Group Shifted Strings

https://leetcode.com/problems/group-shifted-strings/



8.22

### 87. Scramble String

https://leetcode.com/problems/scramble-string/

* Solution-recursion

* Solution-DP

  ***worth doing and thinking***



8.26

#### 161. One Edit Distance

https://leetcode.com/problems/one-edit-distance/



#### 38. Count and Say

https://leetcode.com/problems/count-and-say/



8.29

#### 358. Rearrange String k Distance Apart

https://leetcode.com/problems/rearrange-string-k-distance-apart/

* solution-Priority queue



#### 316. Remove Duplicate Letters

https://leetcode.com/problems/remove-duplicate-letters/

* solution-Stack, greedy



#### 271. Encode and Decode Strings

https://leetcode.com/problems/encode-and-decode-strings/

* Solution-good thoughts



8.30

#### 168. Excel Sheet Column Title

https://leetcode.com/problems/excel-sheet-column-title/





#### 171. Excel Sheet Column Number

https://leetcode.com/problems/excel-sheet-column-number/

* Easy



#### 13. Roman to Integer

https://leetcode.com/problems/roman-to-integer/





9.14

#### 12. Integer to Roman

https://leetcode.com/problems/integer-to-roman/



#### 273. Integer to English Words

https://leetcode.com/problems/integer-to-english-words/

* solution

  感觉如下分组更好

  ```python
  def words(n):
          if n < 20:
              return to19[n-1:n]
          if n < 100:
              return [tens[n/10-2]] + words(n%10)
          if n < 1000:
              return [to19[n/100-1]] + ['Hundred'] + words(n%100)
          for p, w in enumerate(('Thousand', 'Million', 'Billion'), 1):
              if n < 1000**(p+1):
                  return words(n/1000**p) + [w] + words(n%1000**p)
      return ' '.join(words(num)) or 'Zero'
  ```





9.21

#### 247. Strobogrammatic Number II

https://leetcode.com/problems/strobogrammatic-number-ii/

* solution-recursive

  https://leetcode.com/problems/strobogrammatic-number-ii/discuss/67275/Python-recursive-solution-need-some-observation-so-far-97

  > Some observation to the sequence:
  >
  > n == 1: [0, 1, 8]
  >
  > n == 2: [11, 88, 69, 96]
  >
  > 
  >
  > How about n == `3`?
  > => it can be retrieved if you insert `[0, 1, 8]` to the middle of solution of n == `2`
  >
  > 
  >
  > n == `4`?
  > => it can be retrieved if you insert `[11, 88, 69, 96, 00]` to the middle of solution of n == `2`
  >
  > 
  >
  > n == `5`?
  > => it can be retrieved if you insert `[0, 1, 8]` to the middle of solution of n == `4`
  >
  > 
  >
  > the same, for n == `6`, it can be retrieved if you insert `[11, 88, 69, 96, 00]` to the middle of solution of n == `4`

  

#### 248. Strobogrammatic Number III

https://leetcode.com/problems/strobogrammatic-number-iii/



#### 157. Read N Characters Given Read4

https://leetcode.com/problems/read-n-characters-given-read4/



9.28

#### 158. Read N Characters Given Read4 II - Call multiple times

https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/

虽然题目烂，还是可以再做的



#### 68. Text Justification

https://leetcode.com/problems/text-justification/

* Solution-可以改进

  >  看了下，发现思想和自己也是一样的。但是这个速度却打败了 100% ，0 ms。考虑了下，差别应该在我的算法里使用了一个叫做 row 的 list 用来保存当前行的单词，用了很多 row.get ( index )，而上边的算法只记录了 left 和 right 下标，取单词直接用的 words 数组。然后尝试着在我之前的算法上改了一下，去掉 row，用两个变量 start 和 end 保存当前行的单词范围。主要是 ( end - start ) 代替了之前的 row.size ( )， words [ start + k ] 代替了之前的 row.get ( k )。
  >
  > 充分说明 list 的读取还是没有数组的直接读取快呀，还有就是要向上边的作者学习，多封装几个函数，思路会更加清晰，代码也会简明。
  >
  > https://leetcode.wang/leetCode-68-Text-Justification.html



12.13

#### 65. Valid Number

https://leetcode.com/problems/valid-number/

#### DFA(Deterministic Finite Automaton)-worth doing and thinking, 下次画图@12.22

Link: https://leetcode.com/problems/valid-number/discuss/23728/A-simple-solution-in-Python-based-on-DFA

```python
class Solution(object):
  def isNumber(self, s):
      """
      :type s: str
      :rtype: bool
      """
      #define a DFA
      state = [{}, 
              {'blank': 1, 'sign': 2, 'digit':3, '.':4}, 
              {'digit':3, '.':4},
              {'digit':3, '.':5, 'e':6, 'blank':9},
              {'digit':5},
              {'digit':5, 'e':6, 'blank':9},
              {'sign':7, 'digit':8},
              {'digit':8},
              {'digit':8, 'blank':9},
              {'blank':9}]
      currentState = 1
      for c in s:
          if c >= '0' and c <= '9':
              c = 'digit'
          if c == ' ':
              c = 'blank'
          if c in ['+', '-']:
              c = 'sign'
          if c not in state[currentState].keys():
              return False
          currentState = state[currentState][c]
      if currentState not in [3,5,8,9]:
          return False
      return True
    
# with comments
class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        #define DFA state transition tables
        states = [{},
                 # State (1) - initial state (scan ahead thru blanks)
                 {'blank': 1, 'sign': 2, 'digit':3, '.':4},
                 # State (2) - found sign (expect digit/dot)
                 {'digit':3, '.':4},
                 # State (3) - digit consumer (loop until non-digit)
                 {'digit':3, '.':5, 'e':6, 'blank':9},
                 # State (4) - found dot (only a digit is valid)
                 {'digit':5},
                 # State (5) - after dot (expect digits, e, or end of valid input)
                 {'digit':5, 'e':6, 'blank':9},
                 # State (6) - found 'e' (only a sign or digit valid)
                 {'sign':7, 'digit':8},
                 # State (7) - sign after 'e' (only digit)
                 {'digit':8},
                 # State (8) - digit after 'e' (expect digits or end of valid input) 
                 {'digit':8, 'blank':9},
                 # State (9) - Terminal state (fail if non-blank found)
                 {'blank':9}]
        currentState = 1
        for c in s:
            # If char c is of a known class set it to the class name
            if c in '0123456789':
                c = 'digit'
            elif c in ' \t\n':
                c = 'blank'
            elif c in '+-':
                c = 'sign'
            # If char/class is not in our state transition table it is invalid input
            if c not in states[currentState]:
                return False
            # State transition
            currentState = states[currentState][c]
        # The only valid terminal states are end on digit, after dot, digit after e, or white space after valid input    
        if currentState not in [3,5,8,9]:
            return False
        return True
```



## Substring

#### 76. Minimum Window Substring

https://leetcode.com/problems/minimum-window-substring/

* Solution- sliding window, 再做，我觉得不难@12.23

更快的解法

```python
from collections import Counter
class Solution:
    def minWindow(self, s, t):

        if not t or not s:
            return ""

    # Dictionary which keeps a count of all the unique characters in t.
        dict_t = Counter(t)

    # Number of unique characters in t, which need to be present in the desired window.
        required = len(dict_t)

    # left and right pointer
        l, r = 0, 0

    # formed is used to keep track of how many unique characters in t are present in the current window in its desired frequency.
    # e.g. if t is "AABC" then the window must have two A's, one B and one C. Thus formed would be = 3 when all these conditions are met.
        formed = 0

    # Dictionary which keeps a count of all the unique characters in the current window.
        window_counts = {}

    # ans tuple of the form (window length, left, right)
        ans = float("inf"), None, None

        while r < len(s):

        # Add one character from the right to the window
            character = s[r]
            window_counts[character] = window_counts.get(character, 0) + 1

        # If the frequency of the current character added equals to the desired count in t then increment the formed count by 1.
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1

        # Try and contract the window till the point where it ceases to be 'desirable'.
            while l <= r and formed == required:
                character = s[l]

            # Save the smallest window until now.
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)

            # The character at the position pointed by the `left` pointer is no longer a part of the window.
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1

            # Move the left pointer ahead, this would help to look for a new window.
                l += 1    

        # Keep expanding the window once we are done contracting.
            r += 1    
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]
```



#### 30. Substring with Concatenation of All Words

https://leetcode.com/problems/substring-with-concatenation-of-all-words/

* Solution-sliding window, two pointer



#### 3. Longest Substring Without Repeating Characters

https://leetcode.com/problems/longest-substring-without-repeating-characters/

* solution-会做前面两个就根本不用看这个



12.14

#### 340. Longest Substring with At Most K Distinct Characters

https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/

* solution-会做前面的这个也会做



#### 395. Longest Substring with At Least K Repeating Characters

https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/

* Solution-Divide and Conquer- worth doing and thinking-12.23还是不会

```python
def longestSubstring(self, s, k):
    for c in set(s): 
        if s.count(c) < k:
            return max(self.longestSubstring(t, k) for t in s.split(c))
    return len(s)
```



#### 159. Longest Substring with At Most Two Distinct Characters

https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/

* Solution-这就是340的降级版



## Palindrome

#### 125. Valid Palindrome

https://leetcode.com/problems/valid-palindrome/

* Solution-easy, Palindrome感觉一般都用two pointers?

  我这里为了避免大小写，将大写小写都映射到同一个值。在python里并不需要字典
  
* Solution-用re模块, 会很快@12.23



#### 5. Longest Palindromic Substring

https://leetcode.com/problems/longest-palindromic-substring/

* Solution-dynamic programming- **worth doing and thinking**

> Ref：https://leetcode.wang/leetCode-5-Longest-Palindromic-Substring.html
>
> 首先定义 P（i，j）。
>
> *P*(*i*,*j*)= True s[i,j] 是回文串
>
> *P*(*i*,*j*)= False s[i,j] 不是回文串
>
> 接下来
>
> P(*i*,*j*)=(*P*(*i*+1,*j*−1)&&*S*[*i*]==*S*[*j*])
>
> （当求第 i 行的时候我们只需要第 i + 1 行的信息，并且 j 的话需要 j - 1 的信息，所以倒着遍历这样我们可以把二维数组转为用一维数组）
>
> <img src="/Users/leslieren/Desktop/Screen Shot 2019-12-23 at 9.49.52 PM.png" alt="Screen Shot 2019-12-23 at 9.49.52 PM" style="zoom:33%;" />
>
> ```python
> public String longestPalindrome7(String s) {
>         int n = s.length();
>         String res = "";
>         boolean[] P = new boolean[n];
>         for (int i = n - 1; i >= 0; i--) {
>             for (int j = n - 1; j >= i; j--) {
>                 P[j] = s.charAt(i) == s.charAt(j) && (j - i < 3 || P[j - 1]);
>               //j - i < 3 意味着字符串长度小于等于3，很好理解
>               //P[j - 1] 上面那个圆圈是由下面的圆圈决定的，因为下面圆圈代表2起始3终点的substring是否palindromic
>                 if (P[j] && j - i + 1 > res.length()) {
>                     res = s.substring(i, j + 1);
>                 }
>             }
>         }
>         return res;
>     }
> ```
>
> 



* Solution-Manacher Algorithm-worth doing and thinking

https://www.cnblogs.com/grandyang/p/4475985.html

https://www.zhihu.com/question/37289584

*This is also a method to make use of already known palindromes.*

**main idea**:

Add # to string:

For an odd number length example, the original string is "bob". After adding "#", it becomes "#b#o#b#". Also, we add a "$" before it, that is "\$#b#o#b#", we call it "s1".

In this way, the index of the longest palindrome here is "#b#o#b#", and the center is "o" with an index in s1 as 4, with a radius of 3(not including the center), the index of the beginning palindrome without #$ in the "bob" is 0 

For an even number length example, the original string is "122223". After adding "#", it becomes "#1#2#2#2#2#3#". Also, we add a "$" before it, that is "\$#1#2#2#2#2#3#", we call it "s2".

In this way, the index of the longest palindrome here is "#2#2#2#2#", and the center is "#" with an index in s2 as 7, with a radius of 4(not including the center), the index of the beginning palindrome without #$ in the "122223" is 1 

And we found out, the radius is the length of the palindrome; **the difference** of the new index of the center minus the radius **and then divided by 2 (i.e. the quotient) is the palindrome's begining character index** in original string.(not the residual)



**notation**:
ma: strings after adding "#" in it

mp[i]: the maximum radius of the palindrome with the center of i th character

mx: the most right position of already known palindromes

id: the center of the palindrome with the most right position in above

**main problem**: How to update Mp[i]?(here we make use of the already known palindromes in the min part: min(p[2 * id - i], mx - i))

```
mp[i] = mx > i ? min(p[2 * id - i], mx - i) : 1;
```

If mx<=i, we can only use mp[i]=1, and then we would compare the characters beside it step by step

If mx>i, we would try to make use of the index that is symmatric to i with the center of id.

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4wyed1j31c00u0nce.jpg" alt="image-20191216205337687" style="zoom:55%;" />



* solution-扩展中心, 应该会快一些-you can try@12.23

Ref: https://leetcode.wang/leetCode-5-Longest-Palindromic-Substring.html 解法4

Ref2: https://leetcode.com/problems/longest-palindromic-substring/discuss/2954/Python-easy-to-understand-solution-with-comments-(from-middle-to-two-ends).

```python
def longestPalindrome(self, s):
    res = ""
    for i in xrange(len(s)):
        # odd case, like "aba"
        tmp = self.helper(s, i, i)
        if len(tmp) > len(res):
            res = tmp
        # even case, like "abba"
        tmp = self.helper(s, i, i+1)
        if len(tmp) > len(res):
            res = tmp
    return res
 
# get the longest palindrome, l, r are the middle indexes   
# from inner to outer
def helper(self, s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return s[l+1:r]
```





12.15

#### 9. Palindrome Number

https://leetcode.com/problems/palindrome-number/

* Solution-翻转字符串, 通过取整和取余来获得想要的数字
* Solution-只要翻转后半部分，和前面相等则True



#### 214. Shortest Palindrome

https://leetcode.com/problems/shortest-palindrome/

* Solution-Brute force-O(N2)-可以想想

Ref: https://leetcode.com/problems/shortest-palindrome/discuss/60099/AC-in-288-ms-simple-brute-force

```python
def shortestPalindrome(self, s):
    r = s[::-1]
    for i in range(len(s) + 1):
        if s.startswith(r[i:]):
            return r[:i] + s
```

* Solution-recursive-***worth thinking and doing*** @12.23

Ref: https://leetcode.com/problems/shortest-palindrome/discuss/60250/My-recursive-Python-solution

```python
	  if not s or len(s) == 1:
        return s
    j = 0
    for i in reversed(range(len(s))):
        if s[i] == s[j]:
            j += 1
    return s[::-1][:len(s)-j] + self.shortestPalindrome(s[:j-len(s)]) + s[j-len(s):]
```



* Solution-KMP, Knuth–Morris–Pratt, **worth doing and thinking**

算法解释：https://www.cnblogs.com/grandyang/p/6992403.html



#### 336. Palindrome Pairs

https://leetcode.com/problems/palindrome-pairs/

* solution-O(n*L^2), n is the number of words in the dict, L is the average length of each word.

Ref: https://leetcode.com/problems/palindrome-pairs/discuss/79219/Python-solution~

```python
class Solution:
    def is_valid(self, s):
        return s==s[::-1]
    
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        dic = {word: i for i, word in enumerate(words)}
        res = []
        
        for i, word in enumerate(words):
            for j in range(len(word)+1):
                prefix = word[:j]
                suffix = word[j:]
                # when j== len(word), prefix is the complete word, suffix is ""
                # so we should not count it when j==0 in the second condition to eliminate duplicates
                
                if j!=0 and self.is_valid(prefix) and suffix[::-1] in dic and dic[suffix[::-1]]!=i:
                    res.append([dic[suffix[::-1]],i])
                if self.is_valid(suffix) and prefix[::-1] in dic and dic[prefix[::-1]]!=i:
                    res.append([i, dic[prefix[::-1]]])
        return res
```





#### 131. Palindrome Partitioning

https://leetcode.com/problems/palindrome-partitioning/

by huahua: 优化的问题通常用dp或dfs

* Solution- divide and conquer-**worth doing and thinking**@12.24

设置递归出口

ref：https://leetcode.wang/leetcode-131-Palindrome-Partitioning.html

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        return self.helper(s, 0)
           
    def helper(self, s, start):
        if start==len(s):
            return [[]]
        
        ans = []
        
        for i in range(start, len(s)):
            if self.isPali(s[start:i+1]):
                left = s[start:i+1]
                for l in self.helper(s, i+1):
                    ans.append([left]+l)
        return ans
            
    def isPali(self, s):
        # identify if a given string is a palindrome
        return s==s[::-1]
```

加强版，引入动态规划

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        if not s:
            return [[]]
        
        dp = {0: [[]], 1: [[s[0]]]}
        for i in range(1, len(s)):
            dp[i + 1] = []
            for j in range(0, i + 1):
                if self.is_valid(s[j:i + 1]):
                    for sol in dp[j]:
                        dp[i + 1].append(sol + [s[j:i + 1]])
        
        return dp[len(s)]
                
                
    def is_valid(self, s):
        return all(s[i] == s[~i] for i in range(len(s) // 2))
```



* solution-dfs, backtrack-**worth doing and thinking**@12.24

ref: https://leetcode.wang/leetcode-131-Palindrome-Partitioning.html

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s)+1):
            if self.isPal(s[:i]):
                self.dfs(s[i:], path+[s[:i]], res)

    def isPal(self, s):
        return s == s[::-1]
```



12.16

#### 132. Palindrome Partitioning II

https://leetcode.com/problems/palindrome-partitioning-ii/

* Solution- dynamic programming-**worth doing and thinking**

https://www.cnblogs.com/grandyang/p/4271456.html

感觉关键是找递推方程

> 一维的dp数组，其中dp[i]表示子串 [0, i] 范围内的最小分割数，那么我们最终要返回的就是 dp[n-1] 了.
>
> 并且加个corner case的判断，若s串为空，直接返回0。
>
> 而如何更新dp[i], 这个区间的每个位置都可以尝试分割开来，所以就用一个变量j来从0遍历到i，这样就可以把区间 [0, i] 分为两部分，[0, j-1] 和 [j, i]。而因为我们从前往后更新，所以我们已经知道区间 [0, j-1] 的最小分割数 dp[j-1]， 这样我们就只需要判断区间 [j, i] 内的子串是否为回文串了。

如下图，先判断 `start` 到 `i` 是否是回文串，如果是的话，就用 `1 + d` 和之前的 `min` 比较。

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4t6f0gj30t410c0uq.jpg" alt="image-20191216162740834" style="zoom:30%;" />

如下图，`i` 后移，继续判断 `start` 到 `i` 是否是回文串，如果是的话，就用 `1 + c` 和之前的 `min`比较。

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4yhto8j30p4114q4w.jpg" alt="image-20191216162832270" style="zoom:30%;" />

然后 `i` 继续后移重复上边的过程。每次选一个较小的切割次数，最后问号处就求出来了。

接着 `start` 继续前移，重复上边的过程，直到求出 `start` 等于 `0` 的最小切割次数就是我们要找的了。

仔细考虑下上边的状态，其实状态转移方程也就出来了。

用 `dp[i]` 表示字符串 `s[i,s.lenght-1]`，也就是从 `i` 开始到末尾的字符串的最小切割次数。

求 `dp[i]` 的话，假设 `s[i,j]` 是回文串。

那么 `dp[i] = Min(min,dp[j + 1])`.

然后考虑所有的 `j`，其中 `j > i` ，找出最小的即可。

```python
# by myself, slow, time: O(N^3)
class Solution:
    def minCut(self, s: str) -> int:
      	# 加了种特殊的判断就会快很多
        if self.isPal(s):
            return 0
        
        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1
      
        dp = [len(s)-i-1 for i in range(len(s)+1)]
        
        for start in range(len(s)-1, -1, -1):
            for i in range(start+1, len(s)+1):# max of i should be len(s)
                if self.isPal(s[start:i]):#s[]
                    dp[start] = min(dp[start], 1+dp[i])
                    
        return dp[0]
                
    def isPal(self, s):
        return s==s[::-1]
```

可以dp存储一下回文字符串优化，降到O(N^2), 其他的问题，不是palindrome其他的一个function可以做类似处理，在判断valid这里使用dp存储

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4w8fmfj31c00u0gyj.jpg" alt="image-20191225091012152" style="zoom:50%;" />









* Solution-backtrack/dfs-**worth doing and thinking**

by huahua, 这种长度未知的，一般使用dp而不用backtrack？？



#### 267. Palindrome Permutation II

https://leetcode.com/problems/palindrome-permutation-ii/

* Solution-backtrack-**worth doing medium**, 对backtrack的理解还不到位@12.25

https://leetcode.com/problems/palindrome-permutation-ii/discuss/120631/Short-Python-Solution-with-backtracking



## Parentheses

#### 20. Valid Parentheses

https://leetcode.com/problems/valid-parentheses/

* Solution-easy, 稍微想一下就好



#### 22. Generate Parentheses

https://leetcode.com/problems/generate-parentheses/

* Solution-backtrack, 我想出来了！突然不会@12.28



12.17

#### 32. Longest Valid Parentheses

https://leetcode.com/problems/longest-valid-parentheses/

* Solution-dynamic programming

Ref: https://leetcode.wang/leetCode-32-Longest-Valid-Parentheses.html

dp [ i ] 代表以下标 i 结尾的合法序列的最长长度

* solution-stack-**worth thinking and doing**, 想明白了@1.2

Ref: https://leetcode.wang/leetCode-32-Longest-Valid-Parentheses.html



#### 241. Different Ways to Add Parentheses

https://leetcode.com/problems/different-ways-to-add-parentheses/

* Solution-divide and conquer- **worth thinking and doing**@1.2!!!循环找分割点

Ref: https://www.youtube.com/watch?v=gxYV8eZY0eQ&t=280s

http://zxi.mytechroad.com/blog/leetcode/leetcode-241-different-ways-to-add-parentheses/

```python
# based on huahua's solution
class Solution:
    def diffWaysToCompute(self, input):
        # a wise use of lambda
        ops = {'+': lambda x,y:x+y,
              '-': lambda x,y:x-y,
              '*': lambda x,y:x*y}
        
        def ways(s):
            ans=[]
            for i in range(len(s)):
                if s[i] in '+-*':
                    # you csn use string rather than list ['+','-','*']
                    ways1 = ways(s[:i])
                    ways2 = ways(s[i+1:])
                    ans += [ops[s[i]](l, r) for l in ways1 for r in ways2]
                    # a wise use of lambda
            if not ans:
                ans.append(int(s))
            return ans
        return ways(input)
```





#### 301. Remove Invalid Parentheses

https://leetcode.com/problems/remove-invalid-parentheses/

* solution-DFS- **worth thinking and doing**

Ref: https://www.youtube.com/watch?v=2k_rS_u6EBk

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4trb7kj30t410c0uq.jpg" alt="image-20200102154813925" style="zoom:50%;" />



```python
# based on huahua's code
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        l = 0
        r = 0
        
        # calculate redundant left or right parenthesis
        for char in s:
            if char=='(':
                l+=1
            elif char ==')':
                if l==0:
                    r+=1
                else:
                    l-=1
        # check if string of parenthesis is valid or not
        def isValid(s):
            count = 0
            for char in s:
                if char == '(':
                    count+=1
                if char == ')':
                    count-=1
                if count<0:
                    return False
            return count==0
        
        def dfs(s, start, l, r, ans):
            if l==0 and r ==0:
                if isValid(s):
                    ans.append(s)
                return
            
            for i in range(start, len(s)):
                if i!= start and s[i]==s[i-1]:
                    continue
                if s[i]=='(' or s[i]==')':
                    curr = s[:i]+s[i+1:]
                # 一定要先移除右括号再移除左括号
                if r>0 and s[i]==')':
                    dfs(curr, i, l, r-1, ans)
                elif l>0 and s[i]=='(':
                    dfs(curr, i, l-1, r, ans)
        
        
        ans = []
        dfs(s, 0, l, r, ans)
        return ans
```

![image-20191218162133174](https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4ubxo5j31c00u0nce.jpg)



12.18

## Subsequence

#### 392. Is Subsequence

https://leetcode.com/problems/is-subsequence/



#### 115. Distinct Subsequences

https://leetcode.com/problems/distinct-subsequences/

* solution-dynamic programming-**worth thinking and doing**

https://www.youtube.com/watch?v=mPqqXh8XvWY

计数的题目通常用dp来做，两个string通常就是二维数组

```python
# mysolution @ 12.18
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        dp = [[1 for i in range(len(s)+1)]] + [[0 for i in range(len(s)+1)] for j in range(len(t))]

        for i in range(1, len(t)+1):
            for j in range(1, len(s)+1):
                if t[i-1]==s[j-1]:
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1]#用当前匹配上的这个和不用当前匹配上的
                else:
                    dp[i][j] = dp[i][j-1]
                    
        return dp[len(t)][len(s)]
```



<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol4v862dj31c00u0nak.jpg" alt="image-20191225091155062" style="zoom:50%;" />





#### 187. Repeated DNA Sequences

https://leetcode.com/problems/repeated-dna-sequences/

* Solution -hash table, bit manipulation-又忘了字符串操作@1.02

```python
class Solution:# my solution
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        # A: 1000001
        # C: 1000011
        # G: 1000111
        # T: 1010100
        # 7: 0000111
        
        toInt = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

        if len(s) < 11:
            return []
        keys = collections.defaultdict(int)
        key = 0
        answer = []
        mask = (1 << 20) - 1
        # bin(mask): '0b11111111111111111111'后面有20个1
        # bin(): Return the binary representation of an integer.
        # 1<<n则为2的多少次方
        
        # initialize the first 9 chars
        for i in range(9): #要移两位是因为，二进制里要两位才能代表这四个字母，因为这里最大的十进制数位3
            key = (key << 2) | toInt[s[i]] # bit manipulstion
        for i in range(9, len(s)):
            key = (((key << 2) | toInt[s[i]])) & mask # bit manipulstion
            keys[key] += 1
            if keys[key]==2:
                answer.append(s[i-9:i+1])
            
        return answer
```


