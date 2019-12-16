##basics

#### 8.5

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



#### 8.13

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

#### DFA(Deterministic Finite Automaton)-worth doing and thinking

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



### Substring

#### 76. Minimum Window Substring

https://leetcode.com/problems/minimum-window-substring/

* Solution- sliding window



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

* Solution-Divide and Conquer- worth doing and thinking

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



### Palindrome

#### 125. Valid Palindrome

https://leetcode.com/problems/valid-palindrome/

* Solution-easy, Palindrome感觉一般都用two pointers?

  我这里为了避免大小写，将大写小写都映射到同一个值



#### 5. Longest Palindromic Substring

https://leetcode.com/problems/longest-palindromic-substring/

* Solution-dynamic programming
* Solution-Manacher Algorithm-worth doing and thinking

https://www.cnblogs.com/grandyang/p/4475985.html

https://www.zhihu.com/question/37289584



12.15

#### 9. Palindrome Number

https://leetcode.com/problems/palindrome-number/

* Solution-翻转字符串, 通过取整和取余来获得想要的数字
* Solution-只要翻转后半部分，和前面相等则True



#### 214. Shortest Palindrome

https://leetcode.com/problems/shortest-palindrome/

* Solution-Brute force-O(N2)-可以想想
* Solution-KMP, Knuth–Morris–Pratt, **worth doing and thinking**

算法解释：https://www.cnblogs.com/grandyang/p/6992403.html



#### 336. Palindrome Pairs

https://leetcode.com/problems/palindrome-pairs/

* solution-https://www.cnblogs.com/grandyang/p/5272039.html 注意边界的处理

> 要用到哈希表来建立每个单词和其位置的映射.
>
> 然后需要一个set来保存出现过的单词的长度，
>
> Brach 1: 算法的思想是，遍历单词集，对于遍历到的单词，我们对其翻转一下，然后在哈希表查找翻转后的字符串是否存在，注意不能和原字符串的坐标位置相同，因为有可能一个单词翻转后和原单词相等，现在我们只是处理了bat和tab的情况
>
> Branch 2:还存在abcd和cba，dcb和abcd这些情况需要考虑，这就是我们为啥需要用set，由于set是自动排序的，我们可以找到当前单词长度在set中的iterator，然后从开头开始遍历set，遍历比当前单词小的长度，比如abcdd翻转后为ddcba，我们发现set中有长度为3的单词，1⃣️我们判断dd是否为回文串，若是，再看cba是否存在于哈希表，若存在，则说明abcdd和cba是回文对，存入结果中，2⃣️看另一边，判断ab是否是回文串，不是就skip。对于dcb和aabcd这类的情况也是同样处理，我们要在set里找的字符串要在遍历到的字符串的左边和右边分别尝试，看是否是回文对，这样遍历完单词集，就能得到所有的回文对，



#### 131. Palindrome Partitioning

https://leetcode.com/problems/palindrome-partitioning/

* Solution- divide and conquer-worth doing and thinking

设置递归出口

ref：https://leetcode.wang/leetcode-131-Palindrome-Partitioning.html

* solution-dfs, backtrack-worth doing and thinking

ref: https://leetcode.wang/leetcode-131-Palindrome-Partitioning.html







## 语法

#### split()

不管中间空格有几个都可以分组



```python
# Get the ASCII number of a character
number = ord(char)

# Get the character given by an ASCII number
char = chr(number)
```





#### collections.defaultdict

Usually, a Python dictionary throws a `KeyError` if you try to get an item with a key that is not currently in the dictionary. The `defaultdict` in contrast will simply create any items that you try to access (provided of course they do not exist yet). To create such a "default" item, it calls the function object that you pass to the constructor (more precisely, it's an arbitrary "callable" object, which includes function and type objects). 



#### find()

```python
str.find(sub[, start[, end]] )
```

- Parameter

  * **sub** - It's the substring to be searched in the str string.

  * **start** and **end** (optional) - substring is searched within `str[start:end]`

- It returns value: 

  * If substring exists inside the string, it returns the index of first occurence of the substring.
  * If substring doesn't exist inside the string, it returns -1.




#### set()可以对字符串操作

```python
s = 'apple'
set(s) 
# Out[11]: {'a', 'e', 'l', 'p'}
```



#### string.count()

```python
# S.count(sub[, start[, end]]) -> int
s = 'apple'
s.count('p') 
# Out[12]: 2
```



#### string.isalnum()

Return True if the string is an alpha-numeric string, False otherwise.

```python
In [16]: s='a'                                                                  
In [17]: s.isalnum()                                                            
Out[17]: True
```



#### 乘除法

**/** 默认得到float

**//** 舍小数得到int