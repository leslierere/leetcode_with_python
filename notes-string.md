##basics

### 28. Implement strStr()-cc

https://leetcode.com/problems/implement-strstr/

#### Solution-simple loop

加特殊情况判断可speed up

#### Solution-KMP-$

See details on problem214 in this note







### 14. Longest Common Prefix

https://leetcode.com/problems/longest-common-prefix/

#### Solution

用最小的作为参考来比较更快

did@21.5.20

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        prefix = strs[0]
        
        for i in range(1, len(strs)):
            maxLength = min(len(prefix), len(strs[i]))
            prefix = prefix[:maxLength]
            for j in range(maxLength):
                if prefix[j]!=strs[i][j]:
                    prefix = prefix[:j]
                    break
            if len(prefix)==0:
                break
                
        return prefix
```





### 58. Length of Last Word

https://leetcode.com/problems/length-of-last-word/

我本来想到的是用一个flag来表示是否已经有词，但其实用`l > 0` 就可以知道，聪明的做法！

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        l = 0
        for i in range(len(s)-1, -1, -1):
            if s[i] != " ":
                l+=1
            elif l > 0:
                break
        return l
```





### 387. First Unique Character in a String

https://leetcode.com/problems/first-unique-character-in-a-string/

用collections.Counter计数较快

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        c = Counter(s)
        for i, ch in enumerate(s):
            if c[ch] == 1:
                return i
        return -1
```



### 383. Ransom Note

https://leetcode.com/problems/ransom-note/

#### Solution

用Counter的减法，Ref: https://leetcode.com/problems/ransom-note/discuss/85837/O(m%2Bn)-one-liner-Python

```python
def canConstruct(self, ransomNote, magazine):
    return not collections.Counter(ransomNote) - collections.Counter(magazine)
```





### 344. Reverse String-easy

https://leetcode.com/problems/reverse-string/





### 151. Reverse Words in a String

https://leetcode.com/problems/reverse-words-in-a-string/





### 186. Reverse Words in a String II

https://leetcode.com/problems/reverse-words-in-a-string-ii/



### 205. Isomorphic Strings

https://leetcode.com/problems/isomorphic-strings/

#### Solution

Ref: https://leetcode.com/problems/isomorphic-strings/discuss/57838/1-line-in-Python

This is amazing

```python
def isIsomorphic3(self, s, t):
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))
```



### 293. Flip Game-easy

https://leetcode.com/problems/flip-game/

8.13

###  294. Flip Game II-$

https://leetcode.com/problems/flip-game-ii/

#### Solution-minimax&backtracking

Ref: https://leetcode.com/problems/flip-game-ii/discuss/73958/Memoization%3A-3150ms-greater-130ms-greater-44ms-(Python)

第一步有一种机会可以赢就能赢，因为反过来对方的情况是所有的走法都会输

#### Solution-game theory

Ref: https://leetcode.com/problems/flip-game-ii/discuss/73954/Theory-matters-from-Backtracking(128ms)-to-DP-(0ms)



### 290. Word Pattern-解法可以看看

https://leetcode.com/problems/word-pattern/

#### solution

使用了map()，dict.setdefault(), 列表的等于会逐项比较判断

Python dictionary method **setdefault()** is similar to get(), but will set *dict[key]=default* if key is not already in dict.

Ref: https://leetcode.com/problems/word-pattern/discuss/73433/Short-in-Python

```python
def wordPattern(self, pattern, str):
    f = lambda s: map(dict.setdefault, s, range(len(s)))
    return f(pattern) == f(str.split())
```

Here, f is:

```python
def f(sequence):
    first_occurrence = {}
    normalized = []
    for i, item in enumerate(sequence):
        if item not in first_occurrence:
            first_occurrence[item] = i
        normalized.append(first_occurrence[item])
    return normalized
```



### 242.Valid Anagram-easy

#### Solution-Counter()



8.21

### 49. Group Anagrams

https://leetcode.com/problems/group-anagrams/



### 249. Group Shifted Strings

https://leetcode.com/problems/group-shifted-strings/

想想怎么解决一个字母的时候，比如这里对第一个字母同样的操作对key的一一对应没有影响

#### Solution

[Ref](https://leetcode.com/problems/group-shifted-strings/discuss/67466/1-4-lines-Ruby-and-Python): 巧妙解决了只有一个字母的时候的情况

```python
def groupStrings(self, strings):
    groups = collections.defaultdict(list)
    for s in strings:
        groups[tuple((ord(c) - ord(s[0])) % 26 for c in s)] += s,
    return groups.values()
```





### 87. Scramble String-$

注意其实是可以任意分的

https://leetcode.com/problems/scramble-string/

#### Solution-recursion

Ref: https://leetcode.com/problems/scramble-string/discuss/29387/Accepted-Java-solution

```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        if len(s1)!=len(s2):
            return False
        
        if s1 ==s2:
            return True
        elif sorted(s1)!=sorted(s2):# an important step to prevent more recursions
            return False
        else:
            for i in range(len(s1)-1):  
                if self.isScramble(s1[:i+1], s2[:i+1]) and self.isScramble(s1[i+1:], s2[i+1:]):
                    return True
                if self.isScramble(s1[:i+1], s2[-i-1:]) and self.isScramble(s1[i+1:], s2[:-i-1]):
                    return True
        return False
```





#### Solution-DP-$$

Ref: https://leetcode.wang/leetCode-87-Scramble-String.html







### 161. One Edit Distance-看看想想

https://leetcode.com/problems/one-edit-distance/

#### Solution

感觉这个更好吧，切分较少

Ref: https://leetcode.com/problems/one-edit-distance/discuss/50095/Python-concise-solution-with-comments.

did@21.6.17

```python
class Solution:
    def isOneEditDistance(self, s: str, t: str) -> bool:
        if abs(len(s)-len(t))>1:
            return False
        
        s+="*"
        t+="*"
        if len(s)<len(t):
            return self.isOneEditDistance(t,s)
            
        for i in range(len(t)):
            if s[i]!=t[i]:
                if len(s)!=len(t):
                    return s[i+1:]==t[i:]
                else:
                    return s[i+1:]==t[i+1:]
```





8.29

### 358. Rearrange String k Distance Apart-$

https://leetcode.com/problems/rearrange-string-k-distance-apart/

#### solution-Priority queue

下面一题767很像

Ref: https://leetcode.com/problems/rearrange-string-k-distance-apart/discuss/83192/Java-7-version-of-PriorityQueue-O(nlogn)-with-comments-and-explanations

> The greedy algorithm is that in each step, select the char with highest remaining count if possible (if it is not in the waiting queue). PQ is used to achieve the greedy. A regular queue waitQueue is used to "freeze" previous appeared char in the period of k.

did@20.9.1

```python
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        if k==0:
            return s
        
        counter = collections.Counter(s)
        lis = [(-counter[key], key) for key in counter] 
        heapq.heapify(lis)
        output = ""
        backlis = []
        heapq.heapify(backlis)
        
        while lis: # s*logk
            for _ in range(k):
                if not lis:
                    if backlis:
                        return ""
                    else:
                        return output
                value, letter = heapq.heappop(lis)
                output+=letter
                if value+1<0:
                    heapq.heappush(backlis, (value+1,letter))
            while backlis:
                heapq.heappush(lis, heapq.heappop(backlis))
                
                
        return output
        
```



### 767. Reorganize String

https://leetcode.com/problems/reorganize-string/

#### Solution-priority queue

Ref: https://leetcode.com/problems/reorganize-string/discuss/113457/Simple-python-solution-using-PriorityQueue

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        
        counter = collections.Counter(s)
        heap = [(-val, letter) for letter,val in counter.items()]
        heapq.heapify(heap)
        res = ""
        pre_count, pre_letter = 0, None
        
        while heap:
            val, letter = heapq.heappop(heap)
            # no need for this, if right now heap is not empty,we can definitely get a diff letter
            # if len(res)!=0 and res[-1]==letter:
            #     return ""
            res+=letter
            if pre_count!=0:
                heapq.heappush(heap, (pre_count, pre_letter))
            pre_count, pre_letter = val+1, letter
            
        if len(res)!=len(s):
            return ""
        return res
```





### 316. Remove Duplicate Letters-$

https://leetcode.com/problems/remove-duplicate-letters/

* solution-Stack, greedy

Ref: https://leetcode.com/problems/remove-duplicate-letters/discuss/76769/Java-solution-using-Stack-with-comments

did@20.9.2

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        if not s:
            return ""
        stack = [s[0]]
        
        for i in range(1, len(s)):
            char = s[i]
            
            if char not in stack:
                while stack and ord(stack[-1])>ord(char) and stack[-1] in s[i+1:]:
                    stack.pop()
                stack.append(char)
            
        return "".join(stack)
    		
        
```

Explanation:

1. why we only consider it if the char not in stack:
   Between the entering char and the same char currently in the stack, if there is a lower order right after the char in stack, the char will not be in the stack at all, if the right-after one is higher order, pluck the already-in-stack-same char and use the entering char would only make the situation worse

2. when should we pluck out chars in the stack

   * clearly,  if we need pluck out one, there must be one after the current char
   * also the plucked one should have higher order, as we always want the lower orders to be at the front

3. should we pluck out all the higher order ones in the stack or as long as we moved downward in the stack and  reached one of no higher order we should stop

   * the answer is we would never confront with a situation to skip chars in stack and pluck out chars below them.

     as if the skipped one are skipped, this means their oders are lower, say *the current one is **s***, the *skipped one is  **r***, since there are other chars before ***r*** we need pluck out, this means it is one of higher order than ***s***, say ***t***, and there is another ***t*** after ***s***,  however, the ***s*** would be plucked out when we try to insert ***r***

     

### 271. Encode and Decode Strings-$

https://leetcode.com/problems/encode-and-decode-strings/

* Solution-good thoughts

https://leetcode.com/problems/encode-and-decode-strings/discuss/70448/1%2B7-lines-Python-(length-prefixes)

[https://leetcode.com/problems/encode-and-decode-strings/discuss/70402/Java-with-%22escaping%22](https://leetcode.com/problems/encode-and-decode-strings/discuss/70402/Java-with-"escaping")



8.30

### 168. Excel Sheet Column Title

https://leetcode.com/problems/excel-sheet-column-title/

@20.9.5和进制的转换有点像，但是因为起点不是0会有点不一样

> Ref: https://leetcode.wang/leetcode-168-Excel-Sheet-Column-Title.html
>
> 区别就在于题目规定的数字中没有 `0` ，换句话讲，正常的 `26` 进制本应该满 `26` 进 `1`，然后低位补 `0`，但是这里满 `26` 的话就用 `26` 表示。满 `27` 的时候才会向前进 `1`，然后低位补 `1`。所以 `Z(26)` 的下一个数字就是 `A(1)A(1)`，即 `27` 对应 `AA`。



### 171. Excel Sheet Column Number

https://leetcode.com/problems/excel-sheet-column-number/





### 13. Roman to Integer

https://leetcode.com/problems/roman-to-integer/

#### Solution

Ref: https://leetcode.com/problems/roman-to-integer/discuss/6547/Clean-O(n)-c%2B%2B-solution

did@20.8.22

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        output = 0
        last = ""
        dic = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
        dic2 = {"V": "I", "X":"I", "L":"X", "C":"X", "D":"C", "M":"C"}
        
        for char in s:
            
            output+=dic[char]
            if char in dic2 and last==dic2[char]:
                output-=2*dic[last]
            last = char
            
        return output
```

did@21.5.20

I think this one is more straightforward.

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        ref = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
        deduct = {"I":"VX", "X":"LC", "C":"DM"}
        res = 0
        
        for i, char in enumerate(s):
            if char in deduct and i<len(s)-1 and s[i+1] in deduct[char]:
                res-=ref[char]
            else:
                res+=ref[char]
                
        return res
```





9.14

### 12. Integer to Roman

https://leetcode.com/problems/integer-to-roman/

#### Intersting Solution

Ref: https://leetcode.com/problems/integer-to-roman/discuss/6274/Simple-Solution

did@21.5.19

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        dic = {1:"I", 5:"V", 10:"X", 50:"L", 100:"C", 500:"D", 1000:"M"}
        
        number = 10
        res = ""
        
        while num:
            digit = num%10
            if digit<4:
                res = digit*dic[number//10]+res
            elif digit<=5:
                res = (5-digit)*dic[number//10]+dic[5*number//10]+res
            elif digit<9:
                res = dic[5*number//10] + (digit-5)*dic[number//10] + res
            else:
                res = dic[number//10]+dic[number]+res
            
            number*=10
            num = num//10
            
        return res
```





### 246. Strobogrammatic Number

https://leetcode.com/problems/strobogrammatic-number/

Easy



### 273. Integer to English Words-$

https://leetcode.com/problems/integer-to-english-words/

* Solution-recursive 

  Ref: https://leetcode.com/problems/integer-to-english-words/discuss/70632/Recursive-Python

  感觉如下分组更好
  
  ```python
  def words(n):
          if n < 20:
              return to19[n-1:n]
          if n < 100:
              return [tens[n/10-2]] + words(n%10)
          if n < 1000:
              return [to19[n/100-1]] + ['Hundred'] + words(n%100)
          for p, w in enumerate(('Thousand', 'Million', 'Billion'), 1): # 第二个参数1是让index从1开始
              if n < 1000**(p+1):
                  return words(n/1000**p) + [w] + words(n%1000**p)
      return ' '.join(words(num)) or 'Zero'
  ```





9.21

### 247. Strobogrammatic Number II

https://leetcode.com/problems/strobogrammatic-number-ii/

* solution-recursive-试着写写就好

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

  

### 248. Strobogrammatic Number III

https://leetcode.com/problems/strobogrammatic-number-iii/



### 157. Read N Characters Given Read4

https://leetcode.com/problems/read-n-characters-given-read4/

奇怪的题。。。

#### Solution

```python
"""
The read4 API is already defined for you.

    @param buf, a list of characters
    @return an integer
    def read4(buf):

# Below is an example of how the read4 API can be called.
file = File("abcdefghijk") # File is "abcdefghijk", initially file pointer (fp) points to 'a'
buf = [' '] * 4 # Create buffer with enough space to store characters
read4(buf) # read4 returns 4. Now buf = ['a','b','c','d'], fp points to 'e'
read4(buf) # read4 returns 4. Now buf = ['e','f','g','h'], fp points to 'i'
read4(buf) # read4 returns 3. Now buf = ['i','j','k',...], fp points to end of file
"""
class Solution:
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Number of characters to read (int)
        :rtype: The number of actual characters read (int)
        """
        readNumber = 0
        
        while n>0:
            buf4 = ['']*4
            curRead = read4(buf4)
            
            for i in range(0, min(n, 4)):
                buf[i+readNumber] = buf4[i]
            readNumber+=min(curRead, n)
            n-=4
            
        return readNumber
```



9.28

### 158. Read N Characters Given Read4 II - Call multiple times

https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/

虽然题目烂，还是可以再做的

did@20.9.10

```python
# The read4 API is already defined for you.
# def read4(buf4: List[str]) -> int:

class Solution:
    def __init__(self):
    
        self.remain = collections.deque([""]*4)
    
    def read(self, buf: List[str], n: int) -> int:
        position = 0
            
        while position<n:
            if self.remain[0]=="":
                readNo4 = read4(self.remain)
                if readNo4==0:
                    break
            buf[position] = self.remain.popleft()
            self.remain.append("")
            position+=1
            
        return position
```





### 68. Text Justification

https://leetcode.com/problems/text-justification/

* Solution-可以改进

  >  看了下，发现思想和自己也是一样的。但是这个速度却打败了 100% ，0 ms。考虑了下，差别应该在我的算法里使用了一个叫做 row 的 list 用来保存当前行的单词，用了很多 row.get ( index )，而上边的算法只记录了 left 和 right 下标，取单词直接用的 words 数组。然后尝试着在我之前的算法上改了一下，去掉 row，用两个变量 start 和 end 保存当前行的单词范围。主要是 ( end - start ) 代替了之前的 row.size ( )， words [ start + k ] 代替了之前的 row.get ( k )。
  >
  > 充分说明 list 的读取还是没有数组的直接读取快呀，还有就是要向上边的作者学习，多封装几个函数，思路会更加清晰，代码也会简明。
  >
  > https://leetcode.wang/leetCode-68-Text-Justification.html

Ref: https://leetcode.com/problems/text-justification/discuss/24891/Concise-python-solution-10-lines.

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        remain = maxWidth
        line = []
        res = []
        i = 0
        
        def helper(line, res):
            if len(line)==1:
                res.append(line[0]+" "*(maxWidth-len(line[0])))
                return
            spaces = len(line)+remain
            basicSpaces = spaces//(len(line)-1)
            extraNo = spaces%(len(line)-1)
            space = " "*basicSpaces
            
            for i in range(extraNo):#这个处理不错
                line[i]+=" "
            res.append(space.join(line))
            
        
        for word in words:
            if remain-len(word)>=0:
                remain-=len(word)+1
                line.append(word)
                i+=1
            else:
                helper(line, res)
                remain = maxWidth-len(word)-1
                line = [word]
                
        lastLine = " ".join(line)
        res.append(lastLine+" "*(maxWidth-len(lastLine)))
        
        return res
```





12.13

### 65. Valid Number-$

https://leetcode.com/problems/valid-number/

这个很clean：https://leetcode.com/problems/valid-number/discuss/173977/Python-with-simple-explanation

其实这道题里面就只是有e的时候没有和你讲清楚，这个时候就可以向面试官确认。

did@2020.9.12

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        # invalid:
        # invalid char
        # dot after e
        # no digit either before it or after it
        # two dots
        # sign not (at the beginning or right after e) or no number after it)
        # more than 1 e
        # no number after e
        # space between numbers
        # zero
        hasE = False
        hasDot = False
        s = s.strip()
        if not s:
            return False
        
        for i, char in enumerate(s):
            if not char.isdigit() and char not in "+-.e":
                return False
            if char=='e':
                if hasE:
                    return False
                if i+1==len(s) or not (s[i+1].isdigit() or s[i+1] in "+-") or i==0 or not (s[i-1].isdigit() or s[i-1]=='.'):
                    return False
                hasE=True
            elif char=='.':
                if hasDot or hasE:
                    return False
                if not ((i+1<len(s) and (s[i+1].isdigit() or (s[i+1]=='e' and i!=0))) or (i!=0 and (s[i-1].isdigit() or (s[i-1]=='e' and i+1!=len(s))))):
                    return False
                hasDot=True
            elif char in "+-":
                if i+1==len(s) or not (s[i+1].isdigit() or s[i+1]=='.'):
                    return False
                elif i!=0 and s[i-1]!='e':
                    return False
            # elif char=='0':
            #     if i>1 and s[i-1]=='e':
            #         return False
                
        
        return True
            
```



check if the one before one is decimal and the one after is integer

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        # it must be integer after e
        # return True
    
        s = s.strip()
        if not s:
            return False
        parts = s.split("e")
        if len(parts)>2:
            return False
        
        if not self.isValid(parts[0]):
            return False
        elif len(parts)==2 and ("." in parts[1] or not self.isValid(parts[1])):
            return False
        
        return True
            
        
        
    def isValid(self, s):
        if not s:
            return False
        hasDot = False
        
        for i, char in enumerate(s):
            if not char.isdigit() and char not in "+-.e":
                return False
            elif char=='.':
                if hasDot:
                    return False
                if not ((i+1<len(s) and (s[i+1].isdigit())) or (i!=0 and (s[i-1].isdigit()))):# there is no valid number either before it or after it
                    return False
                hasDot = True
            elif char in "+-":
                if i!=0 or i+1==len(s) or not (s[i+1].isdigit() or s[i+1]=='.'):
                    return False
        
        return True
        
        
```



### 6. ZigZag Conversion

https://leetcode.com/problems/zigzag-conversion/

#### Solution

https://leetcode.com/problems/zigzag-conversion/discuss/3404/Python-O(n)-Solution-in-96ms-(99.43)

太妙了！关键是要找到条件变化的地方

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows==1:
            return s
        rows = ["" for _ in range(numRows)]
        
        d = -1
        row = 0
        for i in range(len(s)):
            rows[row]+=s[i]
            if row==numRows-1 or row==0:
                d = -d
            row+=d
            
        return "".join(rows)
```





## Substring

### 76. Minimum Window Substring-$

https://leetcode.com/problems/minimum-window-substring/

@2021.6.2

开始我在找第一个匹配的时候单独把那部分循环拎了出来，但其实这没必要，这跟之后继续找匹配的时候可以放在一起

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



did@21.7.10

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        counter_t = collections.Counter(t)
        counter_s = collections.defaultdict(int)
        
        left = 0
        satisfied = 0
        result = ""
        
        
        for right in range(len(s)):
            char = s[right]
            if char in counter_t:
                counter_s[char] += 1
                if counter_s[char] == counter_t[char]:
                    satisfied += 1
                        
                            
            while satisfied == len(counter_t):
                if not result or (right - left + 1) < len(result):
                    result = s[left:right+1]
                left_char = s[left]
                if left_char in counter_t and counter_s[left_char] == counter_t[left_char]:
                    satisfied -= 1
                counter_s[left_char] -= 1
                left+=1
                
        return result
```





### 30. Substring with Concatenation of All Words-$

https://leetcode.com/problems/substring-with-concatenation-of-all-words/

#### Solution

Ref: https://leetcode.com/problems/substring-with-concatenation-of-all-words/discuss/13658/Easy-Two-Map-Solution-(C%2B%2BJava), time complexity: O((sLength-wordsL)* noOfWords)

```python
from collections import Counter, defaultdict

class Solution(object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        wordBag = Counter(words)   # count the freq of each word
        wordLen, numWords = len(words[0]), len(words)
        totalLen, res = wordLen*numWords, []
        for i in range(len(s)-totalLen+1):   # scan through s
            # For each i, determine if s[i:i+totalLen] is valid
            seen = defaultdict(int)   # reset for each i
            for j in range(i, i+totalLen, wordLen):
                currWord = s[j:j+wordLen]
                if currWord in wordBag:
                    seen[currWord] += 1
                    if seen[currWord] > wordBag[currWord]:
                        break
                else:   # if not in wordBag
                    break    
            if seen == wordBag:
                res.append(i)   # store result
        return res
```



#### Solution-sliding window

Ref: https://leetcode.com/problems/substring-with-concatenation-of-all-words/discuss/13699/92-JAVA-O(N)-with-explaination

did@2020.9.14

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        wordL = len(words[0])
        noWords = len(words)
        wholeL = wordL*noWords
        wordCount = collections.Counter(words)
        curCount = collections.defaultdict(int)
        result = []
        
        for start in range(wordL):
            left = right = start
            forms = 0
            while left<=len(s)-wholeL:
                
                while right<=len(s)-wordL and right-left<wholeL:
                    word = s[right:right+wordL]
                    
                    if word not in wordCount:
                        right = right+wordL
                        left = right
                        curCount.clear()
                        continue
                    while curCount[word]==wordCount[word]:
                        curCount[s[left:left+wordL]]-=1
                        left+=wordL
                        
                    curCount[word]+=1
                    right+=wordL
                    
                if right-left==wholeL:
                    result.append(left)
                    
                curCount[s[left:left+wordL]]-=1
                left+=wordL
                right = left
                curCount.clear()
                
                    
        return result
```



### 3. Longest Substring Without Repeating Characters

https://leetcode.com/problems/longest-substring-without-repeating-characters/

#### Solution

Nice!

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        seen = {}
        start = end = 0
        max_length = 0
        
        for end in range(len(s)):
            if s[end] in seen:
                start = max(seen[s[end]]+1, start)
                # start = seen[s[end]]+1
            
            seen[s[end]] = end
            max_length = max(max_length, end-start+1)
        
        return max_length
```





12.14

### 340. Longest Substring with At Most K Distinct Characters

https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/

* solution-会做前面的这个也会做

没有必要像我开始用了一个deque来存当前window的元素，存index就行

```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        l = r = 0
        charDict = collections.defaultdict(int)
        
        res = []
        
        while r<len(s):
            charDict[s[r]]+=1
            while len(charDict)>k:
                res.append(r-l)
                charDict[s[l]]-=1
                if charDict[s[l]]==0:
                    charDict.pop(s[l])
                l+=1
            r+=1
        res.append(r-l)
        
        return max(res)
```



### 395. Longest Substring with At Least K Repeating Characters-$

https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/

* Solution-Divide and Conquer- worth doing and thinking

https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/discuss/87768/4-lines-Python

3.12终于想出来了

> I can just take the first too rare character instead of a rarest.

```python
import re
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if len(s)<k:
            return 0
        
        charSet = set(s)
        
        for char in charSet:
            if s.count(char)<k: # we can divide as long as we find one not qualified
                substrings = s.split(char)
                return max([self.longestSubstring(substring, k) for substring in substrings])
            
        return len(s)
                
```



### 159. Longest Substring with At Most Two Distinct Characters

https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/

* Solution-这就是340的降级版

```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        l = r = 0
        charDict = collections.defaultdict(int)
        
        res = []
        
        while r<len(s):
            charDict[s[r]]+=1
            while len(charDict)>k:
                res.append(r-l)
                charDict[s[l]]-=1
                if charDict[s[l]]==0:
                    charDict.pop(s[l])
                l+=1
            r+=1
        res.append(r-l)
        
        return max(res)
```



## Palindrome

### 125. Valid Palindrome

https://leetcode.com/problems/valid-palindrome/

#### Solution-easy, 2 pointers

#### Solution-用re模块, 会很快@12.23



### 5. Longest Palindromic Substring-$

https://leetcode.com/problems/longest-palindromic-substring/

#### Solution-dynamic programming- **worth doing and thinking**



Ref：https://leetcode.wang/leetCode-5-Longest-Palindromic-Substring.html

>  首先定义 P（i，j）。
>
> *P*(*i*,*j*)= True s[i,j] 是回文串
>
> *P*(*i*,*j*)= False s[i,j] 不是回文串
>
> 接下来
>
> P(*i*,*j*)=(*P*(*i*+1,*j*−1)&&*S*[*i*]==*S*[*j*])

Without optimization on space

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s)==0:
            return ""
        
        # for "cbbd", since "bb" is a palindrom, it can be represented by dp[1][2]==1
        dp = [[0 for _ in range(len(s))] for _ in range(len(s))]
        res = s[0]
        
        for i in range(len(s)-1, -1, -1): # why we do this in a reversed order? As here, dp[i][j] = dp[i+1][j-1] and s[i]==s[j], we need the i+1 first to get i 
            for j in range(i, len(s)):
                dp[i][j] = s[i]==s[j] and (j-i<=1 or dp[i+1][j-1])
                if dp[i][j] and j-i+1>len(res):
                    res = s[i:j+1]
                    
        return res
```





> 当求第 i 行的时候我们只需要第 i + 1 行的信息，并且 j 的话需要 j - 1 的信息，所以倒着遍历这样我们可以把二维数组转为用一维数组

```python
public String longestPalindrome7(String s) {
  int n = s.length();
  String res = "";
     boolean[] P = new boolean[n];
     for (int i = n - 1; i >= 0; i--) {
         for (int j = n - 1; j >= i; j--) {
             P[j] = s.charAt(i) == s.charAt(j) && (j - i < 3 || P[j - 1]);
           //j - i < 3 意味着字符串长度小于等于3，很好理解
           //P[j - 1] 上面那个圆圈是由下面的圆圈决定的，因为下面圆圈代表2起始3终点的substring是否palindromic
             if (P[j] && j - i + 1 > res.length()) {
                 res = s.substring(i, j + 1);
             }
         }
     }
     return res;
 }
```





#### Solution-Manacher Algorithm-worth doing and thinking

https://www.cnblogs.com/grandyang/p/4475985.html

https://www.zhihu.com/question/37289584

*This is also a method to make use of already known palindromes.*

**main idea**:

Add # to string:

For an odd number length example, the original string is "bob". After adding "#", it becomes "#b#o#b#". Also, we add a "$" before it, that is "\$#b#o#b#", we call it "s1". And the number of chars in such palindrome is always odd, as # would be inserted between every char as well as the beginning and the end.

In this way, the index of the longest palindrome here is "\$#b#o#b#", and the center is "o" with an index in s1 as 4, with a radius of 3(not including the center), the index of the beginning palindrome without #$ in the "bob" is 0 

For an even number length example, the original string is "122223". After adding "#", it becomes "#1#2#2#2#2#3#". Also, we add a "$" before it, that is "\$#1#2#2#2#2#3#", we call it "s2".

In this way, the index of the longest palindrome here is "#2#2#2#2#", and the center is "#" with an index in s2 as 7, with a radius of 4(not including the center), the index of the beginning palindrome without #$ in the "122223" is 1 

And we found out, the radius is the length of the palindrome(why? Cuz the radius doesn't include the center, and the number of # is always one more than the actual letters); **the difference** of the new index of the center minus the radius **and then divided by 2 (i.e. the quotient) is the palindrome's begining character index** in original string.(not the residual)



**notation**:
ma: strings after adding "#" in it

mp[i]: the maximum radius of the palindrome with the center of i th character

mx: the most right position of already known palindromes

id: the center of the palindrome with the most right position in above

**main problem**: How to update mp[i]?(here we make use of the already known palindromes in the min part: min(mp[2 * id - i], mx - i))

If mx<=i, we can only use mp[i]=0, and then we would compare the characters beside it step by step

If mx>i, we can make use of the index that is symmatric to i with the center of id.

```
mp[i] = mx > i ? min(mp[2 * id - i], mx - i) : 1;
```





![image-20210518124241205](/Users/leslieren/Library/Application Support/typora-user-images/image-20210518124241205.png)

##### Questions 

* Why we always remember the palindrome with the most right position?

  !!!As the one with righter boarder will be more useful, because you always want the previous known palindrome to cover more area extending from your current center.

* **Why we add the $**

  Not necessary at all

* Why we add the #

  The padding is useful when we are in the situation of a palindrome with even number of letters. To add it at the beginning and the end, it is convenient for the calculation.



```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        modifiedS = "$#"
        for i in s:
            modifiedS += i+"#"
        
        mp = [0 for i in range(len(modifiedS))]    
        rightBoarder = 0
        rightCenter = 0
        maxRadius = 0
        res = ""
        
        for i in range(len(modifiedS)):
            if rightBoarder>i:
                mp[i] = min(mp[2*rightCenter-i], rightBoarder-i)
            else:
                mp[i] = 0
            
            while i+1+mp[i]<len(modifiedS) and i-1-mp[i]>=0 and modifiedS[i+1+mp[i]]==modifiedS[i-1-mp[i]]:
                mp[i]+=1
            
            if i+mp[i]>rightBoarder:
                rightBoarder = i+mp[i]
                rightCenter = i
            if mp[i]>maxRadius: # update if this is a longer palindrom than the current longest one
                maxRadius = mp[i]
                beginIndex = (i-mp[i])//2
                res = s[beginIndex:beginIndex+maxRadius]
                    
        return res
```





#### solution-扩展中心, 比普通动态规划快很多

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

### 9. Palindrome Number

https://leetcode.com/problems/palindrome-number/

#### Solution-Use what we did in problem 7, reverse integer, then compare

Ref: https://leetcode.wang/leetCode-9-Palindrome-Number.html

#### Solution-只要翻转后半部分，和前面相等则True

Ref: https://leetcode.wang/leetCode-9-Palindrome-Number.html



### 214. Shortest Palindrome-$

https://leetcode.com/problems/shortest-palindrome/

* Solution-Brute force-O(N2)-可以想想

另一种暴力破解, ref: https://leetcode.wang/leetcode-214-Shortest-Palindrome.html

> 先判断整个字符串是不是回文串，如果是的话，就直接将当前字符串返回。不是的话，进行下一步。
>
> 判断去掉末尾 `1` 个字符的字符串是不是回文串，如果是的话，就将末尾的 `1` 个字符加到原字符串的头部返回。不是的话，进行下一步。
>
> 判断去掉末尾 `2` 个字符的字符串是不是回文串，如果是的话，就将末尾的 `2` 个字符倒置后加到原字符串的头部返回。不是的话，进行下一步。
>
> 判断去掉末尾 `3` 个字符的字符串是不是回文串，如果是的话，就将末尾的 `3` 个字符倒置后加到原字符串的头部返回。不是的话，进行下一步。

Ref: https://leetcode.com/problems/shortest-palindrome/discuss/60099/AC-in-288-ms-simple-brute-force

```python
def shortestPalindrome(self, s):
    r = s[::-1]
    for i in range(len(s) + 1):
        if s.startswith(r[i:]):
            return r[:i] + s
```

* Solution-Manacher Algorithm-worth doing and thinking

https://leetcode.wang/leetcode-214-Shortest-Palindrome.html

* Solution-recursive-***worth thinking and doing*** @12.23

Ref: https://leetcode.com/problems/shortest-palindrome/discuss/60250/My-recursive-Python-solution

https://leetcode.wang/leetcode-214-Shortest-Palindrome.html.  解法2

```python
	  if not s or len(s) == 1:
        return s
    j = 0
    for i in reversed(range(len(s))):
        if s[i] == s[j]:
            j += 1
    return s[::-1][:len(s)-j] + self.shortestPalindrome(s[:j-len(s)]) + s[j-len(s):]
```

j一定可以走出最长回文串

* Solution-KMP, Knuth–Morris–Pratt, **worth doing and thinking**

[Here](http://jakeboxer.com/blog/2009/12/13/the-knuth-morris-pratt-algorithm-in-my-own-words/) is a good explaination on what The Partial Match Table is and how to use the table

Why would this work?

As we move right, all we care about is if we would miss any substring that will match, see the following for example

The string is XXXXOOO, and pattern is XXXX&

X are the substring that matches between string and pattern, each X can represent different letters, and O in string are the and & in pattern that don't match, which can also be any letters.

When we at index 4, we didn't match anymore. 

The matched letters are in bold from here to illustrate. For string **XXXX**OOO, for pattern, **XXXX**&, when we move pattern forward, what we want to prevent is that we will miss subffixes in "XXXX" in string that matches prefixes in pattern. Right now, the partial_match_length is 4 which gives us a maximum steps we will move. And let's see one by one, if we just move one step and there is a match, what we have is the X**XXX**OOO matches **XXX**X&,  in this way, **XXX**X& matches X**XXX**&, and the length of current mapping substring is exactly the table[partial_match_length - 1] which is 3, and we will only move partial_match_length - table[partial_match_length - 1] steps, that is one step.





### 336. Palindrome Pairs-$

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



* Solution-trie

Ref: http://www.allenlipeng47.com/blog/index.php/2016/03/15/palindrome-pairs/





### 131. Palindrome Partitioning-$

https://leetcode.com/problems/palindrome-partitioning/

#### Solution-recursive-memorization

did@21.6.11

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        
        return self.helper(s, 0, dict())
        
    def helper(self, s, start, dic):
        if start==len(s):
            return [[]]
        if start in dic:
            return dic[start]
        res = []
        for i in range(start+1, len(s)+1):
            string = s[start:i]
            if string==string[::-1]:
                subres = self.helper(s, i, dic)
                for array in subres:
                    res.append([string]+array)
        dic[start] = res            
        return res
```

#### Solution-dp

did@21.6.11

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        dp = [[] for i in range(len(s)+1)]
        dp[0].append([])
        
        for i in range(1, len(dp)):
            for j in range(0, i):
                string = s[j:i]
                if string==string[::-1]:
                    for before in dp[j]:
                        dp[i].append(before+[string])
        return dp[-1]
```



by huahua: 优化的问题通常用dp或dfs

#### Solution- divide and conquer-**worth doing and thinking**@12.24

@3.16不需要helper函数啦

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        if len(s)==1:
            return [[s]]
        res = []
        
        if s==s[::-1]:
            res.append([s])
      
        for i in range(1, len(s)):
            left = s[:i]
            if left==left[::-1]:
            
                for right in self.partition(s[i:]):
                    res.append([left]+right)
                    
        return res
```



#### solution-dfs, backtrack-**worth doing and thinking**@12.24@3.16

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

### 132. Palindrome Partitioning II-$

https://leetcode.com/problems/palindrome-partitioning-ii/

* Solution- dynamic programming-**worth doing and thinking**

https://www.cnblogs.com/grandyang/p/4271456.html

> 一维的dp数组，其中dp[i]表示子串 [0, i] 范围内的最小分割数，那么我们最终要返回的就是 dp[n-1] 了.
>
> 并且加个corner case的判断，若s串为空，直接返回0。
>
> 而如何更新dp[i], 这个区间的每个位置都可以尝试分割开来，所以就用一个变量j来从0遍历到i，这样就可以把区间 [0, i] 分为两部分，[0, j-1] 和 [j, i]。而因为我们从前往后更新，所以我们已经知道区间 [0, j-1] 的最小分割数 dp[j-1]， 这样我们就只需要判断区间 [j, i] 内的子串是否为回文串了。



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



### 267. Palindrome Permutation II

https://leetcode.com/problems/palindrome-permutation-ii/

* Solution-backtrack-**worth doing medium**, 对backtrack的理解还不到位@12.25

https://leetcode.com/problems/palindrome-permutation-ii/discuss/120631/Short-Python-Solution-with-backtracking



## Parentheses

### 20. Valid Parentheses

https://leetcode.com/problems/valid-parentheses/

* Solution-easy, 稍微想一下就好



### 22. Generate Parentheses

https://leetcode.com/problems/generate-parentheses/

* Solution-backtrack

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        self.helper(n,n,"",res)
        return res
        
    def helper(self, left, right, string, res):
        if left==0 and right ==0:
            res.append(string)
        
        if left>0:
            self.helper(left-1, right, string+"(", res)
        if left==0: # optimised
            res.append(string+right*")")
        elif right>left:
            self.helper(left, right-1, string+")", res)
```



12.17

### 32. Longest Valid Parentheses-$

https://leetcode.com/problems/longest-valid-parentheses/

* solution normal dp

* Solution-dynamic programming

@3.18做出来了，但我也还没想特别清楚

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        if not s:
            return 0
        
        res = 0
        stack = []
        dp = [0]*len(s)
        i = 0
        
        for i in range(len(s)):
            if s[i]=='(':
                stack.append(i)
                continue
            
            if stack:
                leftIndex = stack.pop()
                dp[i] = i-leftIndex+1+dp[leftIndex-1]
                
        return max(dp)
```



Ref: https://leetcode.wang/leetCode-32-Longest-Valid-Parentheses.html

dp [ i ] 代表以下标 i 结尾的合法序列的最长长度

* solution-stack with dp-**worth thinking and doing**, 想明白了@1.2

Ref: https://leetcode.wang/leetCode-32-Longest-Valid-Parentheses.html

Ref: https://leetcode.com/problems/longest-valid-parentheses/discuss/14312/my-ten-lines-python-solution/896329

My explanation [here](https://leetcode.com/problems/longest-valid-parentheses/discuss/14312/My-ten-lines-python-solution/477058)



### 241. Different Ways to Add Parentheses

https://leetcode.com/problems/different-ways-to-add-parentheses/

* Solution-divide and conquer- **worth thinking and doing**@1.2!!!循环找分割点,@3.19会做了

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





### 301. Remove Invalid Parentheses-$$

https://leetcode.com/problems/remove-invalid-parentheses/

#### Solution-BFS

Ref: https://leetcode.com/problems/remove-invalid-parentheses/discuss/75032/Share-my-Java-BFS-solution

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        queue = collections.deque()
        queue.append(s)
        seen = set()
        res = list()
        temp = self.if_valid("(())()")
        
        while queue:
            string = queue.popleft()
            if self.if_valid(string):
                res.append(string)
                
            if len(res)!=0:
                continue
            for i,char in enumerate(string):
                if char not in "()":
                    continue
                new_string = string[:i]+string[i+1:]
                if new_string not in seen:
                    seen.add(new_string)
                    queue.append(new_string)
        return res
        
        
        
    def if_valid(self, string):
        count = 0
        
        for char in string:
            if char=="(":
                count+=1
            elif char==")":
                count-=1
            if count<0:
                return False
            
        return count==0
```

 Time complexity, suppose the original length of the string is N, at level 0, we have one string with length N, we have 1\*N = N, at level 1, we can have at most N candidates with length N-1, at level 2, we can have at most N\*(N-1) candidates with length N-2

So at level i, we at most have  C(N, i)  candidates, and (N-i) length for each candidate, which gives us a C(N, i+1) time complexity at each level, and i ranges from 0 to N in worst case. So in total, time complexity is 2 to the power of N.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grynr5es7lj308u052glo.jpg" alt="image-20210628155236184" style="zoom:50%;" />

Ref of sum computation: https://stats.stackexchange.com/questions/27266/simplify-sum-of-combinations-with-same-n-all-possible-values-of-k



#### Solution-DFS- **worth thinking and doing**

Ref: https://www.youtube.com/watch?v=2k_rS_u6EBk

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
                # 随便先移除谁都行其实！！！！！@21.6.28
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

### 392. Is Subsequence-followup-$

https://leetcode.com/problems/is-subsequence/

Follow-up:

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        charDic = collections.defaultdict(list)
        for i in range(len(t)):
            charDic[t[i]].append(i)
        
        
        sDic = collections.defaultdict(int)
        
        prev = [-1]
        for char in s:
            if char not in charDic:
                return False
            
            while sDic[char]<len(charDic[char]) and charDic[char][sDic[char]]<=prev[-1]:
                sDic[char]+=1
            if sDic[char]>=len(charDic[char]):
                return False
            else:
                prev.append(charDic[char][sDic[char]])
                sDic[char]+=1
                
                
        return True
```



### 115. Distinct Subsequences-$

https://leetcode.com/problems/distinct-subsequences/

#### Solution-recursion

did@21.6.8

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        return self.helper(s, t, 0, 0, dict())
        
    def helper(self, s, t, start1, start2, dic):
        if start2==len(t):
            return 1
        if (start1, start2) in dic:
            return dic[(start1, start2)]
        
        remain_len = len(t)-start2
        temp = 0
        for i in range(start1, len(s)-remain_len+1):
            if s[i]==t[start2]:
                temp+=self.helper(s, t, i+1, start2+1, dic)
                
        dic[(start1, start2)] = temp    
        return temp
```



#### solution-dynamic programming-**worth thinking and doing**

@3.19做出来啦

Space: O(n)

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        lastRow = [1 for i in range(len(s)+1)]
        
        
        for i in range(1, len(t)+1):
            curRow = [0 for i in range(len(s)+1)]
            for j in range(1, len(s)+1):
                if t[i-1]==s[j-1]:
                    curRow[j] = lastRow[j-1] + curRow[j-1]
                else:
                    curRow[j] = curRow[j-1]
            lastRow = curRow
                    
        return lastRow[-1]
```



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



### 187. Repeated DNA Sequences-$$bit看一下

https://leetcode.com/problems/repeated-dna-sequences/

* Solution -hash table, bit manipulation-又忘了字符串操作@1.02

利用bit的点是，通过bit进行一个key的滑动操作

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

