##Practice

###7.15

####500. Keyboard Row

https://leetcode.com/problems/keyboard-row/        

* solution

  ```python
      class Solution:
      def findWords(self, words: List[str]) -> List[str]:
          s1 = set("QWERTYUIOPqwertyuiop")
          s2 = set("ASDFGHJKLasdfghjkl")
          s3 = set("ZXCVBNMzxcvbnm")
      r = []
      for word in words:
          s = set(word)
          if ((s&s1) == s) or ((s&s2) == s) or ((s&s3) == s):
              r.append(word)
      return r
  ```

####463. Island Perimeter

https://leetcode.com/problems/island-perimeter/

* solution

  对每个格子四个边处理

  ```python
  class Solution:
      def islandPerimeter(self, grid: List[List[int]]) -> int:
          m = len(grid)
          n = len(grid[0])
          res = 0
          
          for i in range(m):
              for j in range(n):
                  if grid[i][j] == 1:
                      if j ==0 or grid[i][j-1]==0:#left
                          res+=1
                      if i ==0 or grid[i-1][j]==0:#top
                          res+=1
                      if j == n-1 or grid[i][j+1]==0:#right
                          res+=1
                      if i == m-1 or grid[i+1][j]==0:#bottom
                          res+=1
          return res
  ```



####359.Logger Rate Limiter

https://leetcode.com/problems/logger-rate-limiter/



#### 1002. Find Common Characters

https://leetcode.com/problems/find-common-characters/



#### 811. Subdomain Visit Count-unsubmitted

https://leetcode.com/problems/subdomain-visit-count/



#### 1078. Occurrences After Bigram

https://leetcode.com/problems/occurrences-after-bigram/



#### 961. N-Repeated Element in Size 2N Array

https://leetcode.com/problems/n-repeated-element-in-size-2n-array/solution/

* solution-math

  ```python
  class Solution:
  	def repeatedNTimes(self, A: List[int]) -> int:
  		B = set(A)
  		return (sum(A) - sum(B)) // (len(A) - len(B))
  ```



#### 1086. High Five

https://leetcode.com/problems/high-five/

* Solution-start with sorting items by scores

  ref: https://leetcode.com/problems/high-five/discuss/334179/python-solution-beats-94.78-in-time-and-100-in-space

```python
class Solution(object):
    def highFive(self, items):
        items.sort(key=lambda x: x[1])
        output = []
        freq = {}
        scores = {}
        i = len(items) - 1
        while i >= 0:
            if items[i][0] in freq:
                freq[items[i][0]] += 1
                if freq[items[i][0]] <= 5:
                    scores[items[i][0]] += items[i][1]
            else:
                freq[items[i][0]] = 1
                scores[items[i][0]] = items[i][1]
            i -= 1
            
        for key in freq:
            output.append([key, scores[key] // 5])
        return sorted(output, key=lambda x:x[0])
        """
        :type items: List[List[int]]
        :rtype: List[List[int]]
        """
```



### 7.16

#### 884. Uncommon Words from Two Sentences

https://leetcode.com/problems/uncommon-words-from-two-sentences/

* solution

  ```python
    class Solution(object):
      def uncommonFromSentences(self, A, B):
          count = {}
          for word in A.split():
              count[word] = count.get(word, 0) + 1
          for word in B.split():
              count[word] = count.get(word, 0) + 1  
    	#Alternatively:
      #count = collections.Counter(A.split())
      #count += collections.Counter(B.split())
  
      		return [word for word in count if count[word] == 1]
  ```

#### 136. Single Number-favorites

https://leetcode.com/problems/single-number/

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

* solution1-Bit Manipulation

***Concept***：

1. *If we take XOR of zero and some bit, it will return that bit*

   a*⊕0=*a

2. If we take XOR of two same bits, it will return 0

   *a*⊕*a*=0

3. *a*⊕*b*⊕*a*=(*a*⊕*a*)⊕*b*=0⊕*b*=*b*

* Solution2-hashtable，存在删除，不存在加入

  ```python
  class Solution(object):
      def singleNumber(self, nums):
          """
          :type nums: List[int]
          :rtype: int
          """
          a = 0
          for i in nums:
              a ^= i
          return a
  ```

  



#### 266. Palindrome Permutation

https://leetcode.com/problems/palindrome-permutation/solution/

* Solution1- bit manipulation

  > 我们建立一个 256 大小的 bitset，每个字母根据其 ASCII 码值的不同都有其对应的位置，然后我们遍历整个字符串，遇到一个字符，就将其对应的位置的二进制数 flip 一下，就是0变1，1变0，那么遍历完成后，所有出现次数为偶数的对应位置还应该为0，而出现次数为奇数的时候，对应位置就为1了，那么我们最后只要统计1的个数，就知道出现次数为奇数的字母的个数了，只要个数小于2就是回文数
  >
  > ref: https://www.cnblogs.com/grandyang/p/5223238.html

* solution2-同136, 和奇偶数相关的都可用类似方法

  > 那么我们再来看一种解法，这种方法用到了一个 HashSet，我们遍历字符串，如果某个字母不在 HashSet 中，我们加入这个字母，如果字母已经存在，我们删除该字母，那么最终如果 HashSet 中没有字母或是只有一个字母时，说明是回文串，参见代码如下：
  >
  > ref: https://www.cnblogs.com/grandyang/p/5223238.html

  

#### 575. Distribute Candies

https://leetcode.com/problems/distribute-candies/

* solution

```python
class Solution:
    def distributeCandies(self, candies: List[int]) -> int:
        length1 = len(candies)//2
        s = set(candies)
        length2 = len(s)
        if length2 < length1:
            return length2
        else:
            return length1
```







#### 706. Design HashMap

https://leetcode.com/problems/design-hashmap/

* solution-array+linked list-my answer

  一般有Time Limit Exceeded就一定是循环卡死，所以要记得return，迭代什么的

  ```python
  class Node:
      def __init__(self, key, val):
          self.key = key
          self.val = val
          self.next = None
  
  class MyHashMap:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.array = [None]*10000
          
  
      def put(self, key: int, value: int) -> None:
          """
          value will always be non-negative.
          """
          index = key%10000
          if self.array[index] == None:
              self.array[index] = Node(key, value)
          else:
              n = self.array[index]
              while n!= None:
                  if n.key==key:
                      n.val = value
                      return
                  elif n.next == None:
                      n.next = Node(key, value)
                      ##
                      return
                  else:
                      n = n.next
              
  
      def get(self, key: int) -> int:
          """
          Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
          """
          index = key%10000
          n = self.array[index]
          while n!= None:
              if key == n.key:
                  return n.val
              n = n.next
          return -1
          
  
      def remove(self, key: int) -> None:
          """
          Removes the mapping of the specified value key if this map contains a mapping for the key
          """
          index = key%10000
          
          last = self.array[index]
          if last == None:
              return
          elif last.key == key:
              self.array[index] = last.next
              return
          while last.next!=None:
              if last.next.key == key:
                  last.next = last.next.next
                  return
              else:
                  last = last.next
          
  
          
  
  
  # Your MyHashMap object will be instantiated and called as such:
  # obj = MyHashMap()
  # obj.put(key,value)
  # param_2 = obj.get(key)
  # obj.remove(key)
  ```

  

### 7.17

#### 690. Employee Importance

https://leetcode.com/problems/employee-importance/

* solution

  有点像tree的结构，还可以用queue的方法来做，还没做



### 7.18

#### 705. Design HashSet

https://leetcode.com/problems/design-hashset/



#### 409. Longest Palindrome

https://leetcode.com/problems/longest-palindrome/

* solution

  ```python
  def longestPalindrome(self, s):
      odds = sum(v & 1 for v in collections.Counter(s).values())
      # v&1如果某个字母的个数为单数，即v为单数时，列表中相应元素为1，否则为0
      return len(s) - odds + bool(odds)
  ```



#### 447. Number of Boomerangs

https://leetcode.com/problems/number-of-boomerangs/



#### 387. First Unique Character in a String

https://leetcode.com/problems/first-unique-character-in-a-string/



#### 217. Contains Duplicate

https://leetcode.com/problems/contains-duplicate/



#### 242. Valid Anagram

https://leetcode.com/problems/valid-anagram/

找所含元素相同的方法，加减很常用



#### 389. Find the Difference

https://leetcode.com/problems/find-the-difference/

* solution

  找出一个不同的字母，而其他都为双数，可以考虑bit operation



#### 690. Employee Importance

https://leetcode.com/problems/employee-importance/

* solution

  意识到数据结构为tree的可以用BFS（queue）或者DFS来做

  还没做，ref：https://leetcode.com/problems/employee-importance/discuss/112611/3-liner-Python-Solution-(beats-99)

  ```python
  class Solution(object):
      def getImportance(self, employees, id):
          """
          :type employees: Employee
          :type id: int
          :rtype: int
          """
          # Time: O(n)
          # Space: O(n)
          emps = {employee.id: employee for employee in employees}
          def dfs(id):
              subordinates_importance = sum([dfs(sub_id) for sub_id in emps[id].subordinates])
              return subordinates_importance + emps[id].importance
          return dfs(id)
        
  ```

  

##语法

* set()将一个字符串变为单个字母的set

```python
s1 = set("QWERTYUIOPqwertyuiop")

print(s1)
#{'T', 'e', 't', 'P', 'O', 'r', 'u', 'Y', 'I', 'Q', 'q', 'y', 'p', 'W', 'R', 'w', 'i', 'E', 'o', 'U'}
```

* set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作：

```python
>>> s1 = set([1, 2, 3])
>>> s2 = set([2, 3, 4])
>>> s1 & s2
{2, 3}
>>> s1 | s2
{1, 2, 3, 4}
```

 

* collections.Counter()

  ```python
  class Solution:
      def firstUniqChar(self, s: str) -> int:
          count = collections.Counter(s)
          l = count.values()#返回每个值个数的一个list
          
          for k, v in enumerate(s):
              if count[v]==1:
                  return k
          return -1
  ```

  * &=也可以用在Counter对象上，获得交集（相当于分别变成list再取包含重复数的交集https://leetcode.com/problems/find-common-characters/discuss/247560/Python-1-Line

* join的用法：join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

  The string whose method is called is inserted in between each given string.

  The result is returned as a new string.

  ​    Example: '.'.join(['ab', 'pq', 'rs']) -> 'ab.pq.rs'

* dict的get()

  如果key不存在，可以返回`None`，或者自己指定的value

  ```python
  >>> d.get('Thomas')
  >>> d.get('Thomas', -1)
  -1
  ```

* Ascii码和character的转换

  ```python
  # Get the ASCII number of a character
  number = ord(char)
  
  # Get the character given by an ASCII number
  char = chr(number)
  ```

* Return

  函数体内部可以用`return`随时返回函数结果；

  函数执行完毕也没有`return`语句时，自动`return None`。

  函数可以同时返回多个值，但其实就是一个tuple。
  
* isAlpha()

  检测字符串是否只由字母组成

  `str.isalpha()`

* count()

  ```python
  list.count(obj)
  str.count(sub, start= 0,end=len(string))
  #sub -- 搜索的子字符串
  #start -- 字符串开始搜索的位置。默认为第一个字符,第一个字符索引值为0。
  #end -- 字符串中结束搜索的位置。字符中第一个字符的索引为 0。默认为字符串的最后一个位置。
  ```

* enumerate()

  用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

  `enumerate(sequence, [start=0])`

* Bit Operation

  * ~ 

    flips all the bits

  * &

    bits are turned on only if both original bits are turned on

    > Used with 1, it basically masks the value to extract the lowest bit, or in other words will tell you if the value is even or odd.

  * ^

    XOR(excluesive or), turned on only if exaclty one of the original bits are turned on

* bool()

  The following values are considered false in Python:

  - `None`
  - `False`
  - Zero of any numeric type. For example, `0`, `0.0`, `0j`
  - Empty sequence. For example, `()`, `[]`, `''`.
  - Empty mapping. For example, `{}`
  - objects of Classes which has `__bool__()` or `__len()__` method which returns `0` or `False`