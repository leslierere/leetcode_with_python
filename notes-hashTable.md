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



### 7.19

#### 599. Minimum Index Sum of Two Lists

https://leetcode.com/problems/minimum-index-sum-of-two-lists/

* solution

  有时候并不需要对两个list都做hash处理

* solution

  求相同值可以用交集来做

#### 202. Happy Number

https://leetcode.com/problems/happy-number/

* 别老想着用recursion来做，循环就够了

#### 720. Longest Word in Dictionary

https://leetcode.com/problems/longest-word-in-dictionary/



#### 1. Two Sum

https://leetcode.com/problems/two-sum/

* solution

  ```python
  class Solution(object):
      def twoSum(self, nums, target):
          if len(nums) <= 1:
              return False
          buff_dict = {}
          for i in range(len(nums)):
              if nums[i] in buff_dict:
                  return [buff_dict[nums[i]], i]
              else:
                  buff_dict[target - nums[i]] = i
  ```



#### 594. Longest Harmonious Subsequence

https://leetcode.com/problems/longest-harmonious-subsequence/

* solution

  > 我们其实也可以在一个 for 循环中搞定，遍历每个数字时，先累加其映射值，然后查找该数字加1是否存在，存在的话用 m[num] 和 m[num+1] 的和来更新结果 res，同时，还要查找该数字减1是否存在，存在的话用 m[num] 和 m[num-1] 的和来更新结果 res，这样也是可以的
  >
  > Ref: https://www.cnblogs.com/grandyang/p/6896799.html

* solution

  > 下面方法不用任何 map，但是需要对数组进行排序，当数组有序了之后，我们就可以一次遍历搞定了。这实际上用到了滑动窗口 Sliding Window 的思想，用变量 start 记录当前窗口的左边界，初始化为0。用 new_start 指向下一个潜在窗口的左边界，初始化为0。i为当前窗口的右边界，从1开始遍历，首先验证当前窗口的差值是否小于1，用 nums[i] 减去  nums[start]，若不满足，则将 start 赋值为 new_start，即移动到下一个窗口。然后看当前数字跟之前一个数字是否相等，若不相等，说明当前数字可能是下一个潜在窗口的左边界，将 new_start 赋值为i。然后再看窗口的左右边界值是否刚好为1，因为题目中说了差值必须正好为1，由于我们对数组排序了，所以只要左右边界差值正好为1，那么这个窗口包含的数字就可以组成满足题意的子序列，用其长度来更新结果 res 即可
  >
  > Ref: https://www.cnblogs.com/grandyang/p/6896799.html

### 7.20

#### 645. Set Mismatch

https://leetcode.com/problems/set-mismatch/



#### 246. Strobogrammatic Number

https://leetcode.com/problems/strobogrammatic-number/



#### 734. Sentence Similarity

https://leetcode.com/problems/sentence-similarity/



#### 970. Powerful Integers

https://leetcode.com/problems/powerful-integers/

* solution-***DFS, 也就是用stack来实现***

  ```python
  class Solution(object):
      def powerfulIntegers(self, x, y, bound):
          """
          :type x: int
          :type y: int
          :type bound: int
          :rtype: List[int]
          """
          s = set()
          stack = [(0, 0)]
          while stack:
              i, j = stack.pop()
              t = x ** i + y ** j
              if t <= bound:
                  s.add(t)
                  if x > 1:
                      stack.append((i+1, j))
                  if y > 1:
                      stack.append((i, j+1))
          
          return list(s)
  ```

  



#### 205. Isomorphic Strings

https://leetcode.com/problems/isomorphic-strings/

* solution

  Hashtable记录上一次出现该字母的位置https://www.cnblogs.com/grandyang/p/4465779.html



### 7.21

#### 624. Maximum Distance in Arrays

https://leetcode.com/problems/maximum-distance-in-arrays/

* Solution-避免两个list从同一个array取出

  ```python
  class Solution:
      def maxDistance(self, arrays):
          res, curMin, curMax = 0, 10000, -10000
          for a in arrays :
              res = max(res, max(a[-1]-curMin, curMax-a[0]))
              curMin, curMax = min(curMin, a[0]), max(curMax, a[-1])
          return res
  ```

  

#### 219. Contains Duplicate II

https://leetcode.com/problems/contains-duplicate-ii/

* solution

  Ref: [https://leetcode.com/problems/contains-duplicate-ii/discuss/61375/Python-concise-solution-with-dictionary.](https://leetcode.com/problems/contains-duplicate-ii/discuss/61375/Python-concise-solution-with-dictionary.)

  ```python
  def containsNearbyDuplicate(self, nums, k):
      dic = {}
      for i, v in enumerate(nums):
          if v in dic and i - dic[v] <= k:
              return True
          dic[v] = i
      return False
  ```



#### 290. Word Pattern

https://leetcode.com/problems/word-pattern/



#### 170. Two Sum III - Data structure design

https://leetcode.com/problems/two-sum-iii-data-structure-design/



#### 438. Find All Anagrams in a String

https://leetcode.com/problems/find-all-anagrams-in-a-string/



### 7.22

#### 204. Count Primes

https://leetcode.com/problems/count-primes/

实现同样的功能，for循环比while快







##语法

#### Container

##### list

- 排序

  * sorted()

  It returns a new sorted list

  ```python
  >>> sorted([5, 2, 3, 1, 4])
  [1, 2, 3, 4, 5]
  ```

  * [`list.sort()`](https://docs.python.org/3/library/stdtypes.html#list.sort)

  It modifies the list in-place (and returns `None`).

  Another difference is that the `list.sort()` method is only defined for lists. In contrast, the `sorted()`function accepts any iterable.

- `numbers = list(range(1, 6, 2)) #[1, 3, 5]`

- 复制列表

  `list2 = list1[:]`

##### tuple

* 如果要定义一个空的tuple，可以写成`()`

  ```pytho
  >>> t = ()
  >>> t
  ()
  ```

* 只有1个元素的tuple定义时必须加一个逗号`,`，来消除歧义：

  ```python
  >>> t = (1,)
  >>> t
  (1,)
  ```

##### dict

- dict的get()

  如果key不存在，可以返回`None`，或者自己指定的value

  ```python
  >>> d.get('Thomas')
  >>> d.get('Thomas', -1)
  -1
  ```

##### set

set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。要创建一个set，需要提供一个list作为输入集合.

* 将一个字符串变为单个字母的set

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

 

#### collection module

##### collections.Counter()

Ref: https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter

一个 [`Counter`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter) 是一个 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 的子类，用于计数可哈希对象。它是一个集合，元素像字典键(key)一样存储，它们的计数存储为值。计数可以是任何整数值，包括0和负数。

```python
s='apple'
count = collections.Counter(s) 
In [15]: count                                                                  
Out[15]: Counter({'a': 1, 'p': 2, 'l': 1, 'e': 1})

In [17]: c = collections.Counter({'red': 4, 'blue': 2})                         
In [18]: c                                                                      
Out[18]: Counter({'red': 4, 'blue': 2})
  
In [19]: c =collections.Counter(cats=4, dogs=8)                                 
In [20]: c                                                                      
Out[20]: Counter({'cats': 4, 'dogs': 8})
```



* Counter对象有一个字典接口，如果引用的键没有任何记录，就返回一个0，而不是弹出一个 [`KeyError`](https://docs.python.org/zh-cn/3/library/exceptions.html#KeyError)

  ```python
  >>> c = Counter(['eggs', 'ham'])
  >>> c['bacon']# count of a missing element is zero
  0
  ```

* 使用 `del` 来删除key

* elements()

  返回一个迭代器，每个元素重复计数的个数。元素顺序是任意的。如果一个元素的计数小于1， [`elements()`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter.elements) 就会忽略它。

  ```python
  >>> c = Counter(a=4, b=2, c=0, d=-2)
  >>> sorted(c.elements())
  ['a', 'a', 'a', 'a', 'b', 'b']
  ```

* 通常字典方法都可用于 [`Counter`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter) 对象

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

* 常用方法

```python
In [20]: c                                                                      
Out[20]: Counter({'cats': 4, 'dogs': 8})
  
sum(c.values())                 # total of all counts
c.clear()                       # reset all counts
list(c)                         # list unique elements
In [22]: list(c)                                                                  
Out[22]: ['cats', 'dogs']
  
set(c)                          # convert to a set
In [21]: set(c)                                                                   
Out[21]: {'cats', 'dogs'}
  
dict(c)                         # convert to a regular dictionary
c.items()                       # convert to a list of (elem, cnt) pairs
In [23]: c.items()                                                                
Out[23]: dict_items([('cats', 4), ('dogs', 8)])
  
+c                              # remove zero and negative counts
>>> c = Counter(a=2, b=-4)
>>> +c
Counter({'a': 2})
```

* 数学操作

```python
>>> c = Counter(a=3, b=1)
>>> d = Counter(a=1, b=2)
>>> c + d                       # add two counters together:  c[x] + d[x]
Counter({'a': 4, 'b': 3})
>>> c - d                       # subtract (keeping only positive counts)
Counter({'a': 2})
>>> c & d                       # intersection:  min(c[x], d[x]) 
Counter({'a': 1, 'b': 1})
>>> c | d                       # union:  max(c[x], d[x])
Counter({'a': 3, 'b': 2})
```



##### collections.defaultdict()

* *class* `collections.defaultdict`([*default_factory*[, *...*]])

  返回一个新的类似字典的对象。 [`defaultdict`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict) 是内置 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 类的子类。它重载了一个方法并添加了一个可写的实例变量。其余的功能与 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 类相同，此处不再重复说明。

  第一个参数 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory) 提供了一个初始值。它默认为 `None` 。所有的其他参数都等同与 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 构建器中的参数对待，包括关键词参数。

* 使用 [`list`](https://docs.python.org/zh-cn/3/library/stdtypes.html#list) 作为 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory)

```python
In [24]: s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]    
In [26]: d = collections.defaultdict(list)                                           
In [27]: for k, v in s: 
    ...:     d[k].append(v) 
    ...:                                                                             
In [28]: d                                                                           
Out[28]: defaultdict(list, {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]})
# 当每个键第一次遇见时，它还没有在字典里面；所以条目自动创建，通过 default_factory 方法，并返回一个空的 list 。 list.append() 操作添加值到这个新的列表里。当键再次被存取时，就正常操作， list.append() 添加另一个值到列表中。这个计数比它的等价方法 dict.setdefault() 要快速和简单
```

* 设置 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory) 为 [`int`](https://docs.python.org/zh-cn/3/library/functions.html#int) ，使 [`defaultdict`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict) 在计数方面发挥好的作用（像其他语言中的bag或multiset）

```python
In [32]: d = collections.defaultdict(int)                                            
In [33]: s = 'mississippi'                                                           
In [34]: d = collections.defaultdict(int)                                            

In [35]: for k in s: 
    ...:     d[k] += 1 
    ...:                                                                             
In [36]: d                                                                           
Out[36]: defaultdict(int, {'m': 1, 'i': 4, 's': 4, 'p': 2})
# 当一个字母首次遇到时，它就查询失败，所以 default_factory 调用 int() 来提供一个整数0作为默认值。自增操作然后建立对每个字母的计数。
```

* 设置 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory) 为 [`set`](https://docs.python.org/zh-cn/3/library/stdtypes.html#set) 使 [`defaultdict`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict) 用于构建字典集合

```python
>>> s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
>>> d = defaultdict(set)
>>> for k, v in s:
...     d[k].add(v)
...
>>> sorted(d.items())
[('blue', {2, 4}), ('red', {1, 3})]
```





#### join的用法

* join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

  The string whose method is called is inserted in between each given string.

  The result is returned as a new string.

  ​    Example: '.'.join(['ab', 'pq', 'rs']) -> 'ab.pq.rs'

  

  

#### Ascii码和character的转换

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

* Bit Operation(Python)

  * ~ Binary Ones Complement

    flips all the bits

  * & Binary AND

    bits are turned on only if both original bits are turned on

    > Used with 1, it basically masks the value to extract the lowest bit, or in other words will tell you if the value is even or odd.

  * |Binary OR

    It copies a bit if it exists in either operand.

  * ^ Binary XOR(excluesive or)

    turned on only if exaclty one of the original bits are turned on

  * \>\> Binary Right Shift

    Bitwise right shift, it shifts the bits to the right by the specified number of places.

  ```python
  a = 60            # 60 = 0011 1100 
  b = 13            # 13 = 0000 1101 
  
  c = a & b;        # 12 = 0000 1100
  
  c = a | b;        # 61 = 0011 1101 
  
  c = a ^ b;        # 49 = 0011 0001
  
  c = ~a;           # -61 = 1100 0011
  
  c = a << 2;       # 240 = 1111 0000
  
  c = a >> 2;       # 15 = 0000 1111
  ```

  

* bool()

  The following values are considered false in Python:

  - `None`
  - `False`
  - Zero of any numeric type. For example, `0`, `0.0`, `0j`
  - Empty sequence. For example, `()`, `[]`, `''`.
  - Empty mapping. For example, `{}`
  - objects of Classes which has `__bool__()` or `__len()__` method which returns `0` or `False`

* all()

  https://leetcode.com/problems/sentence-similarity/discuss/109621/Trivial-Python-solution-using-set-comprehension

* max()

  https://leetcode.com/problems/longest-word-in-dictionary/discuss/186128/O(1)-Space!-NlogN-Solution!-PythonC%2B%2B

* zip()

  [https://leetcode.com/problems/sentence-similarity/discuss/109624/Simple-Python-Solution-32ms!](https://leetcode.com/problems/sentence-similarity/discuss/109624/Simple-Python-Solution-32ms!)

  

* Python可以直接比较object吗？

  Counter 对不存在的直接加？