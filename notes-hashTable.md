####500. Keyboard Row

https://leetcode.com/problems/keyboard-row/        

用set的这个特性, 想想就行

```python
# s1 is subset of s2, return boolean
s1 <= s2

# s1 is proper subset of s2
s1 < s2
```



####359.Logger Rate Limiter

https://leetcode.com/problems/logger-rate-limiter/



#### 1002. Find Common Characters

https://leetcode.com/problems/find-common-characters/

使用counter的&和elements(), 想一想思路就好





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



#### 884. Uncommon Words from Two Sentences-想想

https://leetcode.com/problems/uncommon-words-from-two-sentences/

* solution

  Ref: https://leetcode.com/problems/uncommon-words-from-two-sentences/discuss/158967/C%2B%2BJavaPython-Easy-Solution-with-Explanation
  
  这个把复杂化简单的思路太强了！





#### 266. Palindrome Permutation

想想怎么one pass

https://leetcode.com/problems/palindrome-permutation/solution/

* Solution1- bit manipulation，本质跟solution2没有区别

  > 我们建立一个 256 大小的 bitset，每个字母根据其 ASCII 码值的不同都有其对应的位置，然后我们遍历整个字符串，遇到一个字符，就将其对应的位置的二进制数 flip 一下，就是0变1，1变0，那么遍历完成后，所有出现次数为偶数的对应位置还应该为0，而出现次数为奇数的时候，对应位置就为1了，那么我们最后只要统计1的个数，就知道出现次数为奇数的字母的个数了，只要个数小于2就是回文数
  >
  > ref: https://www.cnblogs.com/grandyang/p/5223238.html

* solution2-同136, 和奇偶数相关的都可用类似方法

  > 那么我们再来看一种解法，这种方法用到了一个 HashSet，我们遍历字符串，如果某个字母不在 HashSet 中，我们加入这个字母，如果字母已经存在，我们删除该字母，那么最终如果 HashSet 中没有字母或是只有一个字母时，说明是回文串，参见代码如下：
  >
  > ref: https://www.cnblogs.com/grandyang/p/5223238.html

  



#### 706. Design HashMap

https://leetcode.com/problems/design-hashmap/





#### 690. Employee Importance

https://leetcode.com/problems/employee-importance/

* solution

  有点像tree的结构，还可以用queue的方法来做，还没做



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



#### 204. Count Primes

https://leetcode.com/problems/count-primes/

实现同样的功能，for循环比while快






