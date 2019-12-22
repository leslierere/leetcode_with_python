#### 69. Sqrt(x)

https://leetcode.com/problems/sqrtx/description/



#### 167. Two Sum II - Input array is sorted

https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/



#### 50. Pow(x, n)

https://leetcode.com/problems/powx-n/description/



#### 367. Valid Perfect Square

https://leetcode.com/problems/valid-perfect-square/description/







##遗忘的语法

* lambda

  匿名函数：lambda x: x*x 表示一下代码

  ```python
  def f(x):
      return x * x
  ```

  

* 列表生成器*List Comprehensions*（760）

  生成[1x1, 2x2, 3x3, ..., 10x10], 把要生成的元素`x * x`放到前面，后面跟`for`循环，for循环后面还可以加上if判断，这样我们就可以筛选出仅偶数的平方：

  ```python
  >>>[x * x for x in range(1, 11)]
  [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
  >>> [x * x for x in range(1, 11) if x % 2 == 0]
  [4, 16, 36, 64, 100]
  ```

  列表生成式也可以使用两个变量来生成list:

  ```python
  >>> d = {'x': 'A', 'y': 'B', 'z': 'C' }
  >>> [k + '=' + v for k, v in d.items()]
  ['y=B', 'x=A', 'z=C']
  ```

  

  leetcode:

  ```python
  def numJewelsInStones(self, J: str, S: str) -> int:
          import collections 
          stones = collections.Counter(S)
          return sum([stones[j] for j in J])
  ```

  Ref: https://leetcode.com/problems/jewels-and-stones/discuss/327540/Python-hashmap-esay-solution

  



## 概念理解

* hashtable&dictionary

  > A dictionary is a general concept that maps keys to values. There are many ways to implement such a mapping. A hashtable is a specific way to implement a dictionary. 
  >
  > In python, dictionary is a hash table.

  (ref: [https://stackoverflow.com/questions/2061222/what-is-the-true-difference-between-a-dictionary-and-a-hash-table](https://stackoverflow.com/questions/2061222/what-is-the-true-difference-between-a-dictionary-and-a-hash-table), https://mail.python.org/pipermail/python-list/2000-March/031607.html)



## Binary search

* 对left和right的替换

  我的习惯是，right = len(n)-1, while left>right, left = mid + 1, right  = mid

* 先对特殊值处理会更快（374）

* 使用inorder traversal来实现（744）

  to be continued

* 使用two pointer来实现（167

  大概比较适合有两个list的，类似的概念，总之用到两个元素

  

* 避免溢出（278）

  > 那就是如果left和right都特别大的话，那么left+right可能会溢出，我们的处理方法就是变成left + (right - left) / 2，很好的避免的溢出问题

  Ref: https://www.cnblogs.com/grandyang/p/4790469.html

