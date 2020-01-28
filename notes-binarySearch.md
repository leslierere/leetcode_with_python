#### 69. Sqrt(x)

https://leetcode.com/problems/sqrtx/description/



#### 167. Two Sum II - Input array is sorted

https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/



#### 50. Pow(x, n)

https://leetcode.com/problems/powx-n/description/



#### 367. Valid Perfect Square

https://leetcode.com/problems/valid-perfect-square/description/









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

