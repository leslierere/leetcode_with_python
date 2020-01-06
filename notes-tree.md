1.02

#### 144.Binary Tree Preorder Traversal

https://leetcode.com/problems/binary-tree-preorder-traversal/description/

* Solution-statck, iterative

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = [root]
        
        while stack:
            last = stack.pop()
            if last:# for there be null value there
                res.append(last.val)
                stack.append(last.right)
                stack.append(last.left)
        return res
      
# 更快的一种
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        queue = []
        curr = root
        ret = []
        while True:
            if curr:
                queue.append(curr)
                ret.append(curr.val)
                curr = curr.left
            elif queue:
                tmp = queue.pop() 
                curr = tmp.right
            else:
                break               
        return ret
```

* Solution-recuresive

```python
def preorderTraversal1(self, root):
    res = []
    self.dfs(root, res)
    return res
    
def dfs(self, root, res):
    if root:
        res.append(root.val)
        self.dfs(root.left, res)
        self.dfs(root.right, res)
```



* Solution-Morris Traversal-***worth doing and thinking***

```java
// Step 1: Initialize current as root
// Step 2: While current is not NULL,
// If current does not have left child
	// a. Add current’s value
	// b. Go to the right, i.e., current = current.right
// Else
	//a. In current's left subtree, make "current" the right child of the rightmost node
	//b. Go to this left child, i.e., current = current.left
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    TreeNode cur = root;
    while (cur != null) {
        //情况 1
        if (cur.left == null) {
            list.add(cur.val);
            cur = cur.right;
        } else {
            //找左子树最右边的节点
            TreeNode pre = cur.left;
            while (pre.right != null && pre.right != cur) {
                pre = pre.right;
            }
            //情况 2.1
            if (pre.right == null) {
                list.add(cur.val);
                pre.right = cur;
                cur = cur.left;
            }
            //情况 2.2
            if (pre.right == cur) {
                pre.right = null; //这里可以恢复为 null
                cur = cur.right;
            }
        }
    }
    return list;
}


```



#### 94. Binary Tree Inorder Traversal

https://leetcode.com/problems/binary-tree-inorder-traversal/description/

* Solution- iterative, stack，试着写写

https://leetcode.wang/leetCode-94-Binary-Tree-Inorder-Traversal.html

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        cur = root
        
        while cur!= None or stack:
            
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
            
        return res
```

* solution-recursive, 试着写写



1.03

#### 145. Binary Tree Postorder Traversal-用这道作为所有的tree的iterative模版, 改到前面的基础里面去

https://leetcode.com/problems/binary-tree-postorder-traversal/description/

* Solution-iterative

https://www.youtube.com/watch?v=A6iCX_5xiU4

<img src="/Users/leslieren/Library/Application Support/typora-user-images/image-20200103114811047.png" alt="image-20200103114811047" style="zoom:80%;" />

```python
# 上面的核心思想是先做一个rev_postorder(root),这样就和最后要的结果刚好相反，但我们可以通过特殊操作，使用deque来使得最后的结果不需要reverse
# my solution based on huahua's
# 这一面很重要
from collections import deque
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        res = collections.deque()
        stack = [root]
        
        while stack:
            top = stack.pop()
            res.appendleft(top.val)
            if top.left:
                stack.append(top.left)
            if top.right:
                stack.append(top.right)
        
        return res
```



* Solution-recursive-worth doing





#### 102. Binary Tree Level Order Traversal

https://leetcode.com/problems/binary-tree-level-order-traversal/description/

* Solution-bfs-need speed up

https://www.youtube.com/watch?v=Tuij96VBdu8

* Solution-dfs-worth trying



#### 100. Same Tree-用这道作为所有的tree的recursive模版

https://leetcode.com/problems/same-tree/

* Solution-iterative-32ms
* Solution-recursive-worth doing, 我觉得我对回归有点问题，这个

```python
# solution里面的解法
# 要把握住相同的时候的最终态为两个node为null
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        
        if not p and not q:
            return True
        if not p or not q: # one of p and q is None
            return False
        if p.val!=q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```



#### 101. Symmetric Tree

https://leetcode.com/problems/symmetric-tree/description/

* Solution-recursive

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.helper(root.left, root.right)
        
    def helper(self, l, r):
        if not l and not r:
            return True
        if not l or not r:
            return False
        if l.val!=r.val:
            return False
        return self.helper(l.left, r.right) and self.helper(l.right, r.left)
        # 我开始把上面一句写的是
        # self.helper(l.left, r.right)
        # self.helper(l.right, r.left)
```



* Solution-iterative

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.helper(root.left, root.right)
    
    def helper(self, l, r):
        stack1 = [l]
        stack2 = [r]
        
        while stack1 and stack2:
            top1 = stack1.pop()
            top2 = stack2.pop()
            
            if top1 and top2:
                if top1.val!=top2.val:
                    return False
                stack1.append(top1.left)
                stack2.append(top2.right)
                stack1.append(top1.right)
                stack2.append(top2.left)
            elif top1 or top2:
                return False
        if stack1 or stack2:
            return False
        return True
```



#### 226. Invert Binary Tree

https://leetcode.com/problems/invert-binary-tree/description/

* Solution-recursive

```python
# by me, 28ms
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root:
            root.left, root.right = root.right, root.left
            self.invertTree(root.left)
            self.invertTree(root.right)
        return root
# 24ms
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root:
            root.right, root.left = root.left, root.right
            self.invertTree(root.left)
            self.invertTree(root.right)
            return root
```



#### 257. Binary Tree Paths

https://leetcode.com/problems/binary-tree-paths/

* solution-bfs, iterative, queue-下次把字符串加进去
* Solution-dfs, recursive, worth doing
* Solution-dfs, iterative, stack, worth doing



#### 112. Path Sum

https://leetcode.com/problems/path-sum/description/

* Solution-dfs, recursive, 做出来了

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        
        return self.dfs(root, 0, sum)

        
    def dfs(self, node, agg, sum):
        if node:
            if not node.left and not node.right and agg+node.val==sum:
                return True
            agg += node.val
            return self.dfs(node.left, agg, sum) or self.dfs(node.right, agg, sum)
```



* Solution-dfs, iterative, stack, worth doing
* solution-bfs, iterative, queue，worth doing 



#### 113. Path Sum II

https://leetcode.com/problems/path-sum-ii/description/

* Solution-dfs, recursive, 做出来了, memory很高

```python
class Solution:
    def pathSum(self, root: TreeNode, agg: int) -> List[List[int]]:
        res = []
        
        def dfs(path, node):
            if node:
                if not node.left and not node.right:
                    if agg==sum(path)+node.val:
                        res.append(path+[node.val])
                else:
                    dfs(path+[node.val], node.left)
                    dfs(path+[node.val], node.right)
                    
        dfs([], root)
        return res
```



* Solution-dfs, iterative, stack, worth doing
* solution-bfs, iterative, queue，worth doing 



1.4

#### 129. Sum Root to Leaf Numbers

https://leetcode.com/problems/sum-root-to-leaf-numbers/description/

* Solution-dfs, recursive, 做出来了

* Solution-dfs, iterative, stack, need speed up
* solution-bfs, iterative, queue，worth doing 



#### 298. Binary Tree Longest Consecutive Sequence

https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/description/

* Solution-dfs, recursive, worth doing
* Solution-dfs, stack, 做出来了

```python
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        stack = [(root, float('inf'), 1)]
        res = 1
        
        while stack:
            node, value, layer = stack.pop()
            if not node.left and not node.right:
                if node.val-1 == value:
                    layer+=1
                res = max(res, layer)
            if node.left:
                if node.val-1 == value:
                    stack.append((node.left, node.val, layer+1))
                else:
                    res = max(res, layer)
                    stack.append((node.left, node.val, 1))
            if node.right:
                if node.val-1 == value:
                    stack.append((node.right, node.val, layer+1))
                else:
                    res = max(res, layer)
                    stack.append((node.right, node.val, 1))
                    
        return res
```

* solution-bfs, iterative, queue，worth doing 

```python
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        queue = [(root, float("inf"), 1)] 
        # the middle element remembers the value of parent node
        res = 1
        
        while queue:
            size = len(queue)
            for i in range(size):
                node, value, layer = queue.pop(0)
                
                if not node.left and not node.right:
                    res = max(res, layer+1) if node.val-1 == value else max(res, layer)
                if node.left:
                    if node.val-1 == value:
                        queue.append((node.left, node.val, layer+1))
                    else:
                        res = max(res, layer)
                        queue.append((node.left, node.val, 1))
                if node.right:
                    if node.val-1 == value:
                        queue.append((node.right, node.val, layer+1))
                    else:
                        res = max(res, layer)
                        queue.append((node.right, node.val, 1))
                        
        return res
```



#### 111. Minimum Depth of Binary Tree-可以作为bfs的范例了

https://leetcode.com/problems/minimum-depth-of-binary-tree/description/

* Solution-dfs, recursive, 没做

* Solution-dfs, iterative, stack, 没做
* solution-bfs, iterative, queue

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        queue = [(root, 0)]
        
        while queue:
            node, layer = queue.pop(0)
            if not node.left and not node.right:
                return layer+1
            if node.left:
                queue.append((node.left, layer+1))
            if node.right:
                queue.append((node.right, layer+1))
```



#### 104. Maximum Depth of Binary Tree

https://leetcode.com/problems/maximum-depth-of-binary-tree/description/

* Solution-dfs, recursive, 没做

* Solution-dfs, iterative, stack, 没做
* solution-bfs, iterative, queue, 做了



#### 110. Balanced Binary Tree

https://leetcode.com/problems/balanced-binary-tree/description/

* Solution-recursive-O(n) worth doing and thinking

Ref: http://zxi.mytechroad.com/blog/leetcode/leetcode-110-balanced-binary-tree/

```python
# based on huahua's, O(nlogn)
# 这个时间范围度值得思考
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:
            return True
        left_height = self.height(root.left)
        right_height = self.height(root.right)
        
        return abs(left_height-right_height)<2 and self.isBalanced(root.left) and self.isBalanced(root.right)
        
    def height(self, root):
        if not root:
            return 0
        return max(1+self.height(root.left), 1+self.height(root.right))
      
# based on huahua's, O(logn)
# 求解高度时一边看左右节点高度是否平衡
# @1.4 我开始错的原因是balanced进来是拷贝变量，所以需要使用全局变量或者返回
```

<img src="/Users/leslieren/Library/Application Support/typora-user-images/image-20200104152112111.png" alt="image-20200104152112111" style="zoom:30%;" />

最差的情况是左右两边除了一边最下面一个不平衡其他都平衡

* Solution-dfs, iterative, stack
* solution-bfs, iterative, queue



#### 124. Binary Tree Maximum Path Sum

https://leetcode.com/problems/binary-tree-maximum-path-sum/description/

>  My understanding is that a valid path is a "straight line" that connects all the nodes, in other words, it can't "fork".
>
> Ref: https://leetcode.com/problems/binary-tree-maximum-path-sum/discuss/39811/What-is-the-meaning-of-path-in-this-problem

* Solution-recursive

```python
# based on huahua's
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.res = float("-inf")
        self.dfs(root)
        return self.res
        
    def dfs(self, node):
        if not node:
            return float("-inf") # or if the leaf has a negative value, it will still work
        
        left = max(self.dfs(node.left), 0) # 0 suggests if the left child's max is a negative number, we don't consider them anymore
        right = max(self.dfs(node.right), 0)
        self.res = max(self.res, node.val+left+right)
        return node.val + max(left, right)
```



#### 250. Count Univalue Subtrees

https://leetcode.com/problems/count-univalue-subtrees/description/

* Solution-bottom up-by myself, 下次用手拿test case写一遍

```python
class Solution:
    res = 0
    
    def countUnivalSubtrees(self, root: TreeNode) -> int:
        
        self.helper(root)
        return self.res
        
    def helper(self, node):
        if node:
            if not node.left and not node.right:
                self.res += 1
                return node.val
            if node.left and node.right:
                left = self.helper(node.left)
                right = self.helper(node.right)
                if left!= node.val or right!=node.val:
                    return None
            elif node.left:
                left = self.helper(node.left)
                if left!= node.val:
                    return None
            else:
                right = self.helper(node.right)
                if right!= node.val:
                    return None
            
            self.res+=1
            return node.val
```



#### 366. Find Leaves of Binary Tree

https://leetcode.com/problems/find-leaves-of-binary-tree/description/

* solution- **worth doing and thinking**

Ref: https://leetcode.com/problems/find-leaves-of-binary-tree/discuss/83778/10-lines-simple-Java-solution-using-recursion-with-explanation

```python
class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        res = []
        
        def height(node):
            if not node:
                return -1
            h = max(height(node.left)+1, height(node.right)+1)
            if len(res) < h+1:
                res.append([])
            res[h].append(node.val)
            return h
        
        height(root)
        return res
```





#### 337. House Robber III

https://leetcode.com/problems/house-robber-iii/description/

* Solution-**worth doing and thinking!!! look at the reference**

ref: https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem





1.5

下次接下来三道可试试dfs

#### 107. Binary Tree Level Order Traversal II

https://leetcode.com/problems/binary-tree-level-order-traversal-ii/

* solution-bfs，和103差不多，但可以通过一些方法不用deque



#### 103. Binary Tree Zigzag Level Order Traversal

https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/

* solution-bfs-和107差不多

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        queue = [root]
        left = True
        
        while queue:
            if left:
                res.append([i.val for i in queue])
            else:
                res.append([queue[i].val for i in range(len(queue)-1, -1, -1)])
            
            size = len(queue)
            
            for i in range(size):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            left = not left
            
        return res
```



#### 199. Binary Tree Right Side View

https://leetcode.com/problems/binary-tree-right-side-view/description/

* Solution-bfs

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        res = []
        
        queue = [root]
        
        while queue:
            res.append(queue[-1].val)
            size = len(queue)
            
            for i in range(size):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
        return res
```



#### 98. Validate Binary Search Tree

https://leetcode.com/problems/validate-binary-search-tree/

* Solution-inorder

```python
class Solution:
    def isValidBST(self, node: TreeNode) -> bool:
        if not node:
            return True
        return self.helper(node, float('-inf')) != float("inf")
        
        
    def helper(self, node, flag):
        if node:
            if node.left:
                cur = self.helper(node.left, flag)
                if not cur >flag:
                    return float("inf")
                else:
                    flag = cur
                    
            if not node.val> flag:
                return float("inf")
            flag = node.val
            
            if node.right:
                cur = self.helper(node.right, flag)
                if not cur >flag:
                    return float("inf")
                else:
                    flag = cur
            return flag
```



#### 235. Lowest Common Ancestor of a Binary Search Tree

https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/

* solution-recursive, iterative， worth doing@下次做

利用好bst的特性

Ref: https://leetcode.com/articles/lowest-common-ancestor-of-a-binary-search-tree/



#### 236. Lowest Common Ancestor of a Binary Tree

https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

* Solution-dfs-**worth thinking and doing**

Ref: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/discuss/158060/Python-DFS-tm

https://www.cnblogs.com/grandyang/p/4641968.html



#### 108. Convert Sorted Array to Binary Search Tree

* Solution-dfs, recursive, need speed up

值得一看：https://leetcode.wang/leetcode-108-Convert-Sorted-Array-to-Binary-Search-Tree.html

> 递归都可以转为迭代的形式。
>
> 一部分递归算法，可以转成动态规划，实现空间换时间，例如 [5题](https://leetcode.windliang.cc/leetCode-5-Longest-Palindromic-Substring.html)，[10题](https://leetcode.windliang.cc/leetCode-10-Regular-Expression-Matching.html)，[53题](https://leetcode.windliang.cc/leetCode-53-Maximum-Subarray.html?h=动态规划)，[72题](https://leetcode.wang/leetCode-72-Edit-Distance.html)，从自顶向下再向顶改为了自底向上。
>
> 一部分递归算法，只是可以用栈去模仿递归的过程，对于时间或空间的复杂度没有任何好处，比如这道题，唯一好处可能就是能让我们更清楚的了解递归的过程吧。



> 但这样有一个缺点，我们知道`int`的最大值是 `Integer.MAX_VALUE` ，也就是`2147483647`。那么有一个问题，如果 `start = 2147483645`，`end = 2147483645`，虽然 `start` 和 `end`都没有超出最大值，但是如果利用上边的公式，加起来的话就会造成溢出，从而导致`mid`计算错误。
>
> 解决的一个方案就是利用数学上的技巧，我们可以加一个 `start` 再减一个 `start` 将公式变形。
>
> ```python
> (start + end) / 2 = (start + end + start - start) / 2 = start + (end - start) / 2
> ```
>
> 所以python有溢出的问题吗？



* Solution-dfs, stack, worth doing

Ref: https://leetcode.wang/leetcode-108-Convert-Sorted-Array-to-Binary-Search-Tree.html



#### 109. Convert Sorted List to Binary Search Tree

https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/

* Solution-先把链表转为list
* Solution-快慢指针 **worth doing and thinking**

Ref: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/discuss/35474/Python-recursive-solution-with-detailed-comments-(operate-linked-list-directly).

* Solution-厉害！！**worth doing and thinking**

Ref: https://leetcode.wang/leetcode-109-Convert-Sorted-List-to-Binary-Search-Tree.html





#### 173. Binary Search Tree Iterator

https://leetcode.com/problems/binary-search-tree-iterator/description/

* Solution-**worth doing and thinking** 其实就是94题, inorder traversal

Ref: https://leetcode.wang/leetcode-173-Binary-Search-Tree-Iterator.html

刚开始弄错是因为初始化的时候我就把root加了进去



#### 230. Kth Smallest Element in a BST

https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/

Ref: https://www.cnblogs.com/grandyang/p/4620012.html

* Solution-**worth doing and thinking** 其实就是94题,  inorder traversal, iterative

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        
        while root or stack:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()
            k-=1
            if k==0:
                return root.val
            
            root = root.right
```



* Solution-  inorder traversal, recursive
* Solution- divide and conquer-**worth doing and thinking**

* Solution-**Follow up:**
  What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

B+tree的思想

Ref: https://leetcode.com/articles/kth-smallest-element-in-a-bst/









### Notion

#### Traversal

ref: https://www.youtube.com/watch?v=A6iCX_5xiU4

![image-20200103121202462](/Users/leslieren/Library/Application Support/typora-user-images/image-20200103121202462.png)

* In a **_preorder_** traversal, you visit each node before recursively visiting its children, which are visited from left to right. The root is visited first.

  A preorder traversal is a natural way to print a directory’s structure. Simply have the method visit() *print each node of the tree*.

```java
class SibTreeNode {
  public void preorder() { 
    this.visit(); if (firstChild != null) { 
      firstChild.preorder(); 
    } 
    if (nextSibling != null) { 
      nextSibling.preorder(); 
    }
  }
}
```

* In a **_postorder_** traversal, you visit each node’s children (in left-to-right order) before the node itself.

```java
public void postorder() {
  if (firstChild != null) { 
    firstChild.postorder(); 
  } 
  this.visit(); 
  if (nextSibling != null) {
    nextSibling.postorder(); 
  }
}
```

​	A postorder traversal visits the nodes in this order.

<img src="/Users/leslieren/Library/Application Support/typora-user-images/image-20200102202316343.png" alt="image-20200102202316343" style="zoom:40%;" />

​	The postorder() code is trickier than it looks. The best way to understand it is to draw a depth-two tree on paper, then pretend you’re the computer and execute the algorithm carefully. Trust me on this. It’s worth your time.

​	A postorder traversal is the *natural way to sum the total disk space used in the root directory and its descendants*. The method visit() sums "this" node’s disk space with the disk space of all its children. In the example above, a postorder traversal would first sum the sizes of the files in hw1/ and hw2/; then it would visit hw/ and sum its two children. The last thing it would compute is the total disk space at the root ˜jrs/61b/, which sums all the files in the tree.

* Binary trees allow for an **_inorder_** traversal: recursively traverse the root’s left subtree (rooted at the left child), then the root itself, then the root’s right subtree.
* In a **_level-order_** traversal, you visit the root, then all the depth-1 nodes (from left to right), then all the depth-2 nodes, et cetera. The level-order traversal of our expression tree is "+ * ^ 3 7 4 2" (which doesn’t mean much).Unlike the three previous traversals, a level-order traversal is not straightforward to define recursively. However, a level-order traversal can be done in O(n) time. Use a queue, which initially contains only the root. Then repeat the following steps:
  - Dequeue a node.
  - Visit it.
  - Enqueue its children (in order from left to right). Continue until the queue is empty.

**A final thought**: if you use a stack instead of a queue, and push each node’s children in reverse order--from right to left (so they pop off the stack in order from left to right)--you perform a preorder traversal. Think about why.



#### deque

https://docs.python.org/zh-cn/3/library/collections.html#collections.deque

*class* `collections.deque([*iterable*[, *maxlen*]])`

双向队列对象

```python
>>> from collections import deque
>>> d = deque('ghi')                 # make a new deque with three items
>>> d.append('j')                    # add a new entry to the right side
>>> d.appendleft('f')                # add a new entry to the left side
>>> d                                # show the representation of the deque
deque(['f', 'g', 'h', 'i', 'j'])

>>> d.pop()                          # return and remove the rightmost item
'j'
>>> d.popleft()                      # return and remove the leftmost item
'f'
>>> list(d)                          # list the contents of the deque
['g', 'h', 'i']
>>> d[0]                             # peek at leftmost item
'g'
>>> d[-1]                            # peek at rightmost item
'i'

>>> list(reversed(d))                # list the contents of a deque in reverse
['i', 'h', 'g']
>>> 'h' in d                         # search the deque
True
>>> d.extend('jkl')                  # add multiple elements at once
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> d.rotate(1)                      # right rotation
>>> d
deque(['l', 'g', 'h', 'i', 'j', 'k'])
>>> d.rotate(-1)                     # left rotation
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])

>>> deque(reversed(d))               # make a new deque in reverse order
deque(['l', 'k', 'j', 'i', 'h', 'g'])
>>> d.clear()                        # empty the deque
>>> d.pop()                          # cannot pop from an empty deque
Traceback (most recent call last):
    File "<pyshell#6>", line 1, in -toplevel-
        d.pop()
IndexError: pop from an empty deque

>>> d.extendleft('abc')              # extendleft() reverses the input order
>>> d
deque(['c', 'b', 'a'])
```





#### queue模块

https://docs.python.org/3/library/queue.html

* *class* `queue.Queue(*maxsize=0*)`

Constructor for a **FIFO** queue. *maxsize* is an integer that sets the upperbound limit on the number of items that can be placed in the queue. Insertion will block once this size has been reached, until queue items are consumed. If *maxsize* is less than or equal to zero, the queue size is infinite.

* *class* `queue.LifoQueue(maxsize=0)`

Constructor for a **LIFO** queue. 

* class `queue.PriorityQueue(maxsize=0)`

Constructor for a priority queue. The lowest valued entries are retrieved first (the lowest valued entry is the one returned by `sorted(list(entries))[0]`).

##### Public methods for [`Queue`](https://docs.python.org/3/library/queue.html#queue.Queue), [`LifoQueue`](https://docs.python.org/3/library/queue.html#queue.LifoQueue), or [`PriorityQueue`](https://docs.python.org/3/library/queue.html#queue.PriorityQueue)

* `Queue.qsize()`

  Return the approximate size of the queue. Note, qsize() > 0 doesn’t guarantee that a subsequent get() will not block, nor will qsize() < maxsize guarantee that put() will not block.

* `Queue.empty()`

  Return `True` if the queue is empty, `False` otherwise.

* `Queue.full()`

  Return `True` if the queue is full, `False` otherwise. 

* `Queue.put(item, block=True, timeout=None)`

  Put item into the queue. If optional args block is true and timeout is None (the default), block if necessary until a free slot is available. If timeout is a positive number, it blocks at most timeout seconds and raises the Full exception if no free slot was available within that time. Otherwise (block is false), put an item on the queue if a free slot is immediately available, else raise the Full exception (timeout is ignored in that case).

* `Queue.get(*block=True*, *timeout=None*)`

  Remove and return an item from the queue. 

* `Queue.join()`

  Blocks until all items in the queue have been gotten and processed.