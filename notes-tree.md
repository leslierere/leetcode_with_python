1.02

#### 144.Binary Tree Preorder Traversal

https://leetcode.com/problems/binary-tree-preorder-traversal/description/

* Solution-statck, iterative

```python
# 这个思路，将左右子树分别压栈，然后每次从栈里取元素。需要注意的是，因为我们应该先访问左子树，而栈的话是先进后出，所以我们压栈先压右子树
# 但这样inorder就不太行
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
      
# 上下两个写法没有啥差别，但比第三个写法慢。。。
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        res = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
                
        return res
      
# 时间10.63%, 模拟递归，入栈的时候加值
# 这个通告改value append时间可以改为inorder
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        stack, result = [], []
        while stack or root:
            while root:
                result.append(root.val)
                stack.append(root)
                root = root.left
            root = stack.pop().right

        return result
```

* Solution-recursive

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
        
        while cur or stack:
            
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

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5v7u8sj31c00u0wvg.jpg" alt="image-20200103114811047" style="zoom:80%;" />

```python
# 上面的核心思想
# 后序遍历的顺序是 左 -> 右 -> 根。
# 前序遍历的顺序是 根 -> 左 -> 右，左右其实是等价的，所以我们也可以轻松的写出 根 -> 右 -> 左 的代码。
# 然后把 根 -> 右 -> 左 逆序，就是 左 -> 右 -> 根，也就是后序遍历了
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



### 小结

基本上preorder，inorder， postorder traversal就是用recursive或者iterative(模拟递归)来做





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

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5t4tqij30xq08qmy5.jpg" alt="image-20200104152112111" style="zoom:30%;" />

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



1.6

#### 297. Serialize and Deserialize Binary Tree

https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/

我觉得这个本质上就是preorder<->treenode, 跟前面一个题类似

* Solution-preorder-**worth doing and thinking**, need speed up

Ref: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/discuss/74259/Recursive-preorder-Python-and-C%2B%2B-O(n)

尝试用bit 优化https://www.youtube.com/watch?v=JL4OjKV_pGE

By huahua, 其他人快使用的iterative

* Solution-level order/bfs-**worth doing and thinking**





#### 285. Inorder Successor in BST

https://leetcode.com/problems/inorder-successor-in-bst/

* Solution-iterative-太强了！！**worth doing and thinking**，利用好bst的特性，但还是可以尝试常规的办法

Ref: https://leetcode.com/problems/inorder-successor-in-bst/discuss/72656/JavaPython-solution-O(h)-time-and-O(1)-space-iterative

```python
def inorderSuccessor(self, root, p):
    succ = None
    while root:
        if p.val < root.val:
            succ = root
            root = root.left
        else:
            root = root.right
    return succ

```



#### 270. Closest Binary Search Tree Value

https://leetcode.com/problems/closest-binary-search-tree-value/description/

* Solution-inorder, iterative-**worth doing and thinking**

开始用了一个res来保存当前所有遍历到的，其实没必要，仔细想想这道题最快的还是inorder

```python
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        if not root:
            return None
        
        
        stack = []
        lessnode = root
        while stack or root:
            while root:
                
                stack.append(root)
                root = root.left

            root = stack.pop()

            if root.val>target:
                break
            lessnode = root# the node that is less than target
            root = root.right
        # if root, root should be the node that is just larger than target   
        if root and abs(root.val-target) < abs(target - lessnode.val):
          # 开始没加abs，而target不一定是一个中间值，可能比最小值小，可能比最大值大
            return root.val
        # if not root, which means target is larger than all the nodes
        else:
            return lessnode.val
```



* Solution-preorder, iterative, fast



#### 272. Closest Binary Search Tree Value II

https://leetcode.com/problems/closest-binary-search-tree-value-ii/description/

* Solution-inorder，先inorder-bymyself, slow

```python
class Solution:
    def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
        if not root:
            return []
        
        nodel = []
        self.inorder(root, nodel)
        res = []
        i = 0
        
        # get the index in nodel that is just larger than the target
        while i < len(nodel):
            if nodel[i]>target:
                break
            i+=1
        
        while k>0:
            if i==0:
                res.append(nodel.pop(0))
            elif i== len(nodel):
                res.append(nodel.pop(-1))
                i-=1
            else:
                if abs(nodel[i-1]-target) < abs(nodel[i]-target):
                    res.append(nodel.pop(i-1))
                    i-=1
                else:
                    res.append(nodel.pop(i))
            k -= 1
            
        return res
        
        
    def inorder(self, node, nodel):
        if node:
            self.inorder(node.left, nodel)
            nodel.append(node.val)
            self.inorder(node.right, nodel)
```

* Solution-一边inorder，一边更新, 40ms, **worth doing and thinking**

```python
from collections import deque


class Solution:
    def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
        res = deque()

        def ino(node):
            if node is None:
                return
            ino(node.left)
            if len(res) < k:
                res.append(node.val)
            elif abs(node.val - target) < abs(res[0] - target):
                res.popleft()
                res.append(node.val)
            else:
                return
            ino(node.right)

        ino(root)
        return res
```



* Solution-two stack-**worth doing and trying**



#### 99. Recover Binary Search Tree

https://leetcode.com/problems/recover-binary-search-tree/

* solution- inorder- **worth doing and thinking**

主要是得分两种情况, ref: https://leetcode.wang/leetcode-99-Recover-Binary-Search-Tree.html

> 回到这道题，题目交换了两个数字，其实就是在有序序列中交换了两个数字。而我们只需要把它还原。
>
> 交换的位置的话就是两种情况。
>
> - 相邻的两个数字交换
>
>   [ 1 2 3 4 5 ] 中 2 和 3 进行交换，[ 1 3 2 4 5 ]，这样的话只产生一组逆序的数字（正常情况是从小到大排序，交换后产生了从大到小），3 2。
>
>   我们只需要遍历数组，找到后，把这一组的两个数字进行交换即可。
>
> - 不相邻的两个数字交换
>
>   [ 1 2 3 4 5 ] 中 2 和 5 进行交换，[ 1 5 3 4 2 ]，这样的话其实就是产生了两组逆序的数字对。5 3 和 4 2。
>
>   所以我们只需要遍历数组，然后找到这两组逆序对，然后把第一组前一个数字和第二组后一个数字进行交换即完成了还原。



* Solution-morris!!!



#### 116. Populating Next Right Pointers in Each Node

https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

* solution-dfs-recursive-**worth doing and thinking**, 这个人的写法感觉抓到了关键

Ref: https://leetcode.com/problems/populating-next-right-pointers-in-each-node/discuss/37715/Python-solutions-(Recursively-BFS%2Bqueue-DFS%2Bstack)

```python
def connect1(self, root):
    if root and root.left and root.right:
        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left#这个是关键
        self.connect(root.left)
        self.connect(root.right)
```

* Solution - dis - stack
* Solution - bfs



#### 117. Populating Next Right Pointers in Each Node II

https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/

* Solution-bfs/level order, done
* Solution-recursive-看看别人的





#### 314. Binary Tree Vertical Order Traversal

https://leetcode.com/problems/binary-tree-vertical-order-traversal/description/

* Solution-hash table, **worth thinking and doing**, 很巧妙

https://leetcode.com/problems/binary-tree-vertical-order-traversal/discuss/76424/Python-solution



#### 95. Unique Binary Search Trees II

https://leetcode.com/problems/unique-binary-search-trees-ii/

* Solution-recursive, **worth thinking and doing**

https://leetcode.wang/leetCode-95-Unique-Binary-Search-TreesII.html

* Solution-dynamic programming-太强了-**worth doing and thinking**

https://leetcode.com/problems/unique-binary-search-trees-ii/discuss/31493/Java-Solution-with-DP

```java
public static List<TreeNode> generateTrees(int n) {
    List<TreeNode>[] result = new List[n + 1];
    result[0] = new ArrayList<TreeNode>();
    if (n == 0) {
        return result[0];
    }

    result[0].add(null);
    for (int len = 1; len <= n; len++) {
        result[len] = new ArrayList<TreeNode>();
        for (int j = 0; j < len; j++) {
            for (TreeNode nodeL : result[j]) {
                for (TreeNode nodeR : result[len - j - 1]) {
                    TreeNode node = new TreeNode(j + 1);
                    node.left = nodeL;
                    node.right = clone(nodeR, j + 1);
                    result[len].add(node);
                }
            }
        }
    }
    return result[n];
}

// 因为和左边结构是一样的，所以复制之前的结构，只是把值改成右边的值（反正都是连续自然数）
private static TreeNode clone(TreeNode n, int offset) {
    if (n == null) {
        return null;
    }
    TreeNode node = new TreeNode(n.val + offset);
    node.left = clone(n.left, offset);
    node.right = clone(n.right, offset);
    return node;
}
//result[i] stores the result until length i
```





#### 96. Unique Binary Search Trees

https://leetcode.com/problems/unique-binary-search-trees/description/

* solution-**Catalan number **

Ref: https://leetcode.wang/leetCode-96-Unique-Binary-Search-Trees.html

* Solutions-延续95题的做法





### Notion

#### Catalan number

Definition: 

> 令h ( 0 ) = 1，catalan 数满足递推式：
>
> **h ( n ) = h ( 0 ) \* h ( n - 1 ) + h ( 1 ) \* h ( n - 2 ) + ... + h ( n - 1 ) \* h ( 0 ) ( n >=1 )**
>
> 例如：h ( 2 ) = h ( 0 ) * h ( 1 ) + h ( 1 ) * h ( 0 ) = 1 * 1 + 1 * 1 = 2
>
> h ( 3 ) = h ( 0 ) * h ( 2 ) + h ( 1 ) * h ( 1 ) + h ( 2 ) * h ( 0 ) = 1 * 2 + 1 * 1 + 2 * 1 = 5
>
> 卡塔兰数有一个通项公式。
>
> <img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaolecv6wvj30i808qdgm.jpg" alt="image-20200107130949465" style="zoom:33%;" />



#### Traversal

ref: https://www.youtube.com/watch?v=A6iCX_5xiU4

![image-20200103121202462](https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5wbuomj31c00u0n9b.jpg)

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

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5x3gfzj308y06uaa3.jpg" alt="image-20200102202316343" style="zoom:40%;" />

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