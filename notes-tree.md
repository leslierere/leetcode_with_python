@4.2，感觉tree的部分还比较薄弱

### 144.Binary Tree Preorder Traversal-$

https://leetcode.com/problems/binary-tree-preorder-traversal/description/

#### Solution1-stack, iterative

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
      
# 上下两个写法没有啥差别，但比solution2慢。。。但不用在意吧。。。lc总是怪怪的
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
      

```

#### Solution2-iterative-模拟递归

```python
# 时间10.63%, 模拟递归，一直入栈（入栈时打印），没有了就出栈
# 这个通过改value append的位置可以改为inorder
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



#### Solution3-recursive

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



#### Solution-Morris Traversal-***worth doing and thinking***

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





### 94. Binary Tree Inorder Traversal

https://leetcode.com/problems/binary-tree-inorder-traversal/description/

#### Solution- iterative, stack，试着写写

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

#### solution-recursive, 试着写写



1.03

### 145. Binary Tree Postorder Traversal-$

https://leetcode.com/problems/binary-tree-postorder-traversal/description/

#### Solution1-iterative，对应144-solution1

https://www.youtube.com/watch?v=A6iCX_5xiU4

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5v7u8sj31c00u0wvg.jpg" alt="image-20200103114811047" style="zoom:80%;" />

```python
# 上面的核心思想
# 后序遍历的顺序是 左 -> 右 -> 根。
# 前序遍历的顺序是 根 -> 左 -> 右，左右其实是等价的，所以我们也可以轻松的写出 根 -> 右 -> 左 的代码。
# preorder: self, left, right
# postorder: left, right, self
# reverse of post order: self, right, left

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.left: 
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
                
        result.reverse()        
        return result



# 然后把 根 -> 右 -> 左 逆序，就是 左 -> 右 -> 根，也就是后序遍历了, 而这里res使用deque，这样最后我们就不需要倒叙一遍了
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
            root = stack.pop()
            res.appendleft(root.val)
            if root.left:
                stack.append(root.left)
            if root.right:
                stack.append(root.right)
        
        return res
```

#### Solution2-iterative-模拟递归, break topology

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        stack = []
        node = root
        res = collections.deque()
        
        while node or stack:
            while node:
                res.appendleft(node.val)
                stack.append(node)
                node = node.right
            node = stack.pop().left
            
        return res
```

#### Solution3-iterative

Ref: https://leetcode.com/problems/binary-tree-postorder-traversal/discuss/45551/Preorder-Inorder-and-Postorder-Iteratively-Summarization/120145

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        result = []
        stack = [root]
        pre = root # tricky, dumb, or None or root?
        
        while stack:
            top = stack[-1]
            if (not top.left and not top.right) or top.right==pre or top.left==pre: # Tricky, you may wonder, for node with both left and right children, what if its left node was just added to result, actually we don't need worry about this, as when the pre is your left node, the node currently under inspection is not you.

                pre = stack.pop()
                result.append(pre.val)
            else:   
                if top.right:
                    stack.append(top.right)
                if top.left:
                    stack.append(top.left)
                
        return result
```

![image-20210303144305495](https://tva1.sinaimg.cn/large/e6c9d24ely1go7dz8lhykj21hf0u07ag.jpg)

#### Solution-recursive-worth doing

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.helper(root, res)
        return res
        
        
    def helper(self, node, res):
        if not node:
            return
        self.helper(node.left, res)
        self.helper(node.right, res)
        res.append(node.val)
```





### 小结

Pre, in, post的subject都是指当前node啦

基本上preorder，inorder， postorder traversal的dfs就是用两种方法

* recursive
* iterative
  * 普通stack：其实就是想清楚先进后出的关系
  * 模拟递归

* Dfs
  * preorder traversal (self-left-right)
  * inorder traversal (left-self-right): stack
  * Postorder traversal (left-right-self)

* bfs
  * level-order traversal



### 102. Binary Tree Level Order Traversal

https://leetcode.com/problems/binary-tree-level-order-traversal/description/

#### Solution-bfs-need speed up

https://www.youtube.com/watch?v=Tuij96VBdu8

#### Solution-dfs-worth trying

did@8.9, 不过这个存level的方法，用dfs，bfs都行

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        dic = collections.defaultdict(list)
        self.dfs(root, 0, dic)
        return dic.values() # we can just use dic.values here cuz Changed in version 3.7: Dictionary order is guaranteed to be insertion order. See https://docs.python.org/3.7/library/stdtypes.html?highlight=dict%20values#dict.values for more details   
        
    def dfs(self, node, level, dic):
        if not node:
            return 
        dic[level].append(node.val)
        self.dfs(node.left, level+1, dic)
        self.dfs(node.right, level+1, dic)
```





### 100. Same Tree

https://leetcode.com/problems/same-tree/

#### Solution-iterative-32ms

did@8.9

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        elif not p or not q:
            return False
        
        stack = [(p, q)]
        
        while stack:
            p, q = stack.pop()
            if p.val!=q.val:
                return False
            
            if p.right and q.right:
                stack.append((p.right, q.right))
            elif p.right or q.right:
                return False
            
            if p.left and q.left:
                stack.append((p.left, q.left))
            elif p.left or q.left:
                return False
            
        return True
```



#### Solution-recursive-easy

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



### 101. Symmetric Tree-和上一题差不多-stack做法可以试试

https://leetcode.com/problems/symmetric-tree/description/

#### Solution-recursive

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



#### Solution-iterative-非模拟recursive的解法

```python
# 左右子树分别压栈
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



### 226. Invert Binary Tree-easy

https://leetcode.com/problems/invert-binary-tree/description/

#### Solution-recursive

```python
# did at 2020.8.10

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
       
        if not root:
            return None
        reverted = TreeNode(root.val)
        reverted.left = self.invertTree(root.right)
        reverted.right = self.invertTree(root.left)
        
        return reverted
        
        
# by me, 28ms
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root:
            root.left, root.right = root.right, root.left
            self.invertTree(root.left)
            self.invertTree(root.right)
        return root

```



### 257. Binary Tree Paths-worth

https://leetcode.com/problems/binary-tree-paths/

Ref: https://leetcode.com/problems/binary-tree-paths/discuss/68272/Python-solutions-(dfs%2Bstack-bfs%2Bqueue-dfs-recursively).

#### Solution-bfs, iterative, queue-下次把字符串加进去

#### Solution-dfs, recursive

did@20.8.10

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        
        result = []
        self.helper(root, "", result)
                
        return result
    
    def helper(self, node, path, result):
        path+=str(node.val)
        if not node.left and not node.right:
            result.append(path)
            return
        if node.right:
            self.helper(node.right, path+"->", result)
        if node.left:
            self.helper(node.left, path+"->", result)
```



#### Solution-dfs, iterative, stack, worth doing

did@20.8.10

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        
        result = []
        stack = [(root, "")]
        
        while stack:
            node, path = stack.pop()
            path+=str(node.val)
            if node.right:
                stack.append((node.right, path+"->"))
            if node.left:
                stack.append((node.left, path+"->"))
            if not node.left and not node.right:
                result.append(path)
                
        return result
```



### 112. Path Sum

https://leetcode.com/problems/path-sum/description/

#### Solution-dfs, recursive

did@20.8.11

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, sums: int) -> bool:
        if not root:
            return False
        return self.helper(root, sums-root.val)
        
    def helper(self, node, subSum):
        if not node.left and not node.right:
            return subSum==0
        
        if node.left and self.helper(node.left, subSum-node.left.val):
            return True
        if node.right and self.helper(node.right, subSum-node.right.val):
            return True
```



#### Solution-dfs, iterative, stack, worth doing@1.10

did@20.8.11

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, sums: int) -> bool:
        if not root:
            return False
        
        stack = [(root, root.val)]
        
        while stack:
            node, subSum = stack.pop()
            
            if not node.left and not node.right:
                if subSum==sums:
                    return True
            if node.left:
                stack.append((node.left, subSum+node.left.val))
            if node.right:
                stack.append((node.right, subSum+node.right.val))
                
                
        return False
```



#### Solution-bfs, iterative, queue，worth doing @1.10



### 113. Path Sum II

https://leetcode.com/problems/path-sum-ii/description/

#### Solution-dfs, recursive, 做出来了, memory很高

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

@20.8.11其实对于这题bfs，dfs都差不多啦

#### Solution-dfs, iterative, stack, did@1.9, 4.5

#### solution-bfs, iterative, queue，did@4.5



1.4

### 129. Sum Root to Leaf Numbers

https://leetcode.com/problems/sum-root-to-leaf-numbers/description/

#### Solution-dfs, recursive, 做出来了

@20.8.11其实对于这题bfs，dfs都差不多啦

#### Solution-dfs, iterative, stack, did@1.10@4.5

#### solution-bfs, iterative, queue



### 298. Binary Tree Longest Consecutive Sequence

https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/description/

#### Solution-dfs, recursive, did@4.5

```python
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        if not root:
            return 0
        self.res = 0
        self.dfs(root, 1)
        return self.res
        
        
    def dfs(self, node, length):
        if not node.left and not node.right:
            self.res = max(length, self.res)
        if node.left:
            if node.left.val==node.val+1:
                self.dfs(node.left, length+1)
            else:
                self.res = max(length, self.res)
                self.dfs(node.left, 1)
        if node.right:
            if node.right.val==node.val+1:
                self.dfs(node.right, length+1)
            else:
                self.res = max(length, self.res)
                self.dfs(node.right, 1)
```



#### Solution-dfs, stack, did@20.8.13

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        stack = [(root, 1)]
        maxLength = 1
        
        while stack:
            node, length = stack.pop()
            
            if node.left:
                if node.left.val==node.val+1:
                    stack.append((node.left, length+1))
                    maxLength = max(maxLength, length+1)
                else:
                    stack.append((node.left, 1))
                    
            if node.right:
                if node.right.val==node.val+1:
                    stack.append((node.right, length+1))
                    maxLength = max(maxLength, length+1)
                else:
                    stack.append((node.right, 1))
                    
        return maxLength
```

#### solution-bfs, iterative, queue, did@1.10

@20.8.11其实对于这题bfs，dfs都差不多啦

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



### 111. Minimum Depth of Binary Tree-可以作为bfs的范例了，easy

https://leetcode.com/problems/minimum-depth-of-binary-tree/description/

#### Solution-dfs, recursive, 没必要

#### Solution-dfs, iterative, stack,没必要 did@4.5

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        res = float("inf")
        
        
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            if not node.left and not node.right:
                res = min(depth, res)
                continue
            if node.right:
                stack.append((node.right, depth+1))
            if node.left:
                stack.append((node.left, depth+1))
        
        return res
```



#### solution-bfs, iterative, queue

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



### 104. Maximum Depth of Binary Tree，easy

https://leetcode.com/problems/maximum-depth-of-binary-tree/description/

#### Solution-dfs, recursive, did@20.8.15

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        return self.dfs(root)
        
    def dfs(self, node):
        if not node:
            return 0
        
        return 1+max(self.dfs(node.left), self.dfs(node.right))
```



#### Solution-dfs, iterative, stack, did@20.8.15

#### solution-bfs, iterative, queue, 做了



### 110. Balanced Binary Tree-$

https://leetcode.com/problems/balanced-binary-tree/description/

#### Solution-recursive-O(n) , 感觉复杂了

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
      

```

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5t4tqij30xq08qmy5.jpg" alt="image-20200104152112111" style="zoom:30%;" />

最差的情况是左右两边除了一边最下面一个不平衡其他都平衡

#### Solution-dfs-postorder

@4.5下次用-1替代False

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
         
        return self.dfs(root)!=False
        
        
    def dfs(self, node):
        if not node:
            return 1
        
        leftHeight = self.dfs(node.left)
        rightHeight = self.dfs(node.right)
        if leftHeight==False or rightHeight==False or abs(leftHeight-rightHeight)>1:
            return False
        return max(leftHeight, rightHeight)+1
```



#### Solution-dfs-dictionary@20.8.15

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        dic = dict()
        self.dfs(root, dic)
        return all(dic.values())
        
    def dfs(self, node, dic):
        if not node:
            return 0
        
        left = self.dfs(node.left, dic)
        right = self.dfs(node.right, dic)
        dic[node]= abs(left-right)<2
        return max(left, right)+1
```



### 124. Binary Tree Maximum Path Sum

https://leetcode.com/problems/binary-tree-maximum-path-sum/description/

>  My understanding is that a valid path is a "straight line" that connects all the nodes, in other words, it can't "fork".
>
> Ref: https://leetcode.com/problems/binary-tree-maximum-path-sum/discuss/39811/What-is-the-meaning-of-path-in-this-problem

#### Solution-recursive, 把问题分解

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



### 250. Count Univalue Subtrees

https://leetcode.com/problems/count-univalue-subtrees/description/

#### Solution-bottom up-by myself, 下次用手拿test case写一遍

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

#### 为啥这样写就会错？？？see comment

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def countUnivalSubtrees(self, root: TreeNode) -> int:
        if not root:
            return 0
        self.res = 0
        self.helper(root)
        return self.res
    
    def helper(self, node):
        if node:
            if not node.right and not node.left:
                self.res+=1
                return node.val
            if node.right and node.left:
                # left = self.helper(node.left)
                # right = self.helper(node.right)
                # as here, if self.helper(node.left)!=node.val, the program doesn't bother to analyze the right side, so it will miss some nodes
                if self.helper(node.left)!=node.val or node.val!= self.helper(node.right):
                    return None

            elif node.left:
                if self.helper(node.left)!=node.val:
                    return None
            else:
                if self.helper(node.right)!=node.val:
                    return None
            self.res+=1
            return node.val 
```





### 366. Find Leaves of Binary Tree-想一下看note

https://leetcode.com/problems/find-leaves-of-binary-tree/description/

#### solution- **worth doing and thinking**, did @1.13

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

@1.13 我做的时候建了字典，其实没有必要，因为一定是先出现更小的 height。所以出现一个新的高度就在列表中新加一个子列表就好，最后就可以直接返回结果



### 337. House Robber III

https://leetcode.com/problems/house-robber-iii/description/

#### Solution-**worth doing and thinking!!! look at the reference** @1.13还是没想到did@4.6

ref: https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem





1.5

下次接下来三道可试试dfs

### 107. Binary Tree Level Order Traversal II

https://leetcode.com/problems/binary-tree-level-order-traversal-ii/

#### Solution-bfs，和103差不多

#### Solution-dfs, 思想和366一样, @1.13

Ref: https://leetcode.com/problems/binary-tree-level-order-traversal-ii/discuss/34978/Python-solutions-(dfs-recursively-dfs%2Bstack-bfs%2Bqueue).

```python
# dfs recursively
def levelOrderBottom1(self, root):
    res = []
    self.dfs(root, 0, res)
    return res

def dfs(self, root, level, res):
    if root:
        if len(res) < level + 1:
            res.insert(0, [])
        res[-(level+1)].append(root.val)
        self.dfs(root.left, level+1, res)
        self.dfs(root.right, level+1, res)
```





### 103. Binary Tree Zigzag Level Order Traversal

https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/

#### Solution-bfs-和107差不多

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        queue = collections.deque()
        queue.append(root)
        flag = 0
        
        while queue:
            length = len(queue)
            layer = []
            for _ in range(length):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                layer.append(node.val)
            if flag:
                layer.reverse()
            res.append(layer)       
            flag = flag^1
            
        return res
```



#### Solution-dfs, did@1.13





### 199. Binary Tree Right Side View-想一下就好

https://leetcode.com/problems/binary-tree-right-side-view/description/

#### Solution-bfs

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

#### Solution-dfs, did@1.13, 下次想一下思路就好

不过居然更慢了。。。

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        res = []
        self.dfs(root, 1, res)
        return res
        
    def dfs(self, node, level, res):
        if not node:
            return
        if len(res)<level:
            res.append(node.val)
        self.dfs(node.right, level+1, res)
        self.dfs(node.left, level+1, res)
```





### 98. Validate Binary Search Tree-$$

You can do 333 instead, they are essentially the same.

https://leetcode.com/problems/validate-binary-search-tree/



#### Solution-dfs, did@1.13@4.10，下面那个比较好

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        self.res = True
        self.dfs(root)
        return self.res
        
    # return the smallest and the largest value in the subtree
    # one node's value must be larger than the max in left subtree and
    # must be smaller than the min in the right subtree
    
    def dfs(self, node):
        if not node.left and not node.right:
            return node.val, node.val
        if not self.res:
            return 0,0
        val1 = node.val
        val2 = node.val
        if node.left:
            left1, left2 = self.dfs(node.left) # left1 is the small one
            val1 = left1
            
            if not node.val>left2:
                self.res = False
        if node.right:
            right1, right2 = self.dfs(node.right)
            val2 = right2
            if not node.val<right1:
                self.res = False
            
        return val1, val2
      
      
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        minVal, maxVal = self.dfs(root)
        if minVal==None or maxVal==None:
            return False
        else:
            return True
        
        
        
    def dfs2(self, node):
        
        minVal = maxVal = node.val
        if node.left:
            leftMin, leftMax = self.dfs(node.left)
            if leftMin==None or leftMax==None or leftMax>=node.val:
                return None, None
            minVal = leftMin
        if node.right:
            rightMin, rightMax = self.dfs(node.right)
            if rightMin==None or rightMax==None or rightMin<=node.val:
                return None, None
            maxVal = rightMax
            
        return minVal, maxVal
            
        
        
```





### 235. Lowest Common Ancestor of a Binary Search Tree

https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/

#### Solution-did@1.13

利用好bst的特性

> 1. Left subtree of a node N contains nodes whose values are lesser than or equal to node N's value.
> 2. Right subtree of a node N contains nodes whose values are greater than node N's value.
> 3. Both left and right subtrees are also BSTs.

Ref: https://leetcode.com/articles/lowest-common-ancestor-of-a-binary-search-tree/



### 236. Lowest Common Ancestor of a Binary Tree-$$

https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

* Solution-dfs-1-**worth thinking and doing**@1.13

Ref: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/discuss/158060/Python-DFS-tm

@21.3.3, by myself

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return False
        # Find nothing: return False
        # Find one: return True
        # Find both: return the node
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # First, we would identify if we find it
        if left and left!=True:
            return left
        if right and right!=True:
            return right
        
        # see if the current one can be the result
        if left==True and right==True:
            return root
        if left and (root==p or root==q):
            return root
        if right and (root==p or root==q):
            return root
        
        return left or right or root==p or root==q
```



* Solution-recursive2, 感觉这个比较秒

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q ) {
        return root;
    } 
    TreeNode leftCommonAncestor =  lowestCommonAncestor(root.left, p, q); 
    TreeNode rightCommonAncestor =  lowestCommonAncestor(root.right, p, q); 
    //在左子树中没有找到，那一定在右子树中
    if(leftCommonAncestor == null){
        return rightCommonAncestor;
    }
    //在右子树中没有找到，那一定在左子树中
    if(rightCommonAncestor == null){
        return leftCommonAncestor;
    }
    //不在左子树，也不在右子树，那说明是根节点
    return root;
}

```



* Solution-inorder traversal

Ref: https://leetcode.wang/leetcode-236-Lowest-Common-Ancestor-of-a-Binary-Tree.html

@20.8.19, 感觉很奇怪。。。





### 108. Convert Sorted Array to Binary Search Tree

https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/

#### Solution-dfs, recursive, need speed up

我觉得这个比较好，不要用额外空间@1.13

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        return self.helper(nums, 0, len(nums)-1)        
        
    def helper(self, nums, start, end):
        if start>end:
            return None
        mid = (end+start)//2
        cur = TreeNode(nums[mid])
        
        cur.left = self.helper(nums, start, mid-1)
        cur.right = self.helper(nums, mid+1, end)
        return cur
```



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
> 所以python有溢出的问题吗？没有，python自己帮你转换了，所以python慢



#### Solution-dfs, stack, worth doing

Ref: https://leetcode.wang/leetcode-108-Convert-Sorted-Array-to-Binary-Search-Tree.html



### 109. Convert Sorted List to Binary Search Tree-$

https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/

#### Solution-先把链表转为list

#### Solution-快慢指针

Ref: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/discuss/35474/Python-recursive-solution-with-detailed-comments-(operate-linked-list-directly).

#### Solution-厉害！！**worth doing and thinking**

@1.13, 4.2, 8.20还是想不到, 其实是inorder traversal的思想

Ref: https://leetcode.wang/leetcode-109-Convert-Sorted-List-to-Binary-Search-Tree.html





### 173. Binary Search Tree Iterator

https://leetcode.com/problems/binary-search-tree-iterator/description/

#### Solution-worth其实就是94题, inorder traversal

Ref: https://leetcode.wang/leetcode-173-Binary-Search-Tree-Iterator.html

刚开始弄错是因为初始化的时候我就把root加了进去

```python

class BSTIterator:

    def __init__(self, root: TreeNode):
        self.root = root
        self.stack = []

    def next(self) -> int:
        """
        @return the next smallest number
        """
        while self.root:
            self.stack.append(self.root)
            self.root = self.root.left
        top = self.stack.pop()
        self.root = top.right
        return top.val
        

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return self.root or self.stack
```







### 230. Kth Smallest Element in a BST-$followup

https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/

Ref: https://www.cnblogs.com/grandyang/p/4620012.html

#### Solution-**worth doing and thinking** 其实就是94题,  inorder traversal, iterative, 

easy@1.13

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



#### Solution-inorder traversal, recursive

#### Solution-Follow up:worth@1.13, 8.20

What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

B+tree的思想

> combine an indexing structure (we could keep BST here) with a double linked list.

Ref: https://leetcode.com/articles/kth-smallest-element-in-a-bst/





### 297. Serialize and Deserialize Binary Tree-$$

https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/

我觉得这个本质上就是preorder<->treenode, 跟前面一个题类似, **蛮重要的**

#### Solution-preorder-1-**worth doing and thinking**, need speed up

别人快的区别在deserialize时别人用了iter模块，放在下面了

Ref: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/discuss/74259/Recursive-preorder-Python-and-C%2B%2B-O(n)

尝试用bit 优化https://www.youtube.com/watch?v=JL4OjKV_pGE

By huahua, 其他人快是用的iterative

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def doit(node):
            if node:
                nodel.append(str(node.val))
                doit(node.left)
                doit(node.right)
            else:
                nodel.append("#")
                
        nodel = []
        doit(root)
        return " ".join(nodel)
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if data == '#':
            return []
        
        def doit(value):
            if not nodeL:return
            if value!="#":    
                node = TreeNode(int(value))
                node.left = doit(nodeL.pop(0))
                node.right = doit(nodeL.pop(0))
                return node
            
            
        nodeL = data.split(" ")
        
        return doit(int(nodeL.pop(0)))
```

#### Solution-preorder-2

别人的快的, 下面我做了一个，但本质上就是手动实现了一个iterator

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def preorder(node):
            if node == None:
                vals.append("#")
            else:
                vals.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
        
        vals = [];
        preorder(root)
        return " ".join(vals)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def build():
            val = next(vals)
            if (val == "#"):
                return None
            else:
                node = TreeNode(int(val))
                node.left = build()
                node.right = build()
                return node
            
        vals = iter(data.split())
        return build()
```

#### Solution-preorder-3

did@21.3.3

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        string = ""
        if not root:
            return string
        
        stack = [root]
        while stack:
            root = stack.pop()
            if root is not None:
                string+=str(root.val)+","
                stack.append(root.right)
                stack.append(root.left)
            else:
                string+="#,"
            
        return string

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        if len(data)==0:
            return None
        return self.helper([0], data)
       
    def helper(self, position, data):
        if data[position[0]] == "#":
            position[0]+=2
            return None
        comma = data.find(",", position[0])
        node = TreeNode(int(data[position[0]: comma]))
        position[0] = comma+1
        node.left = self.helper(position, data)
        node.right = self.helper(position, data)
        return node
```



#### Solution-postorder

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        
        result = []
        stack = [root]
        while stack:
            root = stack.pop()
            if root is not None:
                result.append(str(root.val))
                stack.append(root.left)
                stack.append(root.right)
            else:
                result.append("#")
            
        result.reverse()    
        return " ".join(result)
                
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if len(data)==0:
            return None
        
        values = data.split()
        stack = []
        for value in values:
            if value=="#":
                stack.append(None)
            else:
                node = TreeNode(int(value))
                node.right = stack.pop()
                node.left = stack.pop()
                stack.append(node)
                
        return stack[0]   

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
```





#### Solution-level order/bfs-**worth doing and thinking**，did@1.13

did@20.8.22, 我觉得我这个好，不要额外弄一个列表，但是root的处理emmm

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        queue = collections.deque()
        queue.append(root)
        output = ""
        
        while queue:
            node = queue.popleft()
            if node:
                output+=str(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                output+='*'
            output+=','
            
            
        return output
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if data=='':
            return None
        firstComma = data.find(",")
        queue = collections.deque()
        root = TreeNode(int(data[:firstComma]))
        queue.append(root)
        values = []
        value = ""
        
        for char in data[firstComma+1:]:
            
            if char==',':
                values.append(value)
                value=""
                
                if len(values)==2:
                    node = queue.popleft()
                    if values[0]!="*":
                        node.left = TreeNode(int(values[0]))
                        queue.append(node.left)
                    if values[1]!="*":
                        node.right = TreeNode(int(values[1]))
                        queue.append(node.right)
                    values.clear()
                            
            else:
                value+=char
                
        return root
                
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```





### 285. Inorder Successor in BST-$

https://leetcode.com/problems/inorder-successor-in-bst/

#### Solution-iterative-worth doing and thinking，利用好bst的特性，但还是可以尝试常规的办法

@20.8.20, 但做复杂了，可以先看看，仔细想想step2和1的关系

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        #step 1
        parent = None # we only record its parent if it is the left child
        while root.val!=p.val:
            if root.val<p.val:
                root = root.right
            else:
                parent = root
                root = root.left
                
        # step 2        
        if root.right:
            root = root.right
            stack = []
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            return root
                
            
        elif parent:
            return parent
        else:
            return None
```





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

常规

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        
        stack = []
        found = False
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
                
            root = stack.pop()
            if found:
                return root
            if root.val==p.val:
                found=True
            root = root.right
        return None
```



### 270. Closest Binary Search Tree Value

https://leetcode.com/problems/closest-binary-search-tree-value/description/

@20.8.22 you should make us of the properties of BST

#### Solution-inorder, iterative-**worth doing and thinking**

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



#### Solution-preorder, iterative, fast



### 272. Closest Binary Search Tree Value II-$

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

* Solution-一边inorder，一边更新, 40ms, **worth doing and thinking**, did@4.13

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

Ref: https://leetcode.com/problems/closest-binary-search-tree-value-ii/discuss/70534/O(k-%2B-logn)-Python-Solution

先得到了一个包含target的区间（sucessorStack, PredecessorStack), 建造函数来获取next successor和predecessor（从和target最近的往远处找）



### 99. Recover Binary Search Tree

https://leetcode.com/problems/recover-binary-search-tree/

#### solution- inorder

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

did@21.6.7

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        last_node = TreeNode(val=float("-inf"))
        small = large = None
        stack = []
        
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            if last_node.val>=root.val:
                if large==None:
                    large, small = last_node,root
                else:
                    small = root
                    break
            last_node = root
            root = root.right
        
        small.val, large.val = large.val, small.val
        
```



#### Solution-morris!!!



### 116. Populating Next Right Pointers in Each Node-$

https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

#### solution-dfs-recursive-**worth doing and thinking**, 这个人的写法感觉抓到了关键-$$

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

by myself@1.21, 8.22

dfs向下伸时，左儿子指到右儿子，看看爸爸有没有右兄弟，如果有，右儿子指到爸爸兄弟的左儿子

```python
class Solution:
    def connect(self, node: 'Node') -> 'Node':
        if node and node.left:
            node.left.next = node.right
            if node.next:
                node.right.next = node.next.left
            self.connect(node.left)
            self.connect(node.right)
            
        return node
```



#### Solution - dfs - stack

#### Solution - bfs



### 117. Populating Next Right Pointers in Each Node II-$$

https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/

#### Solution-bfs/level order, done

就用普通的level order还是简单的，但这样空间复杂度是O(N)

另一种, 看下面的更好理解：https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/discuss/37824/AC-Python-O(1)-space-solution-12-lines-and-easy-to-understand

```python
def connect(self, node):
    tail = dummy = TreeLinkNode(0)
    while node:
        tail.next = node.left#dummy.next = node.left
        if tail.next:#如果有左崽子
            tail = tail.next#tail变成左崽子
        tail.next = node.right#如果没有左崽子，dummy.next = node.right
        if tail.next:#如果有右崽子
            tail = tail.next#tail变成右崽子
        node = node.next#在本层移动
        if not node:
            tail = dummy
            node = dummy.next#崽子层最左边一个
```



https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/discuss/37811/Simple-solution-using-constant-space

这个人写的比较好理解，但其实tempChild不需要每次都重建

@20.8.23，就是如果我当层有next，我就帮下面的继续连

```python
public class Solution {
    public void connect(TreeLinkNode root) {
        
        while(root != null){
            TreeLinkNode tempChild = new TreeLinkNode(0);
            TreeLinkNode currentChild = tempChild;
            while(root!=null){
                if(root.left != null) { currentChild.next = root.left; currentChild = currentChild.next;}
                if(root.right != null) { currentChild.next = root.right; currentChild = currentChild.next;}
                root = root.next;
            }
            root = tempChild.next;
        }
    }
}
```



#### Solution-recursive-看看别人的







### 314. Binary Tree Vertical Order Traversal

想到思路实现很简单@1.21

https://leetcode.com/problems/binary-tree-vertical-order-traversal/description/

#### Solution-hash table, 

**worth thinking and doing**, 很巧妙

https://leetcode.com/problems/binary-tree-vertical-order-traversal/discuss/76424/Python-solution



### 95. Unique Binary Search Trees II-$$

https://leetcode.com/problems/unique-binary-search-trees-ii/

#### Solution-recursive, **worth thinking and doing**-did@4.17

Ref: https://leetcode.wang/leetCode-95-Unique-Binary-Search-TreesII.html

但可以用memo加快

```python
# Definition for a binary tree node.
# class TfreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:
            return []
        nodes = [1+i for i in range(n)]
        dic = {}
        return self.dfs(1, n, dic)
        
        
    def dfs(self, start, end, dic):
        if start>end:
            return [None]
        if start==end:
            return [TreeNode(start)]
        if (start, end) in dic:
            return dic[(start, end)]
        
        
        validNodes = []
        for i in range(start, end+1):
            
            leftNodes = self.dfs(start, i-1, dic)
            rightNodes = self.dfs(i+1, end, dic)
            for left in leftNodes:
                for right in rightNodes:
                    node = TreeNode(i)
                    node.left = left
                    node.right = right
                    validNodes.append(node)
        dic[(start, end)] = validNodes
            
        return validNodes
```



#### Solution-dynamic programming-太强了-**worth doing and thinking**-$$

Ref: https://leetcode.com/problems/unique-binary-search-trees-ii/discuss/31493/Java-Solution-with-DP

> **result[i]** stores the result until length **i**. For the result for length i+1, select the root node j from 0 to i, combine the result from left side and right side. Note for the right side we have to clone the nodes as the value will be offsetted by **j**.

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





### 96. Unique Binary Search Trees-$

https://leetcode.com/problems/unique-binary-search-trees/description/

* solution-**Catalan number **-$

Ref: https://leetcode.wang/leetCode-96-Unique-Binary-Search-Trees.html

> 令h ( 0 ) = 1，catalan 数满足递推式：
>
> **h ( n ) = h ( 0 ) \* h ( n - 1 ) + h ( 1 ) \* h ( n - 2 ) + ... + h ( n - 1 ) \* h ( 0 ) ( n >=1 )**
>
> 例如：h ( 2 ) = h ( 0 ) * h ( 1 ) + h ( 1 ) * h ( 0 ) = 1 * 1 + 1 * 1 = 2
>
> h ( 3 ) = h ( 0 ) * h ( 2 ) + h ( 1 ) * h ( 1 ) + h ( 2 ) * h ( 0 ) = 1 * 2 + 1 * 1 + 2 * 1 = 5

* Solutions-延续95题的做法



