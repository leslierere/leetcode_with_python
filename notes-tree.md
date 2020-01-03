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





#### 145. Binary Tree Postorder Traversal

https://leetcode.com/problems/binary-tree-postorder-traversal/description/

* Solution-iterative

https://www.youtube.com/watch?v=A6iCX_5xiU4

<img src="/Users/leslieren/Library/Application Support/typora-user-images/image-20200103114811047.png" alt="image-20200103114811047" style="zoom:80%;" />

```python
# 上面的核心思想是先做一个rev_postorder(root),这样就和最后要的结果刚好相反，但我们可以通过特殊操作，使用deque来使得最后的结果不需要reverse
# my solution based on huahua's
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

