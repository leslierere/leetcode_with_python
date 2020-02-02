### 155. Min Stack

https://leetcode.com/problems/min-stack/description/

我一开始想着要把removeMin实现，于是就做复杂了。。。所以好好审题！

#### Solution1-two stacks

Ref: https://leetcode.wang/leetcode-155-Min-Stack.html



#### Solution2-linekd list

Ref: https://leetcode.com/problems/min-stack/discuss/49217/6ms-Java-Solution-using-Linked-List.-Clean-self-explanatory-and-efficient.

> Ref: https://leetcode.wang/leetcode-155-Min-Stack.html 
>
> 直接用一个链表即可实现栈的基本功能，那么最小值该怎么得到呢？我们可以在 `Node` 节点中增加一个 `min` 字段，这样的话每次加入一个节点的时候，我们同时只要确定它的 `min` 值即可。



#### Solution3-list of tuples

其实和solution2是一样的想法

Ref: https://leetcode.com/problems/min-stack/discuss/49022/My-Python-solution





### 232. Implement Queue using Stacks

https://leetcode.com/problems/implement-queue-using-stacks/description/

#### Solution

用python写没啥意义。。。



### 225. Implement Stack using Queues

https://leetcode.com/problems/implement-stack-using-queues/description/

#### Solution

用python写没啥意义。。。





### 150. Evaluate Reverse Polish Notation

https://leetcode.com/problems/evaluate-reverse-polish-notation/description/

#### Solution-stack



### 71. Simplify Path

https://leetcode.com/problems/simplify-path/description/

#### Solution-easy

Using two pointer is slow, use split would be alright



### 388. Longest Absolute File Path

https://leetcode.com/problems/longest-absolute-file-path/description/

```python
In [24]: len("\t")                                                                   
Out[24]: 1

In [25]: len("\t\t")                                                                 
Out[25]: 2
```

#### Solution-worth

Ref: https://leetcode.com/problems/longest-absolute-file-path/discuss/86619/Simple-Python-solution

#### Solution-using stack

with split()