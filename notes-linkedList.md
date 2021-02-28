1.11

### 206. Reverse Linked List

https://leetcode.com/problems/reverse-linked-list/

#### Solution-iterative

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        node = None
        while head:
            temp= head.next#2
            head.next = node#1->null
            node = head#node = 1
            head = temp
            
        return node
```



#### Solution-recursive-worth

Ref: https://leetcode.wang/leetcode-206-Reverse-Linked-List.html

did@20.9.16



### 141. Linked List Cycle

https://leetcode.com/problems/linked-list-cycle/

#### Solution-O(n)

#### Solution-O(n-1)-worth-Floyd's Tortoise and Hare (Cycle Detection)



### 24. Swap Nodes in Pairs-worth

https://leetcode.com/problems/swap-nodes-in-pairs/description/

一定要画图！

#### Solution-iterative

#### Solution-recursive

Ref: https://leetcode.wang/leetCode-24-Swap-Nodes-in-Pairs.html



### 328. Odd Even Linked List

https://leetcode.com/problems/odd-even-linked-list/description/

#### Solution-two pointers



### 92. Reverse Linked List II

https://leetcode.com/problems/reverse-linked-list-ii/description/

#### Solution-iterative



### 237. Delete Node in a Linked List

https://leetcode.com/problems/delete-node-in-a-linked-list/description/



### 19. Remove Nth Node From End of List

https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/

#### Solution-使用dummy node，有思路就很好写了

#### Solution-不使用

Ref: https://leetcode.com/problems/remove-nth-node-from-end-of-list/discuss/8802/3-short-Python-solutions

> The standard solution, but without a dummy extra node. Instead, I simply handle the special case of removing the head right after the fast cursor got its head start.
>
> ```python
> class Solution:
>     def removeNthFromEnd(self, head, n):
>         fast = slow = head
>         for _ in range(n):
>             fast = fast.next
>         if not fast:
>             return head.next
>         while fast.next:
>             fast = fast.next
>             slow = slow.next
>         slow.next = slow.next.next
>         return head
> ```



### 83. Remove Duplicates from Sorted List-easy

https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/



### 203. Remove Linked List Elements-和83差不多

https://leetcode.com/problems/remove-linked-list-elements/



### 82. Remove Duplicates from Sorted List II-$

https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/

#### Solution-下次可以用个flag，感觉做麻烦了

```python
class Solution:
    # the first node has duplicate number with the second one
    # consecutive duplicates
    # missed one: after delete the last one duplicate, nothing after it, thus you haven't seleted the begin duplicate, teat case: [1,1]
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return None
        dum = ListNode(0)
        dum.next = head
        # value = None
        prev = dum
        
        while head.next:
            
            if head.next and head.val == head.next.val:
                head.next = head.next.next
                if not head.next:
                    prev.next = None
                    break
                if head.next and head.val != head.next.val:
                    prev.next = head.next
                    head = prev.next
                continue
            head = head.next
            prev = prev.next
            
        return dum.next
```





### 369. Plus One Linked List

https://leetcode.com/problems/plus-one-linked-list/description/

#### Solution-recursive

```python
class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        if self.dfs(head):
            newHead = ListNode(1)
            newHead.next = head
            return newHead
        else:
            return head
        
    def dfs(self, node):
        if node.next:
            if self.dfs(node.next):
                if node.val==9:
                    node.val = 0
                    return 1
                else:
                    node.val+=1
            
        else:
            
            if node.val==9:
                node.val = 0
                return 1
            else:
                node.val+=1
```



#### Solution-iterative, two pointers, 巧妙

https://leetcode.com/problems/plus-one-linked-list/discuss/84125/Iterative-Two-Pointers-with-dummy-node-Java-O(n)-time-O(1)-space

### 2. Add Two Numbers

https://leetcode.com/problems/add-two-numbers/description/

#### Solution-比我的简洁一点

```java
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode c1 = l1;
        ListNode c2 = l2;
        ListNode sentinel = new ListNode(0);
        ListNode d = sentinel;
        int sum = 0;
        while (c1 != null || c2 != null) {
            sum /= 10;
            if (c1 != null) {
                sum += c1.val;
                c1 = c1.next;
            }
            if (c2 != null) {
                sum += c2.val;
                c2 = c2.next;
            }
            d.next = new ListNode(sum % 10);
            d = d.next;
        }
        if (sum / 10 == 1)
            d.next = new ListNode(1);
        return sentinel.next;
    }
}
```





### 160. Intersection of Two Linked Lists

https://leetcode.com/problems/intersection-of-two-linked-lists/description/

#### Solution-worth thinking

Ref: [https://leetcode.com/problems/intersection-of-two-linked-lists/discuss/49785/Java-solution-without-knowing-the-difference-in-len!](https://leetcode.com/problems/intersection-of-two-linked-lists/discuss/49785/Java-solution-without-knowing-the-difference-in-len!)

> We can use two iterations to do that. In the first iteration, we will reset the pointer of one linkedlist to the head of another linkedlist after it reaches the tail node. In the second iteration, we will move two pointers until they points to the same node. Our operations in first iteration will help us counteract the difference. So if two linkedlist intersects, the meeting point in second iteration must be the intersection point. If the two linked lists have no intersection at all, then the meeting pointer in second iteration must be the tail node of both lists, which is null



### 21. Merge Two Sorted Lists-easy

https://leetcode.com/problems/merge-two-sorted-lists/description/

#### Solution

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # The lengths of two lists are different
        # which one is the beginning node
        dum = ListNode(0)
        cur = dum
        
        while l1 and l2:
            
            if l1.val<=l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
            
        if l1:
            cur.next = l1
        if l2:
            cur.next = l2
            
        return dum.next
```



### 234. Palindrome Linked List-$

https://leetcode.com/problems/palindrome-linked-list/

#### Solution

修改了之前的列表指向

Ref: https://leetcode.com/problems/palindrome-linked-list/discuss/64500/11-lines-12-with-restore-O(n)-time-O(1)-space

```python
def isPalindrome(self, head):
    rev = None
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:# when there are odd number of nodes
        slow = slow.next
    while rev and rev.val == slow.val:
        slow = slow.next
        rev = rev.next
    return not rev
```



### 143. Reorder List

https://leetcode.com/problems/reorder-list/description/

#### Solution-dfs, 写到一半

#### Solution

Ref: https://leetcode.wang/leetcode-143-Reorder-List.html

> 解法三:
>
> 1 -> 2 -> 3 -> 4 -> 5 -> 6
> 第一步，将链表平均分成两半
> 1 -> 2 -> 3
> 4 -> 5 -> 6
>
> 第二步，将第二个链表逆序
> 1 -> 2 -> 3
> 6 -> 5 -> 4
>
> 第三步，依次连接两个链表
> 1 -> 6 -> 2 -> 5 -> 3 -> 4



### 142. Linked List Cycle II

https://leetcode.com/problems/linked-list-cycle-ii/

#### Solution-hashmap, O(n)space

Ref: https://leetcode.wang/leetcode-142-Linked-List-CycleII.html



#### Solution-two pointers

Fast: 快的指针走的步数，每次走两步

Slow：慢指针走的步数，每次走一步

L1: 起点到cycle entry的距离

L2: cycle entry到尾部的距离

Fast = 2 Slow

相遇时，fast比slow多走一圈，所以有fast - slow = l2,  所以当前slow = L2

如图，可以算出阴影部分长度，亦即相遇点到尾部长度为L1 + L2 - slow = L1

此时加一个指针，让新指针和slow同时走，这两个指针相遇时则为entry

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaui749u2cj31i90kz7wh.jpg" alt="image-20200112155222132" style="zoom:23%;" />

Ref: https://leetcode.com/problems/linked-list-cycle-ii/discuss/44902/Sharing-my-Python-solution

```python
def detectCycle(self, head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    while head != slow:
        slow = slow.next
        head = head.next
    return head
```



### 148. Sort List-$

https://leetcode.com/problems/sort-list/description/

#### Solution-O(logN) space

#### Solution-O(NlogN) time, O(1) space- worth, bottom up merge sort, need speed up

Ref: http://zxi.mytechroad.com/blog/list/leetcode-148-sort-list/

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # if head is null or head.next is null, return head
        if not head or not head.next:
            return head
        
        length = 1
        cur = head
        while cur.next:
            length+=1
            cur = cur.next
            
        dum = ListNode(0)
        dum.next = head
        
        n = 1
        while n<length:
            cur = dum.next
            tail = dum
            
            while cur:
                l = cur
                r = self.split(l, n)
                cur = self.split(r, n)
                tail.next, tail = self.merge(l, r)
            
            n = n<<1
                
        return dum.next
        
    # Splits the list into two parts, first n element and the rest.
    # Returns the head of the rest.    
    def split(self, head, n):
        while n>1 and head:
            n-=1
            head = head.next
            
        if head:
            rest = head.next
            head.next = None
        else:
            rest = None
        return rest
    
    # merge 2 sorted lists, returns the head and tail of the merged list.
    def merge(self, l1, l2):
        dum = ListNode(0)
        cur = dum
        
        while l1 and l2:
            if l1.val<=l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
            
        if l1:
            cur.next = l1
            tail = l1
        else:
            cur.next = l2
            tail = l2
            
        while tail.next:
            tail = tail.next
            
        return dum.next, tail
```



### 25. Reverse Nodes in k-Group

https://leetcode.com/problems/reverse-nodes-in-k-group/description/

#### Solution-指针-思路简单

#### Solution-递归-相对巧妙

Ref: https://leetcode.wang/leetCode-25-Reverse-Nodes-in-k-Group.html



### 61. Rotate List

https://leetcode.com/problems/rotate-list/description/

#### Solution-two pointer

比较简单，懒得写了



### 86. Partition List

https://leetcode.com/problems/partition-list/description/

#### Solution1- two pointers， 写到一半

#### Solution2- 这个思路实现起来简单一些

Ref: https://leetcode.wang/leetCode-86-Partition-List.html

> 我们知道，快排中之所以用相对不好理解的双指针，就是为了减少空间复杂度，让我们想一下最直接的方法。new 两个数组，一个数组保存小于分区点的数，另一个数组保存大于等于分区点的数，然后把两个数组结合在一起就可以了。
>
> ```java
> 1 4 3 2 5 2  x = 3
> min = {1 2 2}
> max = {4 3 5}
> 接在一起
> ans = {1 2 2 4 3 5}
> ```
>
> 数组由于需要多浪费空间，而没有采取这种思路，但是链表就不一样了呀，它并不需要开辟新的空间，而只改变指针就可以了。
>
> ```java
> public ListNode partition(ListNode head, int x) { 
>     //小于分区点的链表
>     ListNode min_head = new ListNode(0);
>     ListNode min = min_head;
>     //大于等于分区点的链表
>     ListNode max_head = new ListNode(0);
>     ListNode max = max_head;
> 
>     //遍历整个链表
>     while (head != null) {  
>         if (head.val < x) {
>             min.next = head;
>             min = min.next;
>         } else { 
>             max.next = head;
>             max = max.next;
>         }
> 
>         head = head.next;
>     } 
>     max.next = null;  //这步不要忘记，不然链表就出现环了
>     //两个链表接起来
>     min.next = max_head.next;
> 
>     return min_head.next;
> }
> ```



### 23. Merge k Sorted Lists

https://leetcode.com/problems/merge-k-sorted-lists/description/

#### Solution-两两合并，k个链表，合并log(k)次-可做

#### Solution-priority queue-worth

可以考虑用queue模块的priority queue([一个参考](https://leetcode.com/problems/merge-k-sorted-lists/discuss/10511/10-line-python-solution-with-priority-queue))来实现或者heapq模块







### 147. Insertion Sort List

https://leetcode.com/problems/insertion-sort-list/description/

不难



### 小结

Ref: https://leetcode.com/problems/delete-node-in-a-linked-list/discuss/65454/Why-LeetCode-accepted-such-stupid-question/187862

> The whole point of asking any candidates a linked list problem is to test if the candidates think about edge cases, including:
>
> 1. Dereferencing Null Pointer, usually targeting tail pointer
> 2. When given Head is None
> 3. When there are duplications in the list





