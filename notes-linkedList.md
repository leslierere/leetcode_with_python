1.11

### 206. Reverse Linked List

https://leetcode.com/problems/reverse-linked-list/

#### Solution-iterative

#### Solution-recursive-worth

Ref: https://leetcode.wang/leetcode-206-Reverse-Linked-List.html





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



### 82. Remove Duplicates from Sorted List II

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









### 小结

Ref: https://leetcode.com/problems/delete-node-in-a-linked-list/discuss/65454/Why-LeetCode-accepted-such-stupid-question/187862

> The whole point of asking any candidates a linked list problem is to test if the candidates think about edge cases, including:
>
> 1. Dereferencing Null Pointer, usually targeting tail pointer
> 2. When given Head is None
> 3. When there are duplications in the list