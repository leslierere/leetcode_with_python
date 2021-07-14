### 384. Shuffle an Array

https://leetcode.com/problems/shuffle-an-array/

#### Solution

Ref: https://leetcode.com/problems/shuffle-an-array/discuss/85957/easy-python-solution-based-on-generating-random-index-and-swapping





### 398. Random Pick Index

https://leetcode.com/problems/random-pick-index/

#### Solution- reservoir sampling

> Ref: https://leetcode.com/problems/random-pick-index/discuss/88072/Simple-Reservoir-Sampling-solution/92945
>
> To those who don't understand why it works. Consider the example in the OJ
> **{1,2,3,3,3}** with **target 3,** you want to select 2,3,4 with a probability of 1/3 each.
>
> 
>
> 2 : It's probability of selection is 1 * (1/2) * (2/3) = 1/3
> 3 : It's probability of selection is (1/2) * (2/3) = 1/3
> 4 : It's probability of selection is just 1/3
>
> 
>
> So they are each randomly selected.

https://leetcode.com/problems/random-pick-index/discuss/88153/Python-reservoir-sampling-solution.

```python
def __init__(self, nums):
    self.nums = nums
    
def pick(self, target):
    res = None
    count = 0
    for i, x in enumerate(self.nums):
        if x == target:
            count += 1
            chance = random.randint(1, count)
            if chance == count:
                res = i
    return res
```





### 382. Linked List Random Node

https://leetcode.com/problems/linked-list-random-node/

#### Solution-reservoir sampling-need refine

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:

    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head
        

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        node = self.head
        res = node
        counts = 1
        while node:
            chance = random.randint(1, counts)
            if chance == counts:
                res = node
            node = node.next
            counts+=1
            
        return res.val
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()
```



#### Solution- count the no of nodes first



### 380. Insert Delete GetRandom O(1)-$

https://leetcode.com/problems/insert-delete-getrandom-o1/

> Ref: https://leetcode.com/articles/insert-delete-getrandom-o1/
>
> This is widely used in popular statistical algorithms like [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) and[Metropolis–Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm). 

#### Solution

```python
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.array = []
        self.dic = dict()
        

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dic:
            return False
        else:
            self.array.append(val)
            self.dic[val] = len(self.array)-1
            return True
        

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dic:
            index = self.dic[val]
            lastItem = self.array[-1]
            self.dic[lastItem] = index
            self.array[index]= lastItem # the thought of swapping the 2 values are great
            self.array.pop()
            self.dic.pop(val)
            return True
        else:
            return False
        
        

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        
        return random.choice(self.array)
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```





### 381. Insert Delete GetRandom O(1) - Duplicates allowed

https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/

#### Solution

```python
class RandomizedCollection:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.array = []
        self.dic = dict()
        

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        """
        self.array.append(val)
        if val in self.dic:
            self.dic[val].add(len(self.array)-1)
        else:
            self.dic[val] = set()
            self.dic[val].add(len(self.array)-1)
            return True
        

    def remove(self, val: int) -> bool:
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        """
        if val in self.dic:
            index = self.dic[val].pop()
            lastIndex = len(self.array)-1
            lastItem = self.array[lastIndex]
            self.array[index] = lastItem
            self.array.pop()
            self.dic[lastItem].add(index)
            self.dic[lastItem].remove(lastIndex)
            if not self.dic[val]:
                self.dic.pop(val)
            return True
        else:
            return False
        

    def getRandom(self) -> int:
        """
        Get a random element from the collection.
        """
        return random.choice(self.array)
        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```



### 138. Copy List with Random Pointer

https://leetcode.com/problems/copy-list-with-random-pointer/

#### Solution-no extra space

according to the hints

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        node = head
        while node:
            nextNode = node.next
            node.next = Node(node.val, next = nextNode)
            node = nextNode
            
        node = head
        while node:
            randomNode = node.random
            if randomNode: # as randomNode can be null
                node.next.random = randomNode.next
            node = node.next.next
            
        node = head
        copyHead = head.next
        while node:
            copyNode = node.next
            node.next = copyNode.next
            if node.next:
                copyNode.next = node.next.next
            node = node.next
            
        return copyHead
```



#### Solution-default dic, O(1N)

Ref: https://leetcode.com/problems/copy-list-with-random-pointer/submissions/