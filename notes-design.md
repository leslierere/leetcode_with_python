### 359. Logger Rate Limiter

https://leetcode.com/problems/logger-rate-limiter/

#### Solution-dictionary

#### Solution-queue, set

@ 3.30 我觉得这个好，保留最近的就可以了

```python
class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dic = {}
        
        

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        if not message in self.dic:
            self.dic[message] = timestamp
            return True
        else:
            
            if timestamp-self.dic[message]>=10:
                self.dic[message] = timestamp
                return True
            else:
                # self.dic[message] = timestamp
                return False
```



Ref: https://leetcode.com/articles/logger-rate-limiter/

```python
from collections import deque
class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.words = set()
        self.queue = deque()
        

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        while self.queue and timestamp-self.queue[0][0]>=10:
            i, word = self.queue.popleft()
            self.words.remove(word)
        
        if message not in self.words:
            self.queue.append((timestamp, message))
            self.words.add(message)
            return True
        else:
            return False
            
# Your Logger object will be instantiated and called as such:
# obj = Logger()
# param_1 = obj.shouldPrintMessage(timestamp,message)
```





### 346. Moving Average from Data Stream

https://leetcode.com/problems/moving-average-from-data-stream/description/

#### Solution-deque

可以给deque限制size，Ref: https://leetcode.com/problems/moving-average-from-data-stream/discuss/81495/4-line-Python-Solution-using-deque

```python
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.numbers = collections.deque(maxlen=size)
        self.agg = 0
        self.size = size
        

    def next(self, val: int) -> float:
        
        if len(self.numbers)<self.size:
            self.agg+=val
        else:
            self.agg+=(-self.numbers[0]+val)
        self.numbers.append(val)
        return self.agg/len(self.numbers)
        


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
```







### 362. Design Hit Counter

https://leetcode.com/problems/design-hit-counter/description/

#### Solution-deque

#### **Follow up:-$**

What if the number of hits per second could be very large? Does your design scale?

[https://leetcode.com/problems/design-hit-counter/discuss/83483/Super-easy-design-O(1)-hit()-O(s)-getHits()-no-fancy-data-structure-is-needed!](https://leetcode.com/problems/design-hit-counter/discuss/83483/Super-easy-design-O(1)-hit()-O(s)-getHits()-no-fancy-data-structure-is-needed!)



### 281. Zigzag Iterator

https://leetcode.com/problems/zigzag-iterator/

#### Solution-bfs-using iter

Ref: https://leetcode.com/problems/zigzag-iterator/discuss/71786/Python-O(1)-space-solutions

```python
class ZigzagIterator(object):

    def __init__(self, v1, v2):
        self.data = [(len(v), iter(v)) for v in (v1, v2) if v]

    def next(self):
        len, iter = self.data.pop(0)
        if len > 1:
            self.data.append((len-1, iter))
        return next(iter)

    def hasNext(self):
        return bool(self.data)
```





### 284. Peeking Iterator

https://leetcode.com/problems/peeking-iterator/

```python
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator:
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.structure = iterator
        self.peekValue = None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if not self.peekValue:
            self.peekValue=self.structure.next()
        return self.peekValue
        

    def next(self):
        """
        :rtype: int
        """
        if self.peekValue:
            value = self.peekValue
            self.peekValue=None
            return value
        else:
            return self.structure.next()
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.structure.hasNext() or self.peekValue!=None
        

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].
```



### 251. Flatten 2D Vector

https://leetcode.com/problems/flatten-2d-vector/description/s

#### Solution

```python
class Vector2D:

    def __init__(self, v: List[List[int]]):
        self.vec=v
        self.r=0
        self.c=0
        self.m=len(v)
        

    def next(self) -> int:
        if self.hasNext() is False:
            return None

        x=self.vec[self.r][self.c]
        self.c+=1
        return x
        

    def hasNext(self) -> bool:
        

        while self.r < self.m and self.c == len(self.vec[self.r]):
# self.c == len(self.vec[self.r])这个特别聪明，因为包含了sublist为null的情况
            self.r, self.c = self.r+1, 0
            
        if self.r==self.m:
            return False
        return True
```





### 288. Unique Word Abbreviation

https://leetcode.com/problems/unique-word-abbreviation/description/

#### Solution

题目不清楚，实现很简单，没啥做的必要。。。



### 170. Two Sum III - Data structure design

https://leetcode.com/problems/two-sum-iii-data-structure-design/description/

#### Solution

考虑tradeoff：https://leetcode.com/problems/two-sum-iii-data-structure-design/discuss/52005/Trade-off-in-this-problem-should-be-considered



### 348. Design Tic-Tac-Toe

https://leetcode.com/problems/design-tic-tac-toe/description/

#### Solution

Ref: https://leetcode.com/problems/design-tic-tac-toe/discuss/81898/Java-O(1)-solution-easy-to-understand





### 379. Design Phone Directory

https://leetcode.com/problems/design-phone-directory/description/

下次不用做了。。。





### 353. Design Snake Game-想一想对food的处理

https://leetcode.com/problems/design-snake-game/description/

#### Solution

这里对food的处理比较好

Ref: https://leetcode.com/problems/design-snake-game/discuss/82681/Straightforward-Python-solution-using-deque



### 146. LRU Cache

https://leetcode.com/problems/lru-cache/description/

#### Solution-queue, hashtable, 很慢

#### Solution-dic+doubly linked list

Ref: https://leetcode.com/problems/lru-cache/discuss/45926/Python-Dict-%2B-Double-LinkedList



### 355. Design Twitter

https://leetcode.com/problems/design-twitter/

#### Solution

Ref: https://leetcode.com/problems/design-twitter/discuss/82825/Java-OO-Design-with-most-efficient-function-getNewsFeed

in the getNewsFeed(), we always add the latest tweet among the tweets from diff users, we stop for at most 10 times.





### 303. Range Sum Query - Immutable

https://leetcode.com/problems/range-sum-query-immutable/description/

不用再做了



### 304. Range Sum Query 2D - Immutable

https://leetcode.com/problems/range-sum-query-2d-immutable/description/

想一下就行，https://leetcode.com/articles/range-sum-query-2d-immutable/ 这里面的appraoch3&4





### 307. Range Sum Query - Mutable

https://leetcode.com/problems/range-sum-query-mutable/description/

#### Solution-segment tree$

Ref: https://www.youtube.com/watch?v=rYBtViWXYeI

![image-20200224132016505](https://tva1.sinaimg.cn/large/0082zybpgy1gc83g632exj31c00u07wh.jpg)

#### Solution-array-worth!!!!

* why we need to build from back to front?

  we want to give 2 consecutive nodes one parent, divide by 2 can do this

* how can we certify the length of reserved spaces for parents?

  n/2^1 + (n/2)/2 + (n/4)/2 +.... n/2^i = 1 this can never exceeds n

* also, when we update, we can always just trace back to its parent and add the difference until our root, and we are done then.

* when sum range, consider sum [7, 10]

![image-20210304222855635](https://tva1.sinaimg.cn/large/008eGmZEgy1go8x28nlflj31400u0wq0.jpg)

![image-20210304222910892](https://tva1.sinaimg.cn/large/008eGmZEgy1go8x2dz8o1j30u01hmnhp.jpg)



### 308. Range Sum Query 2D - Mutable

https://leetcode.com/problems/range-sum-query-2d-mutable/description/

#### Solution

Ref: https://leetcode.com/problems/range-sum-query-2d-mutable/discuss/75852/15ms-easy-to-understand-java-solution

304 里面一种cache的方法