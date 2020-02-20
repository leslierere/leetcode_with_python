### 359. Logger Rate Limiter

https://leetcode.com/problems/logger-rate-limiter/

#### Solution-dictionary

#### Solution-queue, set

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
class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = collections.deque(maxlen=size)
        

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        queue = self.queue
        queue.append(val)
        return float(sum(queue))/len(queue)
```





### 362. Design Hit Counter

https://leetcode.com/problems/design-hit-counter/description/

#### Solution-deque

#### **Follow up:-$**

What if the number of hits per second could be very large? Does your design scale?