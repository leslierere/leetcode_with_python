### 	155. Min Stack-$

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





### 150. Evaluate Reverse Polish Notation-$

https://leetcode.com/problems/evaluate-reverse-polish-notation/description/

#### Solution-stack

主要是要注意减法除法的顺序，和除法towards 0的要求

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        operators = set(["+", "-", "*", "/"])
        
        for i in tokens:
            if i not in operators:
                stack.append(int(i))
            else:
                second = stack.pop()
                first = stack.pop()
                if i=="*":
                    stack.append(first* second)
                elif i=="+":
                    stack.append(first+ second)
                elif i=="-":
                    stack.append(first- second)
                else:
                    res = abs(first)//abs(second)
                    if first*second<0:
                        res = -res
                    stack.append(res)
        return stack[-1]
```





### 71. Simplify Path

https://leetcode.com/problems/simplify-path/description/

#### Solution-easy

Using two pointer is slow, use split would be alright



### 388. Longest Absolute File Path-$

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

depth的问题inspired by stephon（我一开始用split来做，但不知为啥有问题，下次再试试），然后code三部分顺序很重要

```python
class Solution:
    def lengthLongestPath(self, string: str) -> int:
        lines = string.split('\n')
        stack = [(-1, -1)]
        res = 0
        
        for line in lines:
            subdir = line.lstrip('\t')
            depth = len(line)-len(subdir)
            # part1
            while stack and depth<=stack[-1][1]:
                stack.pop()
            # part2
            stack.append((len(subdir)+stack[-1][0]+1, depth))
            # part3
            if '.' in subdir:
                res = max(res, stack[-1][0])
            
        # if subdir[-1].find('.')!=-1:
        #     res = max(res, stack[-1][0])
        return res
                    
        
```









### 394. Decode String-$,did@4.15

https://leetcode.com/problems/decode-string/

#### Solution-stack-worth

Ref: https://leetcode.com/problems/decode-string/discuss/87662/Python-solution-using-stack

```python
class Solution(object):
    def decodeString(self, s):
        stack = []; curNum = 0; curString = ''
        for c in s:
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                curNum = curNum*10 + int(c)#对数字这个处理太好了
            else:
                curString += c
        return curString
```



#### Solution-recursive-worth

Ref: https://leetcode.com/problems/decode-string/discuss/87544/Clean-C%2B%2B-Recursive-Solution-with-Explanation

```python
class Solution:
    def decodeString(self, s: str) -> str:
        return self.helper(0, s)
        
    def helper(self, start, s):
        temp = ""
        count = 0
        i = start
        while i < len(s):
            cur = s[i]
            if cur.isdigit():
                count = count*10 + int(s[i])
            elif cur == "[":
                word, nextI = self.helper(i+1, s)
                temp += count*word
                i = nextI
                count = 0
                continue
            elif cur == "]":
                return temp, i+1
            else:
                temp += cur
            i+=1
        return temp
```





### 224. Basic Calculator-$

https://leetcode.com/problems/basic-calculator/description/

#### Solution-stack

这里对+-符号的处理特别好，思想还是跟上一题stack一样

@6.7这次比较好, 真延续上次做法

```python
class Solution:
    def calculate(self, s: str) -> int:
        sign = 1
        num = 0
        
        stack = [0]
        
        for char in s:
            if char.isdigit():
                num = num*10 + int(char)
            elif char=='+':
                stack[-1]+=sign*num
                num = 0
                sign = 1
            elif char=='-':
                stack[-1]+=sign*num
                num = 0
                sign = -1
            elif char=='(':
                stack.append(sign)
                stack.append(0)
                sign = 1
            elif char==')':
                stack[-1]+=sign*num
                subSum = stack.pop()*stack.pop()
                stack[-1]+=subSum
                num = 0
                
        return stack[-1]+num*sign
```



```python
class Solution:
    def calculate(self, s: str) -> int:
        
#本题用stack来做。设三个变量: sums, num, sign。在创建一个stack。总共会有4种情况。 1.是数字，2.字符("+" or "-"), 3."(" 4.")"

#做法是每当遇见 "("的时候把sum清零，然后把之前加的数字还有括号之前的哪一个字符放到stack里面。而当遇到 ")"的时候把stack里面的数字和字符在pop出来

        sums, num, sign = 0, 0, 1
        stack = []
        for i in s:
            if i.isdigit():
                num = 10 * num + int(i)
            elif i == "+" or i == "-":
                sums += num * sign
                num = 0
                sign = 1 if i =="+" else -1
            elif i == "(":
                stack.append(sums)
                stack.append(sign)
                sums = 0 
                sign = 1 
            elif i == ")":
                sums += num * sign
                num = 0
                sums *= stack.pop()
                sums += stack.pop()
        sums += num * sign
        return sums 
```



### 227. Basic Calculator II

https://leetcode.com/problems/basic-calculator-ii/description/

#### Solution-stack

Ref: https://leetcode.com/problems/basic-calculator-ii/discuss/63003/Share-my-java-solution

我做复杂了，比如+-其实可以通过往stack加数字时一起处理

did@6.7, 感觉这个思路比较清晰

```python
class Solution:
    def calculate(self, s: str) -> int:
        s+='+'
        # sign = 1
        stack = []
        num = 0
        last_op = '+'
        
        # 碰到+-就直接处理
        # 碰到*/就继续算
        for char in s:
            if char.isdigit():
                num = num*10 + int(char)
            elif char in "+-*/":
                
                if last_op =='*':
                    stack[-1]*=num
                elif last_op =='/':
                    stack[-1]= int(stack[-1]/num)
                elif last_op == "+":
                    stack.append(num)
                else:
                    stack.append(-num)
                last_op = char
                num = 0
                
        return sum(stack)
```



```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        sign = '+'
        
        for idx, curr in enumerate(s):
            if curr.isdigit():
                num = num * 10 + int(curr)  # to handle integers > 9 e.g. 13

            if curr in '+-*/' or idx == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                if sign == '-':
                    stack.append(-num)
                if sign == '*':
                    stack.append(stack.pop() * num)
                if sign == '/':
                    stack.append(int(stack.pop() / num))
                sign = curr
                num = 0
        return sum(stack)
```





### 385. Mini Parser-$  

https://leetcode.com/problems/mini-parser/description/

#### Solution-stack

@6.7感觉这一次比较好！

```python
class Solution:
    def deserialize(self, s: str) -> NestedInteger:
        num = 0
        stack = []
        sign = 1
        
        for i, char in enumerate(s):
            if char.isnumeric():
                num = num*10 + int(char)
            elif char=='[':
                stack.append(NestedInteger())
            elif char in ',]':
                if s[i-1]==']':
                    subInteger = stack.pop()
                    stack[-1].add(subInteger)
                elif s[i-1]!='[':
                    stack[-1].add(NestedInteger(value=sign*num))
                num = 0
                sign = 1
            else:
                sign = -1
        if stack:
            return stack[-1]
        else:
            return NestedInteger(value=sign*num)
```



Ref: https://leetcode.com/problems/mini-parser/discuss/86066/An-Java-Iterative-Solution

```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to the list.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution:
    def deserialize(self, s: str) -> NestedInteger:
        digits = 0
        stack = []
        l = 0
        
        if s[0]!='[':
            res = NestedInteger()
            res.setInteger(int(s))
            return res
        
        for i in range(len(s)):
            cur = s[i]
            if cur=='[':
                stack.append(NestedInteger())
                l = i+1
            elif cur == ',':
                if s[i-1]!=']':
                    stack[-1].add(NestedInteger(int(s[l:i])))
                l = i+1
            elif cur ==']':
                num = s[l:i]
                if num:
                    stack[-1].add(NestedInteger(int(num)))
                if stack:
                    top = stack.pop()
                    if stack:
                        stack[-1].add(top)
                l = i+1
                
        return top
```



### 84. Largest Rectangle in Histogram-$

Ref: https://leetcode.com/problems/largest-rectangle-in-histogram/

可以根据[这个思路](https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/28917/AC-Python-clean-solution-using-stack-76ms)修改下面代码

上面思路：

```python
def largestRectangleArea(self, height):
    height.append(0)#这个太聪明了，解决了stack会有剩余的问题
    stack = [-1]
    ans = 0
    for i in xrange(len(height)):
        while height[i] < height[stack[-1]]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1
            ans = max(ans, h * w)
        stack.append(i)
    height.pop()#复原也要记得
    return ans
```



```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        area = 0
        i = 0
        while i< len(heights):
            if len(stack)==1 or heights[i]>=heights[stack[-1]]:
                stack.append(i)
                i+=1
            else:
                top = stack.pop()
                area = max(area, (i-stack[-1]-1)*heights[top])
        # stack = [-1,1,4,5]
        
        while len(stack)>1:
            top = stack.pop()
            area = max(area, (i-stack[-1]-1)*heights[top])
        
        return area
```





### 215. Kth Largest Element in an Array

https://leetcode.com/problems/kth-largest-element-in-an-array/description/

#### Solution-quicksort

Since we used a list to store the items that equal to pivot, the time complexity is O(N)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        k = len(nums)-k+1 # turn it into the kth number in an ascending sorted array
      
        while nums:
            l1 = []
            l2 = []
            pivot = []
            midValue = nums[(len(nums))//2]
            for cur in nums:
                if cur<midValue:
                    l1.append(cur)
                elif cur==midValue:
                    pivot.append(cur)
                else:
                    l2.append(cur)
            if k<=len(l1):
                nums = l1
            elif k<=len(l1)+len(pivot):
                return pivot[0]
            else:
                nums = l2
                k = k-len(l1)-len(pivot)
```



#### Solution-priority queue-worth doing

O(N) to transfer to unsorted array to heap queue, and to remove kth time, each time takes O(logN)

so total time complexity should be O(klogN)



### 347. Top K Frequent Elements

Ref: https://leetcode.com/problems/top-k-frequent-elements/

#### Solution-bucket sort

Heap or Counter+sort





### 218. The Skyline Problem-$$$

https://leetcode.com/problems/the-skyline-problem/description/

#### Solution-so worth

Ref: https://leetcode.com/problems/the-skyline-problem/discuss/61261/10-line-Python-solution-104-ms

https://www.youtube.com/watch?v=8Kd-Tn_Rz7s

关于思路的话，因为我们需要的总是最高大楼，所以比较自然的思路就是priority queue 

```python
from heapq import heappush, heappop

class Solution(object):
    def getSkyline(self, buildings):
        # 不难发现这些关键点的特征是：竖直线上轮廓升高或者降低的终点
        # 所以核心思路是：从左至右遍历建筑物，记录当前的最高轮廓，如果产生变化则记录一个关键点
        
        # 首先记录构造一个建筑物的两种关键事件
        # 第一种是轮廓升高事件(L, -H)、第二种是轮廓降低事件(R, 0)
        # 轮廓升高事件(L, -H, R)中的R用于后面的最小堆
        events = [(L, -H, R) for L, R, H in buildings]
        events += list({(R, 0, 0) for _, R, _ in buildings})

        # 先根据L从小到大排序、再根据H从大到小排序(记录为-H的原因)
        # 这是因为我们要维护一个堆保存当前最高的轮廓
        events.sort()

        # 保存返回结果
        res = [[0, 0]]
        
        # 最小堆，保存当前最高的轮廓(-H, R)，用-H转换为最大堆，R的作用是记录该轮廓的有效长度
        live = [(0, float("inf"))]

        # 从左至右遍历关键事件
        for L, negH, R in events:
            
            # 如果是轮廓升高事件，记录到最小堆中
            if negH: heappush(live, (negH, R))
            
            # 获取当前最高轮廓
            # 根据当前遍历的位置L，判断最高轮廓是否有效
            # 如果无效则剔除，让次高的轮廓浮到堆顶，继续判断
            while live[0][1] <= L: 
                heappop(live)
            
            # 如果当前的最高轮廓发生了变化，则记录一个关键点
            if res[-1][1] != -live[0][0]:
                res += [ [L, -live[0][0]] ]
        return res[1:]
```





### 332. Reconstruct Itinerary-$$$

https://leetcode.com/problems/reconstruct-itinerary/description/

#### Solution-dfs-worth

Ref: https://leetcode.com/problems/reconstruct-itinerary/discuss/78768/Short-Ruby-Python-Java-C++/83576

https://www.cnblogs.com/grandyang/p/5183210.html

Eulerian Path: [Eulerian Path ](http://en.wikipedia.org/wiki/Eulerian_path)is a path in graph that visits every edge exactly once. Eulerian Circuit is an Eulerian Path which starts and ends on the same vertex.

@3.28其实我觉得和topological sort的dfs很像





### 341. Flatten Nested List Iterator- $$$

https://leetcode.com/problems/flatten-nested-list-iterator/

#### Solution-stack-worth

@6.8这个思路比较对

Ref: https://leetcode.com/problems/flatten-nested-list-iterator/discuss/80146/Real-iterator-in-Python-Java-C%2B%2B

> In my opinion an iterator shouldn't copy the entire data (which some solutions have done) but just iterate over the original data structure.