12.19

### 7.Reverse Integer

https://leetcode.com/problems/reverse-integer/

* Solution-bit manipulation-easy

```python
class Solution:
    def reverse(self, x: int) -> int:
        number = int(str(abs(x))[::-1])
        if x<0:
            number = -number
        if number>(1<<31)-1 or number< -(1<<31):
            return 0
        return number
```

### 165. Compare Version Numbers

https://leetcode.com/problems/compare-version-numbers/description/

* Solution-easy

```python
class Solution:#my solution
    def compareVersion(self, version1: str, version2: str) -> int:
        l1 = version1.split(".")
        l2 = version2.split(".")
        
        while len(l1)>0 and len(l2)>0:
            x1 = int(l1.pop(0))
            x2 = int(l2.pop(0))
            
            if x1>x2:
                return 1
            elif x1<x2:
                return -1
            
        while len(l1):
            if int(l1.pop(0))>0:
                return 1
        
        while len(l2):
            if int(l2.pop(0))>0:
                return -1
            
        return 0
```



### 66. Plus One

https://leetcode.com/problems/plus-one/description/

* Solution-easy

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        i = len(digits)-1
        
        if digits[-1]==9:
            while i>=0 and digits[i]==9:
                digits[i]=0
                i-=1
                
            if i == -1:
                return [1]+digits     
  
        digits[i]+=1
        return digits
```



### 8. String to Integer (atoi)

https://leetcode.com/problems/string-to-integer-atoi/description/

* Solution 忽略



### 258. Add Digits

https://leetcode.com/problems/add-digits/description/

* Solution-easy



### 67. Add Binary

https://leetcode.com/problems/add-binary/

* Solution-easy



12.21

### 43. Multiply Strings

https://leetcode.com/problems/multiply-strings/description/

* solution

Ref: https://leetcode.wang/leetCode-43-Multiply-Strings.html 解法2



### 29. Divide Two Integers

https://leetcode.com/problems/divide-two-integers/

* solution-二分法，binary search，还可以再考虑递归

Ref: https://leetcode.com/problems/divide-two-integers/discuss/13403/Clear-python-code

```python
class Solution:
    def divide(self, dividend,divisor):
        sig = (dividend < 0) == (divisor < 0)
        dividend, divisor, res = abs(dividend), abs(divisor), 0
        while dividend >= divisor:
            x = 0
            while dividend >= divisor << (x + 1):
                x += 1
            res += 1 << x
            dividend -= divisor << x
        return min(res if sig else -res, 2147483647)
```



### 365. Water and Jug Problem

https://leetcode.com/problems/water-and-jug-problem/

* Solution-math, gcd的求法

Ref: https://leetcode.com/problems/water-and-jug-problem/discuss/83715/Math-solution-Java-solution

```python
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        if x + y < z:
            return False
        if (x==0 or y==0):            
            return x == z or y==z
        
        return  z%self.getGCD(x, y)==0
        
    def getGCD(self, x, y):
        # 輾轉相除法：兩數相除，取餘數重複進行相除，直到餘數為0時，前一個除數即為最大公因數。
        while y != 0:
            x, y = y, x%y
        return x
```

<img src="/Users/leslieren/Library/Application Support/typora-user-images/image-20191222112600719.png" alt="image-20191222112600719" style="zoom:30%;" />



### 204. Count Primes-$

https://leetcode.com/problems/count-primes/description/

#### Solution- Sieve of Eratosthenes

prime number must be larger than 1, 用for循环会更快

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n<=2:
            return 0
        prime = [1]*n#we dont think about element at index 0
        prime[0],prime[1]=0, 0# False for not primes
        divisor = 2
        for divisor in range(2, int(n**0.5)+1):
            if prime[divisor]: # if divisor is prime
                #to replace the following
                # y = 2*divisor
                # while y<n:
                #     prime[y] = 0
                #     y+=divisor
                #use this
                for y in range(2*divisor, n, divisor):
                    prime[y]=0
            
        return prime.count(1)
```



### 1. Two Sum

https://leetcode.com/problems/two-sum/

* Solution-easy- think about how to solve the problem in one pass



### 15. 3Sum

https://leetcode.com/problems/3sum/

* solution-how to achieve O(N2)

Ref: https://leetcode.wang/leetCode-15-3Sum.html

注意边界的处理



### 18. 4Sum

https://leetcode.com/problems/4sum/

* Solution-比上面多加一层循环

更快速的, ref: https://leetcode.com/problems/4sum/discuss/8545/Python-140ms-beats-100-and-works-for-N-sum-(Ngreater2)

```python
def fourSum(self, nums, target):
    nums.sort()
    results = []
    self.findNsum(nums, target, 4, [], results)
    return results

def findNsum(self, nums, target, N, result, results):
    if len(nums) < N or N < 2: return

    # solve 2-sum
    if N == 2:
        l,r = 0,len(nums)-1
        while l < r:
            if nums[l] + nums[r] == target:
                results.append(result + [nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                while r > l and nums[r] == nums[r + 1]:
                    r -= 1
            elif nums[l] + nums[r] < target:
                l += 1
            else:
                r -= 1
    else:
        for i in range(0, len(nums)-N+1):   # careful about range
            if target < nums[i]*N or target > nums[-1]*N:  # take advantages of sorted list
                break
            if i == 0 or i > 0 and nums[i-1] != nums[i]:  # recursively reduce N
                self.findNsum(nums[i+1:], target-nums[i], N-1, result+[nums[i]], results)
    return
```



### 149. Max Points on a Line-$$$

https://leetcode.com/problems/max-points-on-a-line/

直线方程的表现形式:

https://zhuanlan.zhihu.com/p/26263309

**摘要**：在平面解析几何中，直线方程有多种形式，在解决不同的问题时，使用适当的方程形式可以使问题简化，本文将列举出这些方程即性质。

1. 一般式：![[公式]](https://www.zhihu.com/equation?tex=Ax%2BBy%2BC%3D0)
   一般式说明了平面直角坐标系上一个二元一次方程表示一条直线，这是一种一一对应的关系。这里，A、B不同时为0，下面在表达斜率和截距时，分母均不为0，下文不再特殊说明。从直线的一般式中可以知道以下信息：
   斜率：![[公式]](https://www.zhihu.com/equation?tex=k%3D-%5Cfrac%7BA%7D%7BB%7D)
   法向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Bn%7D%7D%3D%28A%2CB%29)
   方向向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Ba%7D%7D%3D%28B%2C-A%29)
   x轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7BC%7D%7BA%7D)，y轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7BC%7D%7BB%7D)
2. 点斜式：![[公式]](https://www.zhihu.com/equation?tex=y-y_0%3Dk%28x-x_0%29)
   点斜式是由一个定点![[公式]](https://www.zhihu.com/equation?tex=P%28x_0%2Cy_0%29)和斜率![[公式]](https://www.zhihu.com/equation?tex=k)确定的直线方程。
   斜率：![[公式]](https://www.zhihu.com/equation?tex=k)
   法向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Bn%7D%7D%3D%28k%2C-1%29)
   方向向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Ba%7D%7D%3D%281%2Ck%29)
   x轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7By_0%7D%7Bk%7D+%2B+x_0)，y轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=y_0-kx_0)
3. 斜截式：![[公式]](https://www.zhihu.com/equation?tex=y%3Dkx%2Bb)
   斜截式是由斜率k和y轴上的截距b确定的直线方程。
   斜率：![[公式]](https://www.zhihu.com/equation?tex=k)
   法向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Bn%7D%7D%3D%28k%2C-1%29)
   方向向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Ba%7D%7D%3D%281%2Ck%29)
   x轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7Bb%7D%7Bk%7D)，y轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=b)
4. 两点式：![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7By-y_1%7D%7By_2-y_1%7D+%3D+%5Cfrac%7Bx-x_1%7D%7Bx_2-x_1%7D)
   两点式是由已知的两个点![[公式]](https://www.zhihu.com/equation?tex=%28x_1%2Cy_1%29)、![[公式]](https://www.zhihu.com/equation?tex=%28x_2%2Cy_2%29)所确定的直线方程。
   斜率：![[公式]](https://www.zhihu.com/equation?tex=k+%3D+%5Cfrac%7By_2-y_1%7D%7Bx_2-x_1%7D)
   法向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Bn%7D%7D%3D%28y_2-y_1%2Cx_1-x_2%29)
   方向向量：![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Ctextbf%7Ba%7D%7D%3D%28x_2-x_1%2Cy_2-y_1%29)
   x轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bx_1y_2-x_2y_1%7D%7By_2-y_1%7D)，y轴上的截距为：![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bx_1y_1-x_1y_2%7D%7Bx_2-x_1%7D)



#### Solution-y=kx+b

Intuition: Cuz we wanna know the count of dots on lines, we can easily think of line as a key, and we just use the count of dots on it as value. So how to uniquely identify a line, we can use y=kx+b. But there are a few problems with this:

* k is a float, there can be cases where a nearly parallel lines land to the same slope, so we can just keep the numerator and the denominator and simplify them, and use the 2 as a tuple to identify
* 2 other special cases, parallel to x and y line, for parallel to x line, the slope is 0, but to y, is infinity, and to represent them, and to be consistent, see the code after. And there is a special case here as well.

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points)<=2:
            return len(points)
        lines = collections.defaultdict(set)
        for i in range(len(points)-1):
            x1,y1 = points[i]
            for j in range(i+1, len(points)):
                x2,y2 = points[j]
                slope = self.get_slope(x1,y1,x2,y2)
                b = 0
                if slope[0]!=0 and slope[1]!=0:
                    b = self.get_b(x1,y1,x2,y2)
                    b = str(b)
                elif slope[0]==0 and slope[1]==0: # didn't distiguish between x=0 and y=0
                    b = "s" # as we cannot identify the lines parallel to x and y line with the origin on them.
                elif slope[0]==0:
                    b = str(slope[1])
                else:
                    b = str(slope[0])
                key = str(slope[0])+"*"+str(slope[1])+"*"+b
                lines[key].add((x2,y2))
                lines[key].add((x1,y1))
        return max([len(value) for value in lines.values()])
                    
                
    def get_b(self, x1, y1, x2, y2):
        return (x2*y1-x1*y2)/(x2-x1)
        
                
    def get_slope(self, x1, y1, x2, y2):
        if x1==x2:
            return 0,x1
        elif y1==y2:
            return y1,0
        else:
            divisor = self.gcd(y2-y1, x2-x1)
            return (y2-y1)//divisor, (x2-x1)//divisor
            
            
    def gcd(self, x, y):
        if y==0:
            return x
        else:
            return self.gcd(y, x%y)                
```



#### Solution-a line with a slope

Ref: https://www.youtube.com/watch?v=7FPL7nAi9aM

![image-20200125175557438](https://tva1.sinaimg.cn/large/006tNbRwgy1gb9mtptw7dj31c00u0nfy.jpg)

```python
# based on huahua's
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        ans = 0
        
        
        for i in range(len(points)):
            sameP = 1 # first add the point itself
            otherP = 0 # 局部最优解
            dic = {}
            point1 = points[i]
            for j in range(i+1, len(points)):
                point2 = points[j]
                if point1[0]==point2[0] and point1[1]==point2[1]:
                    sameP+=1
                else:
                    slope = self.getSlope(point1, point2)
                    dic[slope] = dic.get(slope, 0)+1
                    otherP = max(otherP, dic[slope])
            ans = max(ans, sameP+otherP)
            
        return ans
    
    def getSlope(self, i, j):
        dx = i[0]-j[0]
        dy = i[1]-j[1]
        
        if dx==0:
            return (i[0], 0)
        if dy==0:
            return (0, i[1])
        
        d = self.gcd(dx, dy)
        return (dx//d, dy//d)
    
    def gcd(self, x, y):
        if y==0:
            return x
        else:
            return self.gcd(y, x%y)
```





### 