12.19

#### 7.Reverse Integer

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

#### 165. Compare Version Numbers

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



#### 66. Plus One

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



#### 8. String to Integer (atoi)

https://leetcode.com/problems/string-to-integer-atoi/description/

* Solution 忽略



#### 258. Add Digits

https://leetcode.com/problems/add-digits/description/

* Solution-easy



#### 67. Add Binary

https://leetcode.com/problems/add-binary/

* Solution-easy



12.21

#### 43. Multiply Strings

https://leetcode.com/problems/multiply-strings/description/

* solution

Ref: https://leetcode.wang/leetCode-43-Multiply-Strings.html 解法2



#### 29. Divide Two Integers

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



#### 365. Water and Jug Problem

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



#### 204. Count Primes

https://leetcode.com/problems/count-primes/description/

* Solution- Sieve of Eratosthenes

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
            if prime[divisor]:
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



#### 1. Two Sum

https://leetcode.com/problems/two-sum/

* Solution-easy- think about how to solve the problem in one pass



#### 15. 3Sum

https://leetcode.com/problems/3sum/

* solution-how to achieve O(N2)

Ref: https://leetcode.wang/leetCode-15-3Sum.html

注意边界的处理



#### 18. 4Sum

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







### syntax

#### [Convert base-2 binary number string to int](https://stackoverflow.com/questions/8928240/convert-base-2-binary-number-string-to-int)

```python
>>> int('11111111', 2)
255
```



#### Covert int to binary number string-bin()



