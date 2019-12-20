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



#### 43. Multiply Strings

https://leetcode.com/problems/multiply-strings/description/





### syntax

#### [Convert base-2 binary number string to int](https://stackoverflow.com/questions/8928240/convert-base-2-binary-number-string-to-int)

```python
>>> int('11111111', 2)
255
```



#### Covert int to binary number string-bin()



