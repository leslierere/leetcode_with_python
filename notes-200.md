## 7. Reverse Integer

Ref: https://leetcode.com/problems/reverse-integer/

did@5.18, complicated

```python
class Solution:
    def reverse(self, x: int) -> int:
        # consider overflow
        # consider negative
        res = 0
        
        negative = x<0
        x = -x if negative else x
        upper = 2**31-1 # 2147483647
        upper_less = upper%10**9 # 147483647
        
        while x:
            digit = x%10
            if res//10**8: # if we will have 10 digits after adding digit
                if res//10**8>2:
                    return 0
                elif res//10**8 == 2:
                    temp = (res%10**8)*10+digit # get the number after appending digit at the end and strip out the most significant digit
                    if not negative:
                        if temp>upper_less:
                            return 0
                    if negative:
                        if temp>upper_less+1:
                            return 0
                        else:
                            return -res*10 - digit
                    
                
            res = res*10+digit
            x = x//10
            
        res = -res if negative else res
        return res
```

### Cool way to see overflow

Ref: https://leetcode.wang/leetCode-7-Reverse-Integer.html

Time complexity: x is the input, O(log10(x))

> rev = rev * 10 + pop;
>
> 对于大于 intMax 的讨论，此时 x 一定是正数，pop 也是正数。
>
> - 如果 rev > intMax / 10 ，那么没的说，此时肯定溢出了。
> - 如果 rev == intMax / 10 = 2147483647 / 10 = 214748364 ，此时 rev * 10 就是 2147483640 如果 pop 大于 7 ，那么就一定溢出了。但是！如果假设 pop 等于 8，那么意味着原数 x 是 8463847412 了，输入的是 int ，而此时是溢出的状态，所以不可能输入，所以意味着 pop 不可能大于 7 ，也就意味着 rev == intMax / 10 时不会造成溢出。
> - 如果 rev < intMax / 10 ，意味着 rev 最大是 214748363 ， rev * 10 就是 2147483630 , 此时再加上 pop ，一定不会溢出。 
>
> 对于小于 intMin 的讨论同理

```java
public int reverse(int x) {
    int rev = 0;
    while (x != 0) {
        int pop = x % 10;
        x /= 10;
        if (rev > Integer.MAX_VALUE/10 ) return 0;
        if (rev < Integer.MIN_VALUE/10 ) return 0;
        rev = rev * 10 + pop;
    }
    return rev;
}
```



For languages that have overflows, we can use, but this is not generalized enough as the previous one

Ref: https://leetcode.com/problems/reverse-integer/discuss/4060/My-accepted-15-lines-of-code-for-Java

```java
public int reverse(int x)
{
    int result = 0;

    while (x != 0)
    {
        int tail = x % 10;
        int newResult = result * 10 + tail;
        if ((newResult - tail) / 10 != result) // point here! 
        { return 0; }
        result = newResult;
        x = x / 10;
    }

    return result;
}
```



Did previously

```python
class Solution:
    def reverse(self, x: int) -> int:
        
        number = int(str(abs(x))[::-1])
        if x<0:
            number = -number
        if number>(1<<31)-1 or number< -(1<<31): # but in the question, it says Assume the environment does not allow you to store 64-bit integers (signed or unsigned).
            return 0
        return number

```



## 8. String to Integer (atoi)

Ref: https://leetcode.com/problems/string-to-integer-atoi/

did@2021.5.18

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        max_int = 2**31-1
        min_int = -2**31
        s = s.strip()
        res = 0
        if not s:
            return res
        start = 0
        sign = 1
        if s[0]=="-":
            sign = -1
            start = 1
        elif s[0]=="+":
            start = 1
        elif not s[0].isdigit():
            return res
            
        for i in range(start, len(s)):
            if not s[i].isdigit():
                break
            res = res*10+int(s[i])
        
        res*=sign
            
        if res > max_int:
            return max_int
        elif res < min_int:
            return min_int
        return res
```











## 38. Count and Say

Ref: https://leetcode.com/problems/count-and-say/

这个人做的比较简单，当然我可以不recursive

我开始记了一下begin的index，但其实没必要，因为还得知道现在的index，直接count就行

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        if n == 1:
            return "1"
        
        s = self.countAndSay(n-1)
        cnt = 0
        temp_s = ""
        currChar = s[0]
        for c in s:           
            if c != currChar:
                temp_s += str(cnt) + currChar
                currChar = c
                cnt = 0
                
            cnt += 1

        return (temp_s + str(cnt) + currChar)
```























## 135. Candy

https://leetcode.com/problems/candy/

### Solution1

Ref: https://leetcode.wang/leetcode-135-Candy.html



### Solution2

Ref: https://leetcode.com/problems/candy/discuss/135698/Simple-solution-with-one-pass-using-O(1)-space







## 545. Boundary of Binary Tree

https://leetcode.com/problems/boundary-of-binary-tree/

### Solution-dfs-pre,postorder

did@21.6.16

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: TreeNode) -> List[int]:
        res = []
        res.append(root.val)
        self.preorder(root.left, True, res)
        self.postorder(root.right, True, res)
        return res
        
    def preorder(self, root, is_bound, res):
        if not root:
            return
        if not root.left and not root.right:
            res.append(root.val)
            return
        if is_bound:
            res.append(root.val)
        if root.left and root.right:
            self.preorder(root.left, is_bound, res)
            self.preorder(root.right, False, res)
        elif root.left:
            self.preorder(root.left, is_bound, res)
        else:
            self.preorder(root.right, is_bound, res)
            
    def postorder(self, root, is_bound, res):
        if not root:
            return
        if not root.left and not root.right:
            res.append(root.val)
            return
        
        if root.left and root.right:
            self.postorder(root.left, False, res)
            self.postorder(root.right, is_bound, res)
        elif root.left:
            self.postorder(root.left, is_bound, res)
        else:
            self.postorder(root.right, is_bound, res)
        if is_bound:
            res.append(root.val)
```





## 166. Fraction to Recurring Decimal

https://leetcode.com/problems/fraction-to-recurring-decimal/

did@21.6.19

```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        result = ""
        if numerator*denominator<0:
            result = "-"
        numerator = abs(numerator)
        denominator = abs(denominator)
        
        result += str(numerator//denominator)
        remainder = numerator%denominator
        if remainder==0:
            return result
        
        remainder_dic = dict()
        remainder_dic[remainder] = 0
        floats = ""
        repeat = 0
        while remainder:
            power = 1
            while remainder*10**power<denominator:
                power+=1
            numerator = remainder*10**power
            temp = numerator//denominator
            remainder = numerator%denominator
            floats+="0"*(power-1)+str(temp)
            if remainder in remainder_dic:
                repeat=len(floats)-remainder_dic[remainder]
                return result+"."+floats[:-repeat]+"("+floats[-repeat:]+")"
            remainder_dic[remainder] = len(floats)
        
        return result+"."+floats
```











## 190. Reverse Bits

https://leetcode.com/problems/reverse-bits/

#### Solution

Ref: https://leetcode.com/problems/reverse-bits/discuss/54738/Sharing-my-2ms-Java-Solution-with-Explanation

```java
public int reverseBits(int n) {
    if (n == 0) return 0;
    
    int result = 0;
    for (int i = 0; i < 32; i++) {
        result <<= 1;
        if ((n & 1) == 1) result++;
        n >>= 1;
    }
    return result;
}
```





## 191. Number of 1 Bits

https://leetcode.com/problems/number-of-1-bits/

### Solution-just count

### Solution-tricky

Ref: https://leetcode.wang/leetcode-191-Number-of-1-Bits.html

> 当我们对一个数减 `1` 的话，比如原来的数是 `...1010000`，然后减一就会向前借位，直到遇到最右边的第一个 `1`，变成 `...1001111`，然后我们把它和原数按位与，就会把从原数最右边 `1` 开始的位置全部置零了 `...10000000`。
>
> 有了这个技巧，我们只需要把原数依次将最右边的 `1` 置为 `0`，直到原数变成 `0`，记录总共操作了几次即可。
