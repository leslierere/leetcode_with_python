### 389. Find the Difference

https://leetcode.com/problems/find-the-difference/

#### Solution

Ref: https://leetcode.com/problems/find-the-difference/discuss/86881/Python-solution-which-beats-96

两个相同的东西exclusiveor就会变成0, 并且exclusiveor满足交换律



### 136. Single Number

https://leetcode.com/problems/single-number/

#### Solution

和上题思路一模一样



### 318. Maximum Product of Word Lengths-$

https://leetcode.com/problems/maximum-product-of-word-lengths/description/

#### Solution-bit mask, hash table-worth

Ref: [https://leetcode.com/problems/maximum-product-of-word-lengths/discuss/76959/JAVA-Easy-Version-To-Understand!!!!!!!!!!!!!!!!!](https://leetcode.com/problems/maximum-product-of-word-lengths/discuss/76959/JAVA-Easy-Version-To-Understand!!!!!!!!!!!!!!!!!)

We establish a 26 length list, for every item it would be True as long as there is at least one of that letter(thus we need use or).

To test every combination, for the pair that has no same letter with another, the and result for them should be zero.

> How to set n-th bit? Use standard bitwise trick : `n_th_bit = 1 << n`.

This means 1 would be left shifted n times

> How to compute bitmask for a word? Iterate over the word, letter by letter, compute bit number corresponding to that letter `n = (int)ch - (int)'a'`, and add this n-th bit `n_th_bit = 1 << n`into bitmask `bitmask |= n_th_bit`.





### 169. Majority Element

https://leetcode.com/articles/majority-element/

#### Solution-Boyer-Moore Voting Algorithm-worth

https://leetcode.com/articles/majority-element/

#### Solution-bit

https://leetcode.com/problems/majority-element/discuss/51612/C%2B%2B-6-Solutions

还不太理解。。。。





### 面试题 16.07. 最大数值



编写一个方法，找出两个数字`a`和`b`中最大的那一个。不得使用if-else或其他比较运算符。

**示例：**

```
输入： a = 1, b = 2
输出： 2
```

#### Solution

Did by me-v1

```python
class Solution:
    def maximum(self, a: int, b: int) -> int:
        k = int(b/2-a/2)>>31 # 开始写的(b-a)>>31但是遇到了溢出
        # if b>a, k=0
        # else k=-1
        return -k*a+(1+k)*b
```

但我觉得这个做法比较好, 更加bit manipulation, 为什么要加同号异号的判断呢，因为同号一定不会导致溢出

```c++
class Solution {
    public int maximum(int a, int b) {
        // 先考虑没有溢出时的情况，计算 b - a 的最高位，依照题目所给提示 k = 1 时 a > b，即 b - a 为负
        int k = b - a >>> 31;
        // 再考虑 a b 异号的情况，此时无脑选是正号的数字
        int aSign = a >>> 31, bSign = b >>> 31;
        // diff = 0 时同号，diff = 1 时异号
        int diff = aSign ^ bSign;
        // 在异号，即 diff = 1 时，使之前算出的 k 无效，只考虑两个数字的正负关系
        k = k & (diff ^ 1) | bSign & diff;
        return a * k + b * (k ^ 1);
    }
}

作者：1ujin
链接：https://leetcode-cn.com/problems/maximum-lcci/solution/chun-wei-yun-suan-bu-yong-longzhuan-huan-bu-yong-n/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

did by me, 考虑溢出, 但这个背景应该还是在考虑在java这些里面int是32位, 所以相应的input不会超过32位, 但感觉在python里面也是这个样子，毕竟正数到负数的变换还是flip，+1，只是溢出了会有特殊的处理。。。。。

```python
class Solution:
    def maximum(self, a: int, b: int) -> int:
        k = b-a>>31 # 溢出会变成-2
        k = k>>31
        # if b>a, k=0
        # else k=-1
        return -k*a+(1+k)*b
```



### 268. Missing Number

https://leetcode.com/problems/missing-number/

#### Solution-math

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        return len(nums)*(len(nums)+1)//2-sum(nums)
```



#### Solution-bit-worth

https://leetcode.com/problems/missing-number/discuss/69791/4-Line-Simple-Java-Bit-Manipulate-Solution-with-Explaination

Hint:  a^b^b =a





### 201. Bitwise AND of Numbers Range

https://leetcode.com/problems/bitwise-and-of-numbers-range/

#### Solution-bit manipulation

Ref: https://leetcode.com/problems/bitwise-and-of-numbers-range/

https://www.youtube.com/watch?v=-qrpJykY2gE

> all the columns to the right of flipped bit is also flipped.