### 389. Find the Difference

https://leetcode.com/problems/find-the-difference/

#### Solution

Ref: https://leetcode.com/problems/find-the-difference/discuss/86881/Python-solution-which-beats-96

两个相同的东西exclusiveor就会变成0



### 136. Single Number

https://leetcode.com/problems/single-number/

#### Solution

和上题思路一模一样



### 318. Maximum Product of Word Lengths

https://leetcode.com/problems/maximum-product-of-word-lengths/description/

#### Solution-bit mask, hash table-worth

Ref: [https://leetcode.com/problems/maximum-product-of-word-lengths/discuss/76959/JAVA-Easy-Version-To-Understand!!!!!!!!!!!!!!!!!](https://leetcode.com/problems/maximum-product-of-word-lengths/discuss/76959/JAVA-Easy-Version-To-Understand!!!!!!!!!!!!!!!!!)

We establish a 26 length list, for every item it would be True as long as there is at least one of that letter(thus we need use or).

To test every combination, for the pair that has no same letter with another, the and result for them should be zero.

> How to set n-th bit? Use standard bitwise trick : `n_th_bit = 1 << n`.

This means 1 would be left shifted n times

> How to compute bitmask for a word? Iterate over the word, letter by letter, compute bit number corresponding to that letter `n = (int)ch - (int)'a'`, and add this n-th bit `n_th_bit = 1 << n`into bitmask `bitmask |= n_th_bit`.