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



## 15. 3Sum

Ref: https://leetcode.com/problems/3sum/

#### Solution-2 pointer

https://leetcode.com/problems/3sum/discuss/7380/Concise-O(N2)-Java-solution

Also, unlike 4Sum, the target is just 0, so

> A trick to improve performance: once nums[i] > 0, then break.
> Since the nums is sorted, if first number is bigger than 0, it is impossible to have a sum of 0.

If target varies, then if nums[i] > target, we can break



## 18. 4Sum

Ref: https://leetcode.com/problems/4sum/

> #### General Idea
>
> If you have already read and implement the 3sum and 4sum by using the sorting approach: reduce them into 2sum at the end, you might already got the feeling that, all ksum problem can be divided into two problems:
>
> 
>
> 1. 2sum Problem
> 2. Reduce K sum problem to K – 1 sum Problem

But in the 2sum problem indicated here, it is not the same as the problem1, since in problem 1, it says it only has one valid solution, thus we don't consider duplicates there.

Time complexity: say n numbers in nums, we ask for k sums, first we don't consider the time complexity of sort()

for the most innate 2 sums, we have n-(k-2) numbers when we calculate the 2sum in the subarray, and the time complexity would be O(n-(k-2) ) with 2 pointers, i.e. O(N)

If 3 sum, then we fix one number(O(N))*how_many_sub_results(Maximum is O(numbers in array/2))+O(N) for 2sum, i.e. O(N^2)+O(N)->O(N^2)

if 4 sum, then N* time complexity of 3sum

Time complexity is O(N^(K-1)).



```java
 public class Solution {
        int len = 0;
        public List<List<Integer>> fourSum(int[] nums, int target) {
            len = nums.length;
            Arrays.sort(nums);
            return kSum(nums, target, 4, 0);
        }
       private ArrayList<List<Integer>> kSum(int[] nums, int target, int k, int index) {
            ArrayList<List<Integer>> res = new ArrayList<List<Integer>>();
            if(index >= len) {
                return res;
            }
            if(k == 2) {
            	int i = index, j = len - 1;
            	while(i < j) {
                    //find a pair
            	    if(target - nums[i] == nums[j]) {
            	    	List<Integer> temp = new ArrayList<>();
                    	temp.add(nums[i]);
                    	temp.add(target-nums[i]);
                        res.add(temp);
                        //skip duplication
                        while(i<j && nums[i]==nums[i+1]) i++;
                        while(i<j && nums[j-1]==nums[j]) j--;
                        i++;
                        j--;
                    //move left bound
            	    } else if (target - nums[i] > nums[j]) {
            	        i++;
                    //move right bound
            	    } else {
            	        j--;
            	    }
            	}
            } else{
                for (int i = index; i < len - k + 1; i++) {
                    //use current number to reduce ksum into k-1sum
                    ArrayList<List<Integer>> temp = kSum(nums, target - nums[i], k-1, i+1);
                    if(temp != null){
                        //add previous results
                        for (List<Integer> t : temp) {
                            t.add(0, nums[i]);
                        }
                        res.addAll(temp);
                    }
                    while (i < len-1 && nums[i] == nums[i+1]) {
                        //skip duplicated numbers
                        i++;
                    }
                }
            }
            return res;
        }
    }
```



did@21.5.21, not general enough

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if len(nums)<4:
            return []
        
        # fix first number, and the rest becomes 3sum
        result = []
        nums.sort()
        for i in range(len(nums)-3):
            if i>0 and nums[i]==nums[i-1]:
                continue
                
            sub_target = target-nums[i]
            
            for j in range(i+1, len(nums)-2):
                if j>i+1 and nums[j]==nums[j-1]:
                    continue
                left = j+1
                right = len(nums)-1
                while left<right:
                    sub_result = nums[j]+nums[left]+nums[right]
                    if sub_result == sub_target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        left+=1 # we need increase left first
                        while left<right and nums[left-1]==nums[left]:
                            left+=1 # make sure no duplicate third number
                        right-=1
                        
                    elif sub_result<sub_target:
                        left+=1
                    else:
                        right-=1
                        
        return result
```
