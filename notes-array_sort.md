### 1. Two Sum

https://leetcode.com/problems/two-sum/

#### Solution-one pass

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = dict()
        
        for i,num in enumerate(nums):
            if target-num in seen:
                return [seen[target-num], i]
            seen[num] = i
```



### 15. 3Sum-$

Ref: https://leetcode.com/problems/3sum/

#### Solution-built upon 2sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums)<3:
            return []
        res = []
        nums.sort()
        
        for end in range(2, len(nums)):
            if end+1<len(nums) and nums[end+1]==nums[end]:
                continue
            target = -nums[end]
            seen = set()
            for i in range(end):
                if not (i+1<end and nums[i]==nums[i+1]):
                    diff = target-nums[i]
                    if diff in seen:
                        res.append([diff, nums[i], nums[end]])
                seen.add(nums[i])
                
        return res
```



#### Solution-2 pointer

https://leetcode.com/problems/3sum/discuss/7380/Concise-O(N2)-Java-solution

Also, unlike 4Sum, the target is just 0, so

> A trick to improve performance: once nums[i] > 0, then break.
> Since the nums is sorted, if first number is bigger than 0, it is impossible to have a sum of 0.

If target varies, then if nums[i] > target, we can break

**how to approve we won't miss**?????

like now we are at nums[left] and nums[right], will actually nums[left-1] + nums[right+1]==target???



### 16. 3Sum Closest

https://leetcode.com/problems/3sum-closest/

* Solution-two pointers

how to think of 2 pointers: https://leetcode.com/problems/3sum-closest/discuss/7871/Python-O(N2)-solution/210643

> I think the insight is something like this - Given an array and a brute force algorithm that seems waaay too slow (n^3), try to think of ways that we could get it to n^2, nlogn, n. If the given array problem is the type of a problem where order/index doesn't matter, always consider sorting the array. Once you've got it sorted, you have a great heuristic to use to iterate over the array.
>
> 
>
> If you've gotten to that point, and are wondering how to traverse the array, 1, 2, 3+ pointers is always something that should be at the top of your list of things to consider when tackling an unfamiliar problem.

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        result = sum(nums[:3])
        
        for i in range(len(nums)-2):
            k, j = i+1, len(nums)-1
            
            while k<j:
                candidate = nums[i]+nums[j]+nums[k]
                if candidate==target:
                    return target
                elif candidate<target:
                    k+=1
                else:
                    j-=1
                if abs(candidate-target)<abs(result-target):
                    result = candidate
                    
        return result
```



### 18. 4Sum

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





### 27. Remove Element-two pointers

https://leetcode.com/problems/remove-element/



### 26. Remove Duplicates from Sorted Array-two pointers

https://leetcode.com/problems/remove-duplicates-from-sorted-array/

Ref: https://leetcode.com/problems/remove-duplicates-from-sorted-array/discuss/11782/Share-my-clean-C%2B%2B-code

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0 # last index till which no duplicates exist
        
        for j in range(1, len(nums)):
            if nums[i]!=nums[j]:
                nums[i+1]=nums[j]
                i+=1
                
        return i+1
```



### 80. Remove Duplicates from Sorted Array II

https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/

#### Solution-2 pointers

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        right = 1
        index = 1 # there are no duplicates on the left of index
        last_no = nums[0]
        last_count = 1
        
        while right<len(nums):
            if nums[right]!=last_no:
                last_count = 1
                nums[index], nums[right] = nums[right], nums[index]
                last_no = nums[index]
                index+=1
                
            elif last_count==1:
                last_count+=1
                nums[index], nums[right] = nums[right], nums[index]
                index+=1
            right+=1
                
                
        return index
```





### 277. Find the Celebrity

https://leetcode.com/problems/find-the-celebrity/



7.24

### 189. Rotate Array

https://leetcode.com/problems/rotate-array/

#### solution

Approach #4 in Solution Using Reverse 



### 41. First Missing Positive

https://leetcode.com/problems/first-missing-positive/

* solution

  https://www.cnblogs.com/grandyang/p/4395963.html

   



### 299. Bulls and Cows

https://leetcode.com/problems/bulls-and-cows/

* solution-我做的稍微复杂了点，思想类似，学会这种写法

  ```python
  def getHint(self, secret, guess):
          A = sum(a==b for a,b in zip(secret, guess))
          B = collections.Counter(secret) & collections.Counter(guess)
          return "%dA%dB" % (A, sum(B.values()) - A)
  ```

  

### 274. H-Index

https://leetcode.com/problems/h-index/

* Solution-besides mine

  用extra space来做，faster

  [https://leetcode.com/problems/h-index/discuss/70818/Java-O(n)-time-with-easy-explanation.](https://leetcode.com/problems/h-index/discuss/70818/Java-O(n)-time-with-easy-explanation.)



7.25

### 275. H-Index II

https://leetcode.com/problems/h-index-ii/



### 134. Gas Station

https://leetcode.com/problems/gas-station/

#### Solution

We can absolutely know that if total cost is larger than total gas, we got no solution. And for total cost==total gas, how to prove we must have a solution?

Suppose there is a new array which records the increment amount at a gas station, increments[i] = gas[i]-cost[i].

If no solution, *starting from any gas station, at least one of the cumulative amount of increments will be less than zero*, and suppose when total cost==total gas, there is still no solution.

```
gas = [1,2,3,4,5], cost = [3,4,5,1,2], increments = [-2, -2, -2, 3, 3]
```

Use this example, starting from i = 2, the cumulative value is -2, 1, 4, 2, 0 consecutively. So starting from i=2 is impossible to finish the route. See the cumulative value on the other side, it is actually the sum of the sub array. To reiterate *starting from any gas station, at least one of the cumulative amount of increments will be less than zero*, it is actually for any starting index, at least one sum of subarray is less than zero. Since we know the sum of the max length array is 0, if we find one sum of subarray<0, name it **array1** with sum named **sum1**, then the sum of subarray composed of the remaining numbers>0, name it **array2**, with **sum2**, we should finish going through **array2** before **array1**, in this case, we can make sure in **array1** we won't starve as sum1+sum2 = 0. However, you may ask, in the **array2**, there can be subarray whose sum<0, then for the remaining part, the sum also>0, and we should go through that first, and in the worst case, we always find one subarray whose sum<0, and we do this recursively, we will finnally land at an array whose sum>0 and there is just one element there. So we can definitly has one solution. 

Ref: https://leetcode.com/problems/gas-station/discuss/42572/Proof-of-%22if-total-gas-is-greater-than-total-cost-there-is-a-solution%22.-C%2B%2B

好好体会。。。



### 243. Shortest Word Distance

https://leetcode.com/problems/shortest-word-distance/



### 244. Shortest Word Distance II

https://leetcode.com/problems/shortest-word-distance-ii/





7.26

### 245. Shortest Word Distance III

https://leetcode.com/problems/shortest-word-distance-iii/



### 55. Jump Game

https://leetcode.com/problems/jump-game/

* Solution-greedy method



### 45. Jump Game II

https://leetcode.com/problems/jump-game-ii/

* Solution-greedy method, didn't submit

* Solution-dp

Did@21.5.26, https://leetcode.com/problems/jump-game-ii/discuss/1233094/Easy-to-understand-python-solution

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = [1001 for _ in range(len(nums))]
        steps[0] = 0
        
        for i in range(len(nums)):
            for number in range(1, nums[i]+1):
                next_pos = i+number
                if next_pos<len(nums):
                    steps[next_pos] = min(steps[i]+1, steps[next_pos])
                else:
                    break
                
        return steps[-1]
```





### 121. Best Time to Buy and Sell Stock

https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

#### Solution

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        
        min_price = prices[0]
        
        for price in prices:
            profit = max(price-min_price, profit)
            min_price = min(min_price, price)
            
        return profit
```

#### Solution-maximum subarray

wow

Ref: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/discuss/39038/Kadane's-Algorithm-Since-no-one-has-mentioned-about-this-so-far-%3A)-(In-case-if-interviewer-twists-the-input)



### 122. Best Time to Buy and Sell Stock II

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

#### Solution-stack

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        stack = [prices[0]]
        profit = 0
        
        for i in range(1, len(prices)):
            if prices[i]<=stack[-1]:
                if len(stack)>1:
                    profit += stack[-1]-stack[0]
                stack = [prices[i]]
            else:
                stack.append(prices[i])
        
        if stack:
            profit+=stack[-1]-stack[0]
        return profit
```



#### Solution

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0
        
        for i in range(1, len(prices)):
            diff = prices[i]-prices[i-1]
            if diff>0:
                res+=diff
            
        return res
```





7.27

### 123. Best Time to Buy and Sell Stock III

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

#### Solution-dynamic programming-$$$

Ref: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/135704/Detail-explanation-of-DP-solution

TLE

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<2:
            return 0
        
        dp = [[0 for i in range(len(prices))] for j in range(3)]
        
        for k in range(1, len(dp)):
            for i in range(1, len(prices)): # till day i, if we make k transactions, the maximum profit
                dp[k][i] = dp[k][i-1] # we don't sell today
                for j in range(i):
                    dp[k][i] = max(dp[k][i], prices[i]-prices[j]+dp[k-1][j]) 
        return dp[2][-1]
```



做一个min_val的替换, 为啥是min_val呢，cuz you always want to let selling on i day get the maximum profit, so as like single transaction, you would like to find the price before that is the lowest, however, we cannot forget about dp\[k-1][j].

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<2:
            return 0
        
        dp = [[0 for i in range(len(prices))] for j in range(3)]
        for k in range(1, len(dp)):
            for i in range(1, len(prices)): # till day i, if we make k transactions, the maximum profit
                min_val = prices[0]
                for j in range(1, i): # and we can find out, everytime, we are only updating i-1, so we can replace j with i-1
                    min_val = min(min_val, prices[j]-dp[k-1][j])
                dp[k][i] = max(dp[k][i-1], prices[i]-min_val) 
        return dp[2][-1]
```



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
      	if len(prices)<2:
          return 0
        dp = [[0 for i in range(len(prices))] for j in range(3)]
        
        for k in range(1,3): # number of transactions
            min_val = prices[0]
            dp[k][1] = max(dp[k][0], prices[1]-min_val)
            for i in range(2, len(dp[0])): # selling day
                # min_val = prices[0]
                #for j in range(1,i): # buying day, everytime, what we actually update is just i-1
                min_val = min(min_val, prices[i-1]-dp[k-1][i-2]) 
                   	# min_val = min(min_val, prices[j]-dp[k-1][j-1]) 
                dp[k][i] = max(dp[k][i-1], prices[i]-min_val)
                    
        return dp[2][-1]
      
```



#### Solution-specific to 2 transactions

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        left = [0 for i in range(len(prices))]
        right = [0 for i in range(len(prices))]
        
        min_price = prices[0]
        for i in range(1, len(prices)):
            left[i] = max(left[i-1], prices[i]-min_price)
            min_price = min(min_price, prices[i])
        
        max_price = prices[len(prices)-1]
        for i in range(len(prices)-2, -1, -1):
            right[i] = max(right[i+1], max_price-prices[i])
            max_price = max(max_price, prices[i])
            
        max_profit = 0
        for i in range(len(prices)):
            max_profit = max(max_profit, left[i]+right[i])
            
        return max_profit
```





### 188. Best Time to Buy and Sell Stock IV

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/

* Solution-dp

  ***worth thinking and doing***



Refer to 123

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices)<2:
            return 0
        dp = [[0 for i in range(len(prices))] for j in range(k+1)]
        
        for k in range(1,k+1): # number of transactions
            min_val = prices[0]
            dp[k][1] = max(dp[k][0], prices[1]-min_val)
            for i in range(2, len(dp[0])): # selling day
                min_val = min(min_val, prices[i-1]-dp[k-1][i-2]) 
                dp[k][i] = max(dp[k][i-1], prices[i]-min_val)
                    
        return dp[k][-1]
```





### 11. Container With Most Water

https://leetcode.com/problems/container-with-most-water/

#### solution- 2 pointer

https://leetcode.wang/leetCode-11-Container-With-Most-Water.html

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height)-1
        min_height = min(height[left], height[right])
        res = (right-left)*min_height
        
        while left<right:
            while height[left]<=min_height and left<right:
                left+=1
            while height[right]<=min_height and left<right:
                right-=1
            min_height = min(height[left], height[right])
            res = max((right-left)*min_height, res)
            
        return res
```



### 41. First Missing Positive

Ref: https://leetcode.com/problems/first-missing-positive/

#### Solution

比较容易想到的

Ref: https://leetcode.wang/leetCode-41-First-Missing-Positive.html

#### Solution

很巧妙的, Ref: https://leetcode.com/problems/first-missing-positive/discuss/17080/Python-O(1)-space-O(n)-time-solution-with-explanation



### 334. Increasing Triplet Subsequence

https://leetcode.com/problems/increasing-triplet-subsequence/

* solution

  分类的顺序要么小到大要么大到小，不要一下这这区间一下一下又跳到那个，而且小大，大小也是有讲究的，另外初始值的设置也很关键

  https://leetcode.com/problems/increasing-triplet-subsequence/discuss/78995/Python-Easy-O(n)-Solution

  ```python
  def increasingTriplet(nums):
      first = second = float('inf')
      for n in nums:
          if n <= first:
              first = n
          elif n <= second:
              second = n
          else:
              return True
      return False
  ```



### 42. Trapping Rain Water

Ref: https://leetcode.com/problems/trapping-rain-water/

#### Solution-stack

My post: https://leetcode.com/problems/trapping-rain-water/discuss/1232796/Very-simple-and-clear-code-with-stack-python-solution

The basic idea is we count the area level by level horizontally. 

To illustrate, in below, the number in shaded area means the order when it is counted. 

Area 1 would be counted at index 6, area 2 will be counted at index 7, area 3 and 4 will be counted at index 9.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqwimj7l9wj31400u0he6.jpg" alt="image-20210526160325696" style="zoom:30%;" />

When using stack, we only add the current index to the stack if the stack is empty, or after adding the index, the height of indexes in the stack is still in descending order.

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        area = 0
        stack = []
        
        for i, h in enumerate(height): # stack keeps descending order
            
            while stack and h>=height[stack[-1]]: 
                if len(stack)==1: 
					# this would only happen when we at the beggining when we have been unable to trap water yet 
					# or we have counted the water before index i, so when the current height is higher then the 
					# only one we have, that is the height of stack[-1](stack[-1] will always equal to i-1). We just 
					# pop it out as stack[-1] won't be used to count at all, and add i in stack.
                    stack.pop()
                else:
                    mid = stack.pop()
                    left = stack[-1]
                    area+=(min(height[left], h)-height[mid])*(i-left-1)
                    
            stack.append(i)
                    
        return area
```





Side notes: I feel like when we use stack, we should try to think of is under what condition should we add element to it and when to pop.

#### Solution-two pointers

***worth thinking and doing***

similar to problem 11



### 128. Longest Consecutive Sequence

https://leetcode.com/problems/longest-consecutive-sequence/

#### Solution-hashset

Ref: https://leetcode.com/problems/longest-consecutive-sequence/discuss/41057/Simple-O(n)-with-Explanation-Just-walk-each-streak



#### Solution-hash map

Ref: https://leetcode.com/problems/longest-consecutive-sequence/discuss/41055/My-really-simple-Java-O(n



### 164. Maximum Gap

https://leetcode.com/problems/maximum-gap/submissions/

#### Solution-radix sort-$

@21.6.18

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums)<2:
            return 0
        maxNum = max(nums)
        if maxNum==0:
            return 0
        
        radix = 0 # the number of digits we have
        num = maxNum
        while num:
            num=num//10
            radix+=1
         
        for i in range(radix):
            divisor=10**i
            temp = [[] for i in range(10)] 
            for num in nums:
                index = (num//divisor)%10
                temp[index].append(num)
            nums = []
            for subarray in temp:
                for num in subarray:
                    nums.append(num)
        
        maxDiff = 0
        for i in range(1, len(nums)):
            maxDiff = max(maxDiff, nums[i]-nums[i-1])
            
        return maxDiff
```



#### Solution-bucket sort

Ref: https://leetcode.com/problems/maximum-gap/discuss/1240543/Python-Bucket-sort-explained

可以看一下cs61b notes里关于bucket sort的理解

Ref: https://leetcode.com/problems/maximum-gap/discuss/50643/bucket-sort-JAVA-solution-with-explanation-O(N)-time-and-space

> Let gap = ceiling[(***max*** - ***min*** ) / (N - 1)]. We divide all numbers in the array into n-1 buckets, where k-th bucket contains all numbers in [***min*** + (k-1)gap, ***min*** + k*gap). Since there are n-2 numbers that are not equal ***min*** or ***max*** and there are n-1 buckets, **at least one of the buckets are empty**. We only need to store the largest number and the smallest number in each bucket.

The key here is that **at least one of the buckets are empty**. Also, I feel like it is needed that we get the ceiling rather than the floor.

did@21.6.25

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        min_val = min(nums)
        max_val = max(nums)
        if len(nums)<3 or max_val==min_val:
            return max_val-min_val
        bucket = [[float("inf"), float("-inf")] for i in range(len(nums)-1)]
        
        for num in nums:
            idx = (num-min_val)*len(bucket)//(max_val-min_val)
            if idx==len(bucket):
                idx = len(bucket)-1
            bucket[idx][0] = min(bucket[idx][0], num)
            bucket[idx][1] = max(bucket[idx][1], num)
        
        last_max = float("inf")
        res = 0
        for cur_min, cur_max in bucket:
            if cur_min!=float("inf"):
                res = max(res, cur_min-last_max)
                last_max = cur_max
                
        return res
```



### 179. Largest Number

https://leetcode.com/problems/largest-number/

Ref: https://leetcode.com/problems/largest-number/discuss/53298/Python-different-solutions-(bubble-insertion-selection-merge-quick-sorts).

multiple sort here, all in-place



### 287. Find the Duplicate Number-similar to 136 Single number

https://leetcode.com/problems/find-the-duplicate-number/

***worth thinking and doing***

* solution-Floyd's Tortoise and Hare (Cycle Detection)

  > First assume when **fast** and **slow** meet, slow has moved **a** steps, and fast has moved **2a** steps. They meet in the circle, so the difference **a** must be a multiple of the length of the circle.
  > Next assume the distance between beginning to the entry point is **x**, then we know that the **slow** has traveled in the circle for **a-x** steps.
  > How do we find the entry point? Just let **slow** move for another **x** steps, then **slow** will have moved **a** steps in the circle, which is a multiple of length of the circle.
  > So we start another pointer at the beginning and let **slow** move with it. Remember **x** is the distance between beginning to the entry point, after **x**steps, both pointer will meet at the entry of circle.
  >
  > https://leetcode.com/problems/find-the-duplicate-number/discuss/72846/My-easy-understood-solution-with-O(n)-time-and-O(1)-space-without-modifying-the-array.-With-clear-explanation./268392



### 4. Median of Two Sorted Arrays

https://leetcode.com/problems/median-of-two-sorted-arrays/

* Solution-divide&conquer

https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2652/Share-one-divide-and-conquer-O(log(m%2Bn))-method-with-clear-description

***worth thinking and doing***

* Solution-binary search

***worth thinking and doing***



### 289. Game of Life

https://leetcode.com/problems/game-of-life/

***follow up-worth thinking and doing***





### 56. Merge Intervals

https://leetcode.com/problems/merge-intervals/

***worth thinking the use of sort***





### 57. Insert Interval

https://leetcode.com/problems/insert-interval/

转化为56

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        position = -1 # for line far below, as if we don't add anything in the first step, the position will start from 1 in thrid step
        res = []
        
        # s1: add all intervals whose start is less than the start of newInterval
        for i in range(len(intervals)):
            interval = intervals[i]
            if interval[0]<newInterval[0]:
                res.append(interval)
                position = i
            else:
                break
        # s2: add newInterval
        if len(res)!=0:
            last_added = res[-1]
            if newInterval[0] <= last_added[1]:
                last_added[1] = max(last_added[1], newInterval[1])
            else:
                res.append(newInterval)
        else:
            res.append(newInterval)
        
        for i in range(position+1, len(intervals)):
            interval = intervals[i]
            last_added = res[-1]
            if interval[0] <= last_added[1]:
                last_added[1] = max(last_added[1], interval[1])
            else:
                res.append(interval)
                
        return res
```







7.31

### 253. Meeting Rooms II

https://leetcode.com/problems/meeting-rooms-ii/



### 352. Data Stream as Disjoint Intervals

https://leetcode.com/problems/data-stream-as-disjoint-intervals/

* Similar questions

  [57.Insert Interval](https://leetcode.com/problems/insert-interval/)

* solution

  Worth try binary search



### 53. Maximum subarray

https://leetcode.com/problems/maximum-subarray/submissions/

* Solution-dynamic programming

  ***worth thinking***

  > Apparently, this is a optimization problem, which can be usually solved by DP. So when it comes to DP, the first thing for us to figure out is the format of the sub problem(or the state of each sub problem). The format of the sub problem can be helpful when we are trying to come up with the recursive relation.
  >
  > https://leetcode.com/problems/maximum-subarray/discuss/20193/DP-solution-and-some-thoughts

  > algorithm that operates on arrays: it starts at the left end (element A[1]) and scans through to the right end (element A[n]), keeping track of the maximum sum subvector seen so far. The maximum is initially A[0]. Suppose we've solved the problem for A[1 .. i - 1]; how can we extend that to A[1 .. i]?
  >
  > https://leetcode.com/problems/maximum-subarray/discuss/20211/Accepted-O(n
  >
  > ***超级像演绎推理，把问题一般化***

* Solution-divide and conquer

  ***worth thinking and doing***
  
  Ref: https://leetcode.com/problems/maximum-subarray/discuss/199163/Python-O(N)-Divide-and-Conquer-solution-with-explanations
  
  Time complexity is O(n), cuz divide and conquer calls will be made sum of 2^i for i in range(1, log base 2 of N).



### 239. Sliding Window Maximum

https://leetcode.com/problems/sliding-window-maximum/

#### Solution-pq

did@21.6.16

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        pq = []
        for i in range(k):
            pq.append((-nums[i],i))
        heapq.heapify(pq)
        res = [-pq[0][0]]
        
        for i in range(1, len(nums)-k+1):
            num = nums[i+k-1]
            heapq.heappush(pq, (-num, i+k-1))
            value, index = pq[0]
            while index<i:
                heapq.heappop(pq)
                value, index = pq[0]
            res.append(-pq[0][0])
            
        return res
```



#### Solution-deque-not done

***worth thinking and doing***

Ref: https://leetcode.com/problems/sliding-window-maximum/discuss/65884/Java-O(n)-solution-using-deque-with-explanation

Our current window always keeps the things in the range and keeps in a descending order.

did@21.6.25

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = collections.deque()
        
        res = []
        for i in range(k-1):
            while queue and queue[-1][0]<=nums[i]:
                queue.pop()
            queue.append((nums[i], i))
        
        for i in range(k-1, len(nums)):    
            if len(queue)!=0 and queue[0][1]<i-k+1:
                queue.popleft()
            while len(queue)!=0 and queue[-1][0]<=nums[i]:
                queue.pop()
            queue.append((nums[i], i))
            res.append(queue[0][0])
            
        return res
```



#### Solution- dynamic programming-not done



### 295. Find Median from Data Stream

https://leetcode.com/problems/find-median-from-data-stream/

* Solution-two priority queue

  ***worth thinking and doing***

  虽然heapq是最小堆，但换为负数就好了。

  善用两个heap



8.2

### 325. Maximum Size Subarray Sum Equals k

https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/

* solution

  > Hint2: Given *S[i]* a partial sum that starts at position *0* and ends at *i*, what can *S[i - k]* tell you ?
  >
  > Hint3: Use HashMap + prefix sum array.

  应该想到既然要使S[i - k]有意义的话，要如何设计dic，这里则是将常规的value用作key



### 209. Minimum Size Subarray Sum

https://leetcode.com/problems/minimum-size-subarray-sum/

* Solution-two pointer

  思考怎么精简我的做法

* solution-binary search

  not done



### 238. Product of Array Except Self

https://leetcode.com/problems/product-of-array-except-self/

* solution

  最开始的想法还是维持两个array，这样就可以从O(n^2)降到O(n)



### 228. Summary Ranges-easy

https://leetcode.com/problems/summary-ranges/





### 152. Maximum Product Subarray

https://leetcode.com/problems/maximum-product-subarray/

#### solution

***worth thinking and doing***

Ref: https://leetcode.com/problems/maximum-product-subarray/discuss/48230/Possibly-simplest-solution-with-O(n)-time-complexity

```java
int maxProduct(int A[], int n) {
    // store the result that is the max we have found so far
    int r = A[0];

    // imax/imin stores the max/min product of
    // subarray that ends with the current number A[i]
    for (int i = 1, imax = r, imin = r; i < n; i++) {
        // multiplied by a negative makes big number smaller, small number bigger
        // so we redefine the extremums by swapping them
        if (A[i] < 0)
            swap(imax, imin);

        // max/min product for the current number is either the current number itself
        // or the max/min by the previous number times the current one
        imax = max(A[i], imax * A[i]);
        imin = min(A[i], imin * A[i]);

        // the newly computed max value is a candidate for our global result
        r = max(r, imax);
    }
    return r;
}
```

Comment:

> 这道题妙就妙在它不仅仅依赖了一个状态（前一个数所能获得的最大乘积），而是两个状态（最大和最小乘积）。比较简单的dp问题可能就只是会建立一个`dp[]`，然后把最大值放到其中。但是这道题给我们打开了新的思路：我们的dp数组里面可以存更多的信息。而上面的解法之所以没有用dp数组的原因是`dp[i]`只依赖于`dp[i - 1]`因此没有必要把前面所有的信息都存起来，只需要存前一个`dp[i-1]`的最大和最小的乘积就可以了。下面的代码使用了自定义的内部类`Tuple`,从而可以同时存`imax`和`imin`,并将所有的`imax`和imin存到了dp数组中。虽然稍显复杂，但是有助于加深理解。



8.4

### 163. Missing Ranges

https://leetcode.com/problems/missing-ranges/

* solution

  对upper bound一个很常见的处理：

  ```python
  def findMissingRanges(self, A, lower, upper):
          result = []
          A.append(upper+1)
          pre = lower - 1
          for i in A:
             if (i == pre + 2):
                 result.append(str(i-1))
             elif (i > pre + 2):
                 result.append(str(pre + 1) + "->" + str(i -1))
             pre = i
          return result
  ```



### 88. Merge Sorted Array

https://leetcode.com/problems/merge-sorted-array/

#### Solution-2 pointer

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
       
        i = m-1
        j = n-1
        index = m+n-1
        
        while index>=0:
            if j<0 or i>=0 and nums1[i]>=nums2[j]:
                nums1[index] = nums1[i]
                i-=1
            else:
                nums1[index] = nums2[j]
                j-=1
            index-=1
```



但是这个人的判断条件写得更好, 我们最后只需要考虑剩下j的情况，因为nums1中前i个自然就在那里了

Ref: https://leetcode.com/problems/merge-sorted-array/discuss/29522/This-is-my-AC-code-may-help-you

```java
class Solution {
public:
    void merge(int A[], int m, int B[], int n) {
        int i=m-1;
		int j=n-1;
		int k = m+n-1;
		while(i >=0 && j>=0)
		{
			if(A[i] > B[j])
				A[k--] = A[i--];
			else
				A[k--] = B[j--];
		}
		while(j>=0)
			A[k--] = B[j--];
    }
};
```





### 75. Sort Colors

https://leetcode.com/problems/sort-colors/

#### solution-two pointers

\#Array Transformation\#常用思路

想这个思路时，可以先考虑只有两个数的时候再推导到三个数的

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = p = 0
        right = len(nums)-1
        
        while p<=right: # think of why equal here, cuz we haven't examined it yet
            if nums[p]==1:
                p+=1
            elif nums[p]==0:
                nums[p], nums[left] = nums[left], nums[p]
                left+=1
                p+=1 # cuz now, nums[p] must be 1 after exchange, so we can move forward
            else:
                nums[p],nums[right] = nums[right], nums[p]
                right-=1
```



#### Solution-worth thinking

本质是我们常用的递归思想，先假设一个小问题解决了，然后假如再来一个数该怎么操作。

Ref: https://leetcode.wang/leetCode-75-Sort-Colors.html, https://leetcode.com/problems/sort-colors/discuss/26500/Four-different-solutions

```java
// one pass in place solution
void sortColors(int A[], int n) {
    int n0 = -1, n1 = -1, n2 = -1;
    for (int i = 0; i < n; ++i) {
        if (A[i] == 0) 
        {
            A[++n2] = 2; A[++n1] = 1; A[++n0] = 0;
        }
        else if (A[i] == 1) 
        {
            A[++n2] = 2; A[++n1] = 1;
        }
        else if (A[i] == 2) 
        {
            A[++n2] = 2;
        }
    }
}
```



### 283. Move Zeroes

https://leetcode.com/problems/move-zeroes/

* Solution-two pinter

  \#Array Transformation\#

  和75.sort colors一起想



### 376. Wiggle Subsequence

https://leetcode.com/problems/wiggle-subsequence/

* Solution-greedy
* Solution-dynamic programming-not done



8.5

### 280. Wiggle Sort

https://leetcode.com/problems/wiggle-sort/

* Solution-还可以更精简

  思路参照一下，index为偶数和奇数分情况即可，ref: https://leetcode.com/problems/wiggle-sort/discuss/71692/Java-O(N)-solution

  ```java
  public void wiggleSort(int[] nums) {
      for (int i=1; i<nums.length; i++) {
          int a = nums[i-1];
          if ((i%2 == 1) == (a > nums[i])) {
              nums[i-1] = nums[i];
              nums[i] = a;
          }
      }
  }
  ```

  

### 324. Wiggle Sort II

https://leetcode.com/problems/wiggle-sort-ii/

* Solution-not done!!!





### 349. Intersection of Two Arrays

https://leetcode.com/problems/intersection-of-two-arrays/





### 1481. Least Number of Unique Integers after K Removals

https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/

#### Solution-:jeans:

Ref: https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/discuss/686343/Python-oror-3-Line-oror-Shortest-Simplest

```python
def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        c = Counter(arr)
        s = sorted(arr,key = lambda x:(c[x],x))
        return len(set(s[k:]))
```

