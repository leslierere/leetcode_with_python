想想数学里面的排列组合

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaq7kqsca6j32ji0u0kjn.jpg" alt="image-20200108224221734" style="zoom:13%;" />



### 78. Subsets-$bit solution

https://leetcode.com/problems/subsets/description/

#### solution-recursive

Ref: https://www.cnblogs.com/grandyang/p/4309345.html

而且还是dfs的preorder

> 下面来看递归的解法，相当于一种深度优先搜索，参见网友 [JustDoIt的博客](http://www.cnblogs.com/TenosDoIt/p/3451902.html)，由于原集合每一个数字只有两种状态，要么存在，要么不存在，那么在构造子集时就有选择和不选择两种情况，所以可以构造一棵二叉树，左子树表示选择该层处理的节点，右子树表示不选择，最终的叶节点就是所有子集合，树的结构如下：
>
> ```c++
>            						 []        
>                    /          \        
>                   /            \     
>                  /              \
>               [1]                []
>            /       \           /    \
>           /         \         /      \        
>        [1 2]       [1]       [2]     []
>       /     \     /   \     /   \    / \
>   [1 2 3] [1 2] [1 3] [1] [2 3] [2] [3] []
> ```
>
> 整个添加的顺序为：
>
> []
> [1]
> [1 2]
> [1 2 3]
> [1 3]
> [2]
> [2 3]
> [3]                 

```java
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.dfs(res, 0, nums, [])
        return res
        
        
        
    def dfs(self, res, i, nums, path):
        if i==len(nums):
            res.append(path)
            return
            
        self.dfs(res, i+1, nums, path+[nums[i]])
        self.dfs(res, i+1, nums, path)
```



#### solution-did@4.22, 相当于bfs了

Ref: https://leetcode.com/problems/subsets/discuss/27278/C%2B%2B-RecursiveIterativeBit-Manipulation

> Using `[1, 2, 3]` as an example, the iterative process is like:
>
> 1. Initially, one empty subset `[[]]`
> 2. Adding `1` to `[]`: `[[], [1]]`;
> 3. Adding `2` to `[]` and `[1]`: `[[], [1], [2], [1, 2]]`;
> 4. Adding `3` to `[]`, `[1]`, `[2]` and `[1, 2]`: `[[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]`

```c++
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        
        for num in nums:
            length = len(res)
            for i in range(length):
                copySub = res[i][::]
                res.append(copySub+[num])
                
        return res
```



#### Solution-bit manipulation-$

Ref: https://leetcode.wang/leetCode-78-Subsets.html

```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    int bit_nums = nums.length;
    int ans_nums = 1 << bit_nums; //执行 2 的 n 次方
    for (int i = 0; i < ans_nums; i++) {
        List<Integer> tmp = new ArrayList<>();
        int count = 0; //记录当前对应数组的哪一位
        int i_copy = i; //用来移位
        while (i_copy != 0) { 
            if ((i_copy & 1) == 1) { //判断当前位是否是 1
                tmp.add(nums[count]);
            }
            count++;
            i_copy = i_copy >> 1;//右移一位
        }
        ans.add(tmp);

    }
    return ans;
}
```





### 90. Subsets II-$

https://leetcode.com/problems/subsets-ii/description/

#### Solution-iterative, 在上一道基础上检查

> Ref: https://leetcode.com/problems/subsets-ii/discuss/30137/Simple-iterative-solution
>
> If we want to insert an element which is a dup, we can only insert it after the newly inserted elements from last step.

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        nums.sort()
        last = None
        lastSize = 1
        
        for i in nums:
            size = len(res)
            begin = 0
            if i==last:
                begin = lastSize
            lastSize = size
            for j in range(begin, size):
                cur = res[j]
                res.append(cur+[i])
            last = i
            
        return res
```



#### Solution-recursive-在上一道上改





### 77. Combinations-$$

https://leetcode.com/problems/combinations/description/

Thinking process: 

backtrack is easy to think of, which is a recursive solution, since this is like DFS, so of course we can have an according iterative solution as well. As there is a DFS solution, ofcourse we can have a BFS solution as well, which can be implemented iteratively.

Since this is in essence a math problem,  C ( n, k ) = C ( n - 1, k - 1) + C ( n - 1, k ) , and from this formula we can see that a problem can be solved by solving subproblems first, thus we can implement recursively or using dp.

#### Solution1-backtrack

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        self.dfs(1, [], res, n, k)
        return res
        
    def dfs(self, start, temp, res, n, k):
        if len(temp)==k:
            res.append(temp)
            return
        for i in range(start, n+1):
            self.dfs(i+1, temp+[i], res, n, k)
            # 开始我写的是, 但这样会导致我只是copy了一个pointer进去，最后temp变成[],得到的res里也全是[]
            # temp.append(i)
            # self.dfs(i+1, temp+[i], res, n, k)
            # temp.pop()
```



#### Solution2-backtrack-improvement-$

Ref: https://leetcode.com/problems/combinations/discuss/27002/Backtracking-Solution-Java/173730

> For anyone stumped by why this change is necessary, it's because you should not continue exploring (recursing) when you know that there won't be enough numbers left until `n` to fill the needed `k` slots. If n = 10, k = 5, and you're in the outermost level of recursion, you choose only i = 1...6 , because if you pick i=7 and go into `backTracking()` you only have 8,9,10 to pick from, so at most you will get [7,8,9,10]... but we need 5 elements!

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        self.dfs(1, [], res, n, k)
        return res
        
    def dfs(self, start, temp, res, n, k):
        if k==0:#here
            res.append(temp)
            return
        for i in range(start, n-k+2):#here
            self.dfs(i+1, temp+[i], res, n, k-1)#here
```



did@4.22, 这不好,

一看就有重复的结构，combine和backtrack里面都有同样的循环，所以其实只要backtrack里面有就好

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        res = []
        for i in range(1, n-k+2):
            self.backtrack(i, n, k, [i], res)
        return res
            
    def backtrack(self, i, n, k, path, res):
        if len(path)==k:
            res.append(path)
            return
        
        if len(path)+n-i+1<k:
            return
        # length is not enough
        
        for j in range(i+1, n+1):
            self.backtrack(j, n, k, path+[j], res)
            
```



#### Solution3-iterative based on backtrack



#### Solution4-the other iterative

Ref: https://leetcode.wang/leetCode-77-Combinations.html#%E8%A7%A3%E6%B3%95%E4%B8%89-%E8%BF%AD%E4%BB%A3%E6%B3%952

> 解法三 迭代法2

```java
public List<List<Integer>> combine(int n, int k) {
    if (n == 0 || k == 0 || k > n) return Collections.emptyList();
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    //个数为 1 的所有可能
    for (int i = 1; i <= n + 1 - k; i++) res.add(Arrays.asList(i));
    //第一层循环，从 2 到 k
    for (int i = 2; i <= k; i++) {
        List<List<Integer>> tmp = new ArrayList<List<Integer>>();
        //第二层循环，遍历之前所有的结果
        for (List<Integer> list : res) {
            //第三次循环，对每个结果进行扩展
            //从最后一个元素加 1 开始，然后不是到 n ，而是和解法一的优化一样
            //(k - (i - 1） 代表当前已经有的个数，最后再加 1 是因为取了 n
            for (int m = list.get(list.size() - 1) + 1; m <= n - (k - (i - 1)) + 1; m++) {
                List<Integer> newList = new ArrayList<Integer>(list);
                newList.add(m);
                tmp.add(newList);
            }
        }
        res = tmp;
    }
    return res;
}

```



#### Solution5-recursive

Ref: https://leetcode.wang/leetCode-77-Combinations.html

>  基于这个公式 C ( n, k ) = C ( n - 1, k - 1) + C ( n - 1, k ) 所用的思想

Did @21.6.2

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        return self.helper(1, n, k, dict())
        
    def helper(self, start, n, k, seen):
        if k==0:
            return [[]]
        elif n-start+1 == k:
            return [[i for i in range(start, n+1)]]
        elif (start, k) in seen:
            return seen[(start, k)]
        
        res = []
        res1 = self.helper(start+1, n, k-1, seen)
        for subres in res1:
            res.append([start]+subres)
        res2 = self.helper(start+1, n, k, seen)
        res.extend(res2)
        seen[(start, k)] = res
        
        return res
```





#### Solution4-dynamic programming

根据solution3来写, not done





### 39. Combination Sum

https://leetcode.com/problems/combination-sum/description/

#### Solution-backtrack

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()
        self.backtrack(candidates, 0, target, [], res)
        return res
        
        
    def backtrack(self, candidates, start, target, path, res):
        if target==0:
            res.append(path)
            return
        elif target<0 or candidates[start]>target: # candidates[start]>target 这个可以加快108ms->44ms
            return
        
        for i in range(start, len(candidates)):
            
            number = candidates[i]
            self.backtrack(candidates, i, target-candidates[i],path+[number], res)
```





### 40. Combination Sum II-看看comment

https://leetcode.com/problems/combination-sum-ii/

#### Solution-backtrack

 在上一题基础上改一改

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        self.backtrack(candidates, target, res, [], 0)
        return res
        
        
        
    def backtrack(self, candidates, target, res, path, start):
        
        if target==0:
            res.append(path)
            return
        elif target<0 or start>=len(candidates) or candidates[start]>target:# candidates[start]>target 这个很重要，避免不可能的继续操作，speed up!
            return
            
        for i in range(start, len(candidates)):
            if i>start and candidates[i]==candidates[i-1]:
                continue
            number = candidates[i]
            self.backtrack(candidates, target-number, res, path+[number], i+1)
```



### 216. Combination Sum III

https://leetcode.com/problems/combination-sum-iii/description/

#### Solution-backtrack

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        self.backtrack(k, n, 1, res, [])
        return res
        
    
    def backtrack(self, k, n, start, res, path):
        if k==0:
            if n==0:
                res.append(path)
            return
        elif start>n:
            return
        
        for i in range(start, 10):
            self.backtrack(k-1, n-i, i+1, res, path+[i])
```



### 377. Combination Sum IV-$

https://leetcode.com/problems/combination-sum-iv/description/

#### Solution-dynamic programming-worth



Ref: http://zxi.mytechroad.com/blog/dynamic-programming/leetcode-377-combination-sum-iv/

> 计数题目一般用dp来做

```c++
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1]+[0]*target # dp[i] represents for a target at i, the number of combinations
        
        for i in range(1, target+1):
            for j in nums:
                if i-j>=0:
                    dp[i] += dp[i-j]
                    
        return dp[target]
```



#### Solution-Recursion + Memorization

Ref: http://zxi.mytechroad.com/blog/dynamic-programming/leetcode-377-combination-sum-iv/

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        self.m = [1] + [-1] * target 
        # target为i时可能的结果，设为-1表示我还不知道为i时的结果方便后面判断
        # 根据stephan的意见改了一下
        return self.dp(nums, target)
    
    def dp(self, nums, target):
        if target<0:
            return 0
        if self.m[target]!=-1:
            return self.m[target]
        
        ans = 0
        for num in nums:
            ans += self.dp(nums, target-num)
        
        self.m[target] = ans
        return ans
```

did@4.23

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        nums.sort()
        dic = {}
        return self.backtrack(nums, target, dic)
        
    def backtrack(self, nums, target, dic):
        if target<0:
            return 0
        elif target==0:
            return 1
        elif target in dic:
            return dic[target]
        
        paths = 0
        for num in nums:
            if num>target:
                break
            paths+= self.backtrack(nums, target-num, dic)
            
        dic[target] = paths        
                
        return paths
```



#### Solution-followup

https://leetcode.com/problems/combination-sum-iv/discuss/85041/7-liner-in-Python-and-follow-up-question



### 254. Factor Combinations-$

https://leetcode.com/problems/factor-combinations/

#### Solution1-backtrack

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        res = []
        self.helper(res, 2, [], n)
        return res
        
        
    def helper(self, res, start, temp, remain):
        if remain==1 and len(temp)>1:
            res.append(temp)
        else:    
            for i in range(start, remain+1):
                if remain%i == 0:
                    self.helper(res, i, temp+[i], remain//i)
```



#### Solution2-backtrack-improvement-worth

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        res = []
        self.helper(res, 2, [], n)
        return res
        
        
    def helper(self, res, start, path, remain):
        if len(path)>0:
            res.append(path+[remain])
        for i in range(start, int(remain**0.5)+1):
            if remain%i == 0:
                self.helper(res, i, path+[i], remain//i)
```



@1.23 by myself

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        res = []
        self.helper(res, [], n, 2)
        return res
        
    # ensure we add the factors in the ascending order
    # this is through the variable start and the first if
    def helper(self, res, path, remain, start):
        if path:
            if remain<path[-1]:
                return
            else:
                res.append(path+[remain])
        
        for i in range(start, int(remain**0.5)+1):
            if remain%i==0:
                self.helper(res, path+[i], remain//i, i)
```





### 46. Permutations

https://leetcode.com/problems/permutations/description/

#### Solution-Recursive, take any number as first

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.helper(res, nums, [])
        return res
        
        
    def helper(self, res, nums, path):
        if not nums:
            res.append(path)
            
        for i in range(len(nums)):
            self.helper(res, nums[:i]+nums[i+1:], path+[nums[i]])
```



#### Solution-Recursive, insert the remaining first number

Ref: https://leetcode.wang/leetCode-46-Permutations.html



#### Solution-recursive, swap

还没想清楚, 不想想这个



### 47. Permutations II

https://leetcode.com/problems/permutations-ii/description/

#### Solution-backtrack

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        self.helper(res, nums, [])
        return res
        
    def helper(self, res, nums, path):
        if not nums:
            res.append(path)
            
        for i in range(len(nums)):
            if i>0 and nums[i]==nums[i-1]:
                continue
            self.helper(res, nums[:i]+nums[i+1:], path+[nums[i]])
```



### 31. Next Permutation-看comment

https://leetcode.com/problems/next-permutation/description/

#### Solution-iterative

my explanation: https://leetcode.com/problems/next-permutation/discuss/13867/C++-from-Wikipedia/948416

注意等号

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        first = len(nums)-2
        
        while first>=0 and nums[first]>=nums[first+1]:
            first-=1
            
        if first==-1:
            self.reverse(nums, 0)
            return
        
        larger = first+1 # the index of the number on the first right that is just larger than the one with first index
        for i in range(first+2, len(nums)):
            if nums[i]<=nums[first]:
                larger = i-1
                break
            if i == len(nums)-1:
                larger = i
        nums[first], nums[larger] = nums[larger], nums[first]
        self.reverse(nums, first+1)
        
    # reverse这个写法我觉得比较好            
    def reverse(self, lis, start):
        end = len(lis)-1
        while start<end:
            lis[start], lis[end] = lis[end], lis[start]
            start+=1
            end-=1
```







### 60. Permutation Sequence-$c comments

https://leetcode.com/problems/permutation-sequence/description/

#### Solution-iterative

did@21.5.29

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # every position can have n diff digits
        permutations = 1
        for i in range(1, n+1):
            permutations*=i
        
        numbers = [i for i in range(1, n+1)]
        res = ""
        while len(numbers)!=1:
            group_size = permutations//len(numbers) 
            group_no = (k-1)//group_size # start from index 0
            k = (k-1)%group_size+1
            number = numbers.pop(group_no)
            res+=str(number)
            permutations = group_size
            
        res+=str(numbers[0])
        
        return res
```



#### Solution-recursive,不难，主要是细节, 不用一遍遍地算groupsize，每次除以一下就好了

```python
import math
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        groupSize = math.factorial(n)
        nums = list(range(1,n+1))
        return self.helper("", nums, k, groupSize)
        
        
    def helper(self, path, nums, k, groupSize):
        if len(nums) == 0:
            return path
        groupSize = groupSize//len(nums)
        initial = nums.pop((k-1)//groupSize)
        nextK = groupSize if k%groupSize==0 else k%groupSize
        return self.helper(path+str(initial), nums, nextK,groupSize)
```





### 291. Word Pattern II-$

https://leetcode.com/problems/word-pattern-ii/description/

#### Solution-backtraking-worth

Ref: [https://leetcode.com/problems/word-pattern-ii/discuss/73675/*Java*-HashSet-%2B-backtracking-(2ms-beats-100)](https://leetcode.com/problems/word-pattern-ii/discuss/73675/*Java*-HashSet-%2B-backtracking-(2ms-beats-100))

```python
class Solution:
    def wordPatternMatch(self, pattern: str, string: str) -> bool:
        words = set()
        dic = {}
        return self.helper(0, 0, pattern, string, words, dic)
        
    def helper(self, p1, p2, pattern, string, words, dic):
        if p1 == len(pattern) and p2==len(string):
            return True
        if (p1 == len(pattern) and p2<len(string)) or (p1 < len(pattern) and p2==len(string)):
            return False
        char = pattern[p1]
        if char in dic:
            word = dic[char]
            return word == string[p2:p2+len(word)] and self.helper(p1+1, p2+len(word), pattern, string, words, dic)
            
        else:
            for i in range(p2, len(string)):
                word = string[p2:i+1]
                if word in words: # different pattern cannot map to same substring
                    continue
                dic[char] = word
                words.add(word)
                
                if self.helper(p1+1, i+1, pattern, string, words, dic):
                    return True
                else:
                    dic.pop(char)
                    words.remove(word)
                    
        return False
```



### 17. Letter Combinations of a Phone Number-c comments

https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/

#### Solution-backtrack-常规

相当于dfs

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        
        res = []
        if not digits:
            return res
        dic = {'2':list('abc'), '3':list('def'),'4':list('ghi'),'5':list('jkl'),'6':list('mno'), '7':list('pqrs'), '8':'tuv', '9':'wxyz'}
        self.helper(0, digits, res, dic, "")
        return res
        
    def helper(self, i, digits, res, dic, path):
        if i==len(digits):
            res.append(path)
            return
        for char in dic[digits[i]]: # you don't need write index here, which will slow down
            self.helper(i+1, digits, res, dic, path+char)
```

#### Solution-循环

相当于bfs

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits)==0:
            return []
        
        mapping = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
        res = [""]
            
        for digit in digits:
            new_res = []
            for element in res:
                for letter in mapping[digit]:
                    new_res.append(element+letter)
                    
            res = new_res
            
        return res
```





### 320. Generalized Abbreviation

#### Solution-dfs

Ref: https://leetcode.com/problems/generalized-abbreviation/discuss/77190/Java-backtracking-solution

```java
public List<String> generateAbbreviations(String word){
        List<String> ret = new ArrayList<String>();
        backtrack(ret, word, 0, "", 0);

        return ret;
    }

    private void backtrack(List<String> ret, String word, int pos, String cur, int count){
        if(pos==word.length()){
            if(count > 0) cur += count;
            ret.add(cur);
        }
        else{
            backtrack(ret, word, pos + 1, cur, count + 1);
            backtrack(ret, word, pos+1, cur + (count>0 ? count : "") + word.charAt(pos), 0);
        }
    }
```

did@5.2

```python
class Solution:
    def generateAbbreviations(self, word: str) -> List[str]:
        res = []
        self.helper(word, 0, "", res)
        return res
        
    def helper(self, word, number, path, res):
        if not word:
            if number!=0:
                res.append(path+str(number))
            else:
                res.append(path)
            return
        
        
        
        self.helper(word[1:], number+1, path, res)
        if number==0:
            self.helper(word[1:], 0, path+word[0], res)
        else:
            self.helper(word[1:], 0, path+str(number)+word[0], res)
```



#### Solution-bit manipulation-worth

[Reference](https://leetcode.com/problems/generalized-abbreviation/discuss/77209/O(m*n)-bit-manipulation-java)

```java
public List<String> generateAbbreviations(String word) {
    List<String> ret = new ArrayList<>();
    int n = word.length();
    for(int mask = 0;mask < (1 << n);mask++) {//二进制数
        int count = 0;
        StringBuffer sb = new StringBuffer();
      //循环word里面每一个单词，加上去
        for(int i = 0;i <= n;i++) {
            if(((1 << i) & mask) > 0) {//当前又要变数字
                count++;
            } else {
                if(count != 0) {
                    sb.append(count);
                    count = 0;
                }
                if(i < n) sb.append(word.charAt(i));
            }
        }
        ret.add(sb.toString());
    }
    return ret;
}
```





### 282. Expression Add Operators-$

https://leetcode.com/problems/expression-add-operators/description/

#### Solution-backtrack-worth

[Reference](https://leetcode.com/problems/expression-add-operators/discuss/71895/Java-Standard-Backtrace-AC-Solutoin-short-and-clear)

```java
public class Solution {
    public List<String> addOperators(String num, int target) {
        List<String> rst = new ArrayList<String>();
        if(num == null || num.length() == 0) return rst;
        helper(rst, "", num, target, 0, 0, 0);
        return rst;
    }
    public void helper(List<String> rst, String path, String num, int target, int pos, long eval, long multed){//eval is the current evaluation
        if(pos == num.length()){
            if(target == eval)
                rst.add(path);
            return;
        }
        for(int i = pos; i < num.length(); i++){
            if(i != pos && num.charAt(pos) == '0') break;//deal with 0 sequence
            long cur = Long.parseLong(num.substring(pos, i + 1));
            if(pos == 0){
                helper(rst, path + cur, num, target, i + 1, cur, cur);
            }
            else{
                helper(rst, path + "+" + cur, num, target, i + 1, eval + cur , cur);
                
                helper(rst, path + "-" + cur, num, target, i + 1, eval -cur, -cur);
                
                helper(rst, path + "*" + cur, num, target, i + 1, eval - multed + multed * cur, multed * cur );
            }
        }
    }
}
```



did@5.3

```python
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        res = []
        for i in range(len(num)):
            if i!=0 and num[0]=='0':
                break
            self.helper(num[i+1:], target, num[:i+1], 0, int(num[:i+1]), res)
            
        return res
        
        
        
    def helper(self, num, target, path, temp, last, res):
        if not num:
            if temp+last==target:
                res.append(path)
            return
        
        for i in range(len(num)):
            if i!=0 and num[0]=='0':
                break
            number = num[:i+1]
            self.helper(num[i+1:], target, path+"*"+number, temp, last*int(number), res)
            self.helper(num[i+1:], target, path+"+"+number, temp+last, int(number), res)
            self.helper(num[i+1:], target, path+"-"+number, temp+last, -int(number), res)
```



### 140. Word Break II-$$

https://leetcode.com/problems/word-break-ii/description/

#### Solution1-dfs+memory

想清楚为啥要根据start momorize, 因为切的时候，就是到后面有的地方就是之前已经切过了

Ref: https://leetcode.com/problems/word-break-ii/discuss/44169/9-lines-Python-10-lines-C%2B%2B

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        if not wordDict:
            return []
        # res = []
        wordSet = set(wordDict)
        temp ={}
        
        return self.helper(0, s, wordSet, temp)
        
            
        
    def helper(self, start, s, wordSet, temp):
        
        if start==len(s):
            return ['']
        elif start in temp:
            return temp[start]
        
        subRes = []
        for i in range(start, len(s)):
            word = s[start:i+1]
            if word in wordSet:
                postRes = self.helper(i+1, s, wordSet, temp)      
                if postRes:
                    for post in postRes:
                        subRes.append((word+' '+post).strip())
        
        temp[start] = subRes
        return temp[start]
```





#### Solution2

Ref: https://www.youtube.com/watch?v=JqOIRBC0_9c

![image-20200110131829109](https://tva1.sinaimg.cn/large/006tNbRwgy1gas2if0ca2j31c00u0npd.jpg)

> 找分割点，右边子串必须在字典里，再对左边子串递归求解。
>
> 每一个wordbreak返回来，我们都记忆一下，比如这里catsand->['cat sand', 'cats and'], 下次就可以直接返回





### 351. Android Unlock Patterns-$$

https://leetcode.com/problems/android-unlock-patterns/

#### Solution1

Ref: https://www.cnblogs.com/grandyang/p/5541012.html

像9到2是可以的

> 那么我们先来看一下哪些是非法的，首先1不能直接到3，必须经过2，同理的有4到6，7到9，1到7，2到8，3到9，还有就是对角线必须经过5，例如1到9，3到7等。我们建立一个二维数组jumps，用来记录两个数字键之间是否有中间键，然后再用一个一位数组visited来记录某个键是否被访问过，然后我们用递归来解，我们先对1调用递归函数，在递归函数中，我们遍历1到9每个数字next，然后找他们之间是否有jump数字，如果next没被访问过，并且jump为0，或者jump被访问过，我们对next调用递归函数。数字1的模式个数算出来后，由于1,3,7,9是对称的，所以我们乘4即可，然后再对数字2调用递归函数，2,4,6,9也是对称的，再乘4，最后单独对5调用一次，然后把所有的加起来就是最终结果了
>
> ```c++
> class Solution {
> public:
>     int numberOfPatterns(int m, int n) {
>         int res = 0;
>         vector<bool> visited(10, false);
>         vector<vector<int>> jumps(10, vector<int>(10, 0));
>         jumps[1][3] = jumps[3][1] = 2;
>         jumps[4][6] = jumps[6][4] = 5;
>         jumps[7][9] = jumps[9][7] = 8;
>         jumps[1][7] = jumps[7][1] = 4;
>         jumps[2][8] = jumps[8][2] = 5;
>         jumps[3][9] = jumps[9][3] = 6;
>         jumps[1][9] = jumps[9][1] = jumps[3][7] = jumps[7][3] = 5;
>         res += helper(1, 1, 0, m, n, jumps, visited) * 4;
>         res += helper(2, 1, 0, m, n, jumps, visited) * 4;
>         res += helper(5, 1, 0, m, n, jumps, visited);
>         return res;
>     }
>     int helper(int num, int len, int res, int m, int n, vector<vector<int>> &jumps, vector<bool> &visited) {
>         if (len >= m) ++res;
>         ++len;
>         if (len > n) return res;
>         visited[num] = true;
>         for (int next = 1; next <= 9; ++next) {
>             int jump = jumps[num][next];
>             if (!visited[next] && (jump == 0 || visited[jump])) {
>                 res = helper(next, len, res, m, n, jumps, visited);
>             }
>         }
>         visited[num] = false;
>         return res;
>     }
> };
> ```





### 93. Restore IP Addresses

https://leetcode.com/problems/restore-ip-addresses/



