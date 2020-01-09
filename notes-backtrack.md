想想数学里面的排列组合

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaq7kqsca6j32ji0u0kjn.jpg" alt="image-20200108224221734" style="zoom:13%;" />

1.7

### 78. Subsets-经典题

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
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> ans = new ArrayList<>();
    getAns(nums, 0, new ArrayList<>(), ans);
    return ans;
}

private void getAns(int[] nums, int start, ArrayList<Integer> temp, List<List<Integer>> ans) { 
    ans.add(new ArrayList<>(temp));
    for (int i = start; i < nums.length; i++) {
        temp.add(nums[i]);
        getAns(nums, i + 1, temp, ans);
        temp.remove(temp.size() - 1);
    }
}
```



#### solution-iterative

Ref: https://leetcode.com/problems/subsets/discuss/27278/C%2B%2B-RecursiveIterativeBit-Manipulation

> Using `[1, 2, 3]` as an example, the iterative process is like:
>
> 1. Initially, one empty subset `[[]]`
> 2. Adding `1` to `[]`: `[[], [1]]`;
> 3. Adding `2` to `[]` and `[1]`: `[[], [1], [2], [1, 2]]`;
> 4. Adding `3` to `[]`, `[1]`, `[2]` and `[1, 2]`: `[[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]`

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> subs = {{}};
        for (int num : nums) {
            int n = subs.size();
            for (int i = 0; i < n; i++) {
                subs.push_back(subs[i]); 
                subs.back().push_back(num);
            }
        }
        return subs;
    }
}; 
```



* Solution-bit manipulation

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





### 90. Subsets II

https://leetcode.com/problems/subsets-ii/description/

#### Solution-iterative, 在上一道基础上检查

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



#### Solution-recursive-worth





### 77. Combinations-经典

https://leetcode.com/problems/combinations/description/

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



#### Solution2-backtrack-improvement

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





#### Solution3-recursive

Ref: https://leetcode.wang/leetCode-77-Combinations.html

>  基于这个公式 C ( n, k ) = C ( n - 1, k - 1) + C ( n - 1, k ) 所用的思想





#### Solution4-dynamic programming

根据solution4来写





### 39. Combination Sum

https://leetcode.com/problems/combination-sum/description/

#### Solution-backtrack-worth 

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        self.helper(candidates, target, 0, [], res)
        return res
    
    def helper(self, candidates, remains, start, temp, res):
        if remains<0:
            return
        elif remains==0:
            res.append(temp)
        else:   
            for i in range(start, len(candidates)):
                self.helper(candidates, remains-candidates[i], i, temp+[candidates[i]], res)
```



#### Solution-backtrack-worth

还没写







### 40. Combination Sum II

https://leetcode.com/problems/combination-sum-ii/

#### Solution-backtrack

 在上一题基础上改一改

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()
        self.helper(res, candidates, 0, [], target)
        return res
        
    def helper(self, res, candidates, start, temp, remain):
        if remain<0:
            return
        elif remain==0:
            res.append(temp)
            return
        for i in range(start, len(candidates)):
            if i>start and candidates[i]==candidates[i-1]:
                continue
            # 加下面两行可以大幅提速，因为如果你当前值都大于remain，往后走肯定也不行，所以break
            if candidates[i]>remain:
                break
            self.helper(res, candidates, i+1, temp+[candidates[i]], remain-candidates[i])
```



### 216. Combination Sum III

https://leetcode.com/problems/combination-sum-iii/description/

#### Solution-backtrack

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        self.helper(k, n, res, 1, [])
        return res
        
        
    def helper(self, k, remain, res, start, temp):
        if k<0 or remain<0:
            return
        
        if k==0 and remain==0:
            res.append(temp)
        #k>0 and remain>0
        elif k>0:    
            for i in range(start,10):
                self.helper(k-1, remain-i, res, i+1, temp+[i])
```



### 377. Combination Sum IV

https://leetcode.com/problems/combination-sum-iv/description/

* Solution-dynamic programming

> 技术题目一般用dp来做