### 5345. Rank Teams by Votes

[My Submissions](https://leetcode.com/contest/weekly-contest-178/problems/rank-teams-by-votes/submissions/)[Back to Contest](https://leetcode.com/contest/weekly-contest-178/)

- **User Accepted:**512
- **User Tried:**727
- **Total Accepted:**514
- **Total Submissions:**871
- **Difficulty:****Medium**

In a special ranking system, each voter gives a rank from highest to lowest to all teams participated in the competition.

The ordering of teams is decided by who received the most position-one votes. If two or more teams tie in the first position, we consider the second position to resolve the conflict, if they tie again, we continue this process until the ties are resolved. If two or more teams are still tied after considering all positions, we rank them alphabetically based on their team letter.

Given an array of strings `votes` which is the votes of all voters in the ranking systems. Sort all teams according to the ranking system described above.

Return *a string of all teams* **sorted** by the ranking system.

```
Input: votes = ["ABC","ACB","ABC","ACB","ACB"]
Output: "ACB"
Explanation: Team A was ranked first place by 5 voters. No other team was voted as first place so team A is the first team.
Team B was ranked second by 2 voters and was ranked third by 3 voters.
Team C was ranked second by 3 voters and was ranked third by 2 voters.
As most of the voters ranked C second, team C is the second team and team B is the third.
```

```python
class Solution:
    def rankTeams(self, votes: List[str]) -> str:
        dicti = {}
        for vote in votes:
            for i, v in enumerate(vote):
                if v not in dicti:
                    dicti[v] = [0] * 26
                dicti[v][i] -= 1 # 用减法就避免了reverse
        return "".join(sorted(dicti, key = lambda x: [dicti[x], x]))# key的使用, 如果传入的是一个
```





### 1367. Linked List in Binary Tree

https://leetcode.com/problems/linked-list-in-binary-tree/

#### Solution-worth

Ref: https://leetcode.com/problems/linked-list-in-binary-tree/discuss/524852/Python-DFS

```python
class Solution:
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        res = []
        while head:
            res.append(str(head.val))
            head = head.next
        head = "".join(res)
    
        def dfs(root, path):
            if head in path:
                return True
            if not root:
                return False
            return dfs(root.left, path+str(root.val)) or dfs(root.right, path+str(root.val))
            
        return dfs(root, "")
```





### 1365. How Many Numbers Are Smaller Than the Current Number

[My Submissions](https://leetcode.com/contest/weekly-contest-178/problems/how-many-numbers-are-smaller-than-the-current-number/submissions/)[Back to Contest](https://leetcode.com/contest/weekly-contest-178/)

Given the array `nums`, for each `nums[i]` find out how many numbers in the array are smaller than it. That is, for each `nums[i]` you have to count the number of valid `j's` such that `j != i` **and** `nums[j] < nums[i]`.

Return the answer in an array.

```python
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        counter, cur = collections.Counter(nums), 0
        for c in sorted(counter):
            counter[c], cur = cur, cur + counter[c]
        return [counter[n] for n in nums]
```

