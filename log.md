

## binary search

@21.5.26

Enhancement

* how to set boarder



## Problems using stack

* lc42, key point is to decide the invariant of stack, and think of when to add and when to pop.





* For the problem that can be tackled recursively/iteratively, sometimes I made it verbose in one round or recursion, try to avoid this. Once I made it verbose in lc206 Reverse Linked List.

## Greedy

@21.5.31

Review greedy using Jeff's book

* storing files on tape
* Class scheduling

证明的话，反证法，演绎推理

* Assume an optimnal solution not gernerated by greedy algorithm
* find the first/one difference between the 2 solutions
* Argue that we can exchange the optimal choice without making the current solution worse

Problems: 

| 767  | Reorganize String                  |      | string | not yet |      |      | https://leetcode.com/problems/reorganize-string/             |
| ---- | ---------------------------------- | ---- | ------ | ------- | ---- | ---- | ------------------------------------------------------------ |
| 358  | Rearrange  String k Distance Apart |      | string | not yet |      |      | https://leetcode.com/problems/rearrange-string-k-distance-apart/ |



## Iterative&recursive

@21.6.2

iterative和recursive的转化(dp and recursive)

Problems:

| 77   | Combinations                                |      | backtrack | great problem,  transition between iterative and recursive |      |      | https://leetcode.com/problems/combinations/                  |
| ---- | ------------------------------------------- | ---- | --------- | ---------------------------------------------------------- | ---- | ---- | ------------------------------------------------------------ |
| 108  | Convert Sorted Array to Binary  Search Tree |      | tree      | Just try the  iterative one                                |      |      | https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/ |
| 114  | Flatten Binary Tree to Linked List          |      |           | $$ kind like  109&106                                      |      |      | https://leetcode.com/problems/flatten-binary-tree-to-linked-list/ |



## 2 pointers

Problems:

| 71   | Simplify Path                           |      | stack-pq |         |      |      | https://leetcode.com/problems/simplify-path/                 |
| ---- | --------------------------------------- | ---- | -------- | ------- | ---- | ---- | ------------------------------------------------------------ |
| 80   | Remove Duplicates from Sorted  Array II |      | array    | not yet |      |      | https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/ |



## tree

所以可能面对tree的题目，先可以有一些比较straightforward递归的形式，然后可以把它改装成普通的traversal, like 106,109,114



## Tolological sort&strong connectivity

在notability里面，tarjan在jeff那本书里



## sorting and selection
