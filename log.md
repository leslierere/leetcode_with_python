

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





## k sum&stocks

理解了ksum一切都好说，当然第一道题有点不一样，因为要返回index就不能sort而且题目说了不会有重复解。但不太理解为什么不会miss的问题。





## Process

* First 200: 21.5.16-21.6.21
* Review unfamiliar questions in first 200 by catrgoris, every time think about the question at most 15 min. Speak out when coding.
* Learning sort, greedy
* Just by categories





## Sort

### Comparison-based sorting algorithm

#### quick sort

Though also divide and conquer, but unlike merge sort, it does all the work in the beginning stage of divide, through choosing a pivot and dividing the array into different parts.

There are a few things we need take care of:

* how to choose the pivot

  If just choosing the first one, we risk the N-square time complexity if it is sorted already. In this regard, it is awkward for linked list, if choose a random one in the 1/4 to 3/4 part of the array, we must walk half the array on average to get the pivot.

* what about values equal to pivot

  For linked list, we partition the list to 3 parts, left part less than pivot, middle part equal to pivot, right part larger than pivot.

  For array, we stop and swap as it is crucial to avoiding quadratic running time in certain typical applications. 

  Think of an example, refer to princeton book

  > Suppose that we scan over items with keys equal to the partitioning item’s key instead of stopping the scans when we encounter them. Show that the running time of this version of quicksort is quadratic for all arrays with just a constant number of distinct keys.



#### quick select

partition list into 3 lists, it doesn't matter that the value equal to pivot go to which list.

O(N) average time if we select pivot randomly.



For every comparison-based sorting algorithm, it takes Omega(n log n) worst-case time.



### Linear-time sorting

#### Bucket sort

Normally, bucket sort is useful when keys are in a small range (say from 0 to **q**-1) while the number of items n is larger. But in essence, it is we have a bucket array **B**, based on key **k**, we put value to bucket **B[k]**, which itself is also a sequence.

Theta of q to initialize bucket, theta of n to put items into buckets.

If q is in O(n), then theta of n time in total.

#### Radix sort

TODO



### Other sorting

#### Insertion sort

For linked list, theta of N time to find the right position to insert;

For array, theta of N time to find the right position and shit numbers over.

O(N^2)



#### Selection sort

Always find current smallest number and append to the partially sorted list.

Theta of N-square time.



#### Heap sort

Selection sort in which we enforce the heap-order property first(Takes O(N) time). When removing the min, each takes O(log N) time. Thus overall, it takes O(N log N) time.



#### Merge sort

O(N) time per level, O(N logN) time in total.
