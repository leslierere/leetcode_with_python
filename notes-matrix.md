### 48. Rotate Image

#### Solution-upside down flip+symmatry flip

Ref: https://leetcode.com/problems/rotate-image/discuss/18872/A-common-method-to-rotate-the-image

@可以利用它本身有的reverse先上下翻，而且线上下翻再对角线翻坐标更容易写



#### Solution-$

Ref: https://leetcode.com/problems/rotate-image/discuss/18884/Seven-Short-Solutions-(1-to-7-lines)

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n//2):
            for j in range(i, n-i-1): #constraining the left and right boundaries after each outer loop
                matrix[i][j], matrix[j][~i], matrix[~i][~j], matrix[~j][i] = \
                matrix[~j][i], matrix[i][j], matrix[j][~i], matrix[~i][~j]
```



#### Solution-using zip

Ref: https://leetcode.com/problems/rotate-image/discuss/18884/Seven-Short-Solutions-(1-to-7-lines)

```python
class Solution:
    def rotate(self, A):
        A[:] = zip(*A[::-1])
```





### 54. Spiral Matrix

https://leetcode.com/problems/spiral-matrix/description/

#### Solution

就按照提示来做

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        if len(matrix)==1:
            return matrix[0]
        
        right = len(matrix[0])
        left = -1
        bottom = len(matrix)
        top = 0
        number = right*bottom
        
        i,j = 0,0
        res = []
        
        while True:
            while j<right:
                res.append(matrix[i][j])
                j+=1
            if len(res)==number:
                return res
            
            right-=1
            j-=1
            i+=1
            
            while i<bottom:
                res.append(matrix[i][j])
                i+=1
            if len(res)==number:
                return res
            bottom -=1
            i-=1
            j-=1
            
            while j>left:
                res.append(matrix[i][j])
                j-=1
            if len(res)==number:
                return res
            left+=1
            i-=1
            j+=1
            
            while i>top:
                res.append(matrix[i][j])
                i-=1
            if len(res)==number:
                return res
            top+=1
            i+=1
            j+=1
            
        return res
```





#### Solution-$

我惊呆了

Ref: https://leetcode.com/problems/spiral-matrix/discuss/20571/1-liner-in-Python-%2B-Ruby





### 59. Spiral Matrix II

https://leetcode.com/problems/spiral-matrix-ii/

#### Solution

just like 54, Solution1





### 73. Set Matrix Zeroes-$

https://leetcode.com/problems/set-matrix-zeroes/description/

#### Solution

can be tricky 

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        col0 = False
        
        for i in range(len(matrix)):
            if matrix[i][0] == 0:
                col0 = True
            for j in range(1, len(matrix[0])):
                if matrix[i][j]==0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0 # mark row
                    
                    
        for i in range(len(matrix)-1, -1, -1):
            # we need bottom up here, or the first row used for mark would be destroyed
            for j in range(1, len(matrix[0])):
                if matrix[i][0]==0 or matrix[0][j]==0:
                    matrix[i][j]=0
            if col0:
                matrix[i][0] = 0
```





### 311. Sparse Matrix Multiplication-$

https://leetcode.com/problems/sparse-matrix-multiplication/description/

#### Solution-worth

Ref: https://www.cnblogs.com/grandyang/p/5282959.html

i * k 的矩阵A乘以k * j 的矩阵B会得到一个i * j 的矩阵C, 对于矩阵C中的每个元素C\[i][j]的计算，A\[i][0]\*B\[0][j] + A\[i][1]\*B\[1][j] + ... + A\[i][k]*B\[k][j]

Ref: https://leetcode.com/problems/sparse-matrix-multiplication/discuss/76154/Easiest-JAVA-solution

> Let's look at brute force solution:
>
> ```java
> public int[][] multiply_bruteForce(int[][] A, int[][] B) {
> 	int m = A.length, n = A[0].length;
> 	int nB = B[0].length;
> 	int [][] C = new int[m][nB];
> 	for (int i = 0; i<m; i++) {
> 		for (int j = 0; j<nB; j++){
> 			C[i][j] = 0;
> 			for( int k = 0; k<n; k++)
> 				C[i][j] += A[i][k]*B[k][j];
> 		}
> 	}
> 	return C;
> }
> ```
>
> For brute force solution, for each C[ i ] [ j ], it uses C[ i ] [ j ] += A[ i ] [ k ] * B[ k ] [ j ] where k = [ 0, n].Note: even A[ i ] [ k ] or B[ k ] [ j ] is 0, the multiplication is still executed.
>
> 
>
> For the above smart solution, if A[ i ] [ k ] == 0 or B[ k ] [ j ] == 0, it just skip the multiplication . This is achieved by moving for-loop" for ( k = 0; k < n; k++ ) " from inner-most loop to middle loop, so that we can use if-statement to tell whether A[ i ] [ k ] == 0 or B[ k ] [ j ] == 0. 





### 329. Longest Increasing Path in a Matrix

https://leetcode.com/problems/longest-increasing-path-in-a-matrix/

#### Solution-dfs, memorization

其实挺常规的

Ref: https://leetcode.com/problems/longest-increasing-path-in-a-matrix/discuss/78334/Python-solution-memoization-dp-288ms





### 378. Kth Smallest Element in a Sorted Matrix-$

#### Solution-heap-worth

Ref: https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/301357/Java-0ms-(added-Python-and-C%2B%2B)%3A-Easy-to-understand-solutions-using-Heap-and-Binary-Search

```python
from heapq import *

def find_Kth_smallest(matrix, k):
    minHeap = []

    # put the 1st element of each row in the min heap
    # we don't need to push more than 'k' elements in the heap
    for i in range(min(k, len(matrix))):
        heappush(minHeap, (matrix[i][0], 0, matrix[i]))
        # the 0 in the middle represents the index in the row

    # take the smallest(top) element form the min heap, if the running count is equal to k' return the number
    # if the row of the top element has more elements, add the next element to the heap
    numberCount, number = 0, 0
    while minHeap:
        number, i, row = heappop(minHeap)
        numberCount += 1
        if numberCount == k:
            break
        if len(row) > i+1:
            heappush(minHeap, (row[i+1], i+1, row))
    return number
```





#### Solution-binary search-worth

Ref: https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/301357/Java-0ms-(added-Python-and-C%2B%2B)%3A-Easy-to-understand-solutions-using-Heap-and-Binary-Search

>  We cannot get a middle index here, an alternate could be to apply the **Binary Search** on the “number range” instead of the “index range”. As we know that the smallest number of our matrix is at the top left corner and the biggest number is at the bottom lower corner. These two number can represent the “range” i.e., the `start` and the `end` for the **Binary Search**. The middle would be the middle value of the start number and the end number, this `middle` number is NOT necessarily an element in the matrix.

可下面这种我总觉得会弄出一个不存在的数字。。。可能就是一定要刚好等于matrix中的那个数，才满足有多少个小于等于他

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        """
        Use binary search solution.
        """
        if not matrix or not matrix[0]:
            return None

        start, end = matrix[0][0], matrix[-1][-1]
        while start < end:
            m = start + ((end - start) >> 1)
            total = 0
            for row in matrix:
                # bisect_right is to find how many numbers in the current
                # row that is <= m.
                total += bisect_right(row, m)

            if total < k:
                # This means m is too small and we need to find our
                # target value in [m + 1, end]
                start = m + 1
            else:
                # This means m is big enough for searching the target value.
                # In other words, the target value must exist in [start, m].
                # If total == k, we cannot say if m is our target value as
                # it could be possible that m does not exist in the matrix.
                # But again, our target value must exist in [start, m], so
                # we set end to m again to narrow the search in the next round.
                end = m

        # In the end start must be the same as the end, which means we have
        # found the target value.
        return start
```





### 74. Search a 2D Matrix

https://leetcode.com/problems/search-a-2d-matrix/description/

#### Solution-binary search

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        rows = len(matrix)
        cols = len(matrix[0])
        
        
        order1 = 0
        order2 = rows*cols-1

        while order1<order2:
            midOrder = (order1+order2)//2
            midi, midj = self.getCodi(midOrder, cols)
            if matrix[midi][midj]<target:
                order1 = midOrder+1
            elif matrix[midi][midj]==target:
                return True
            else:
                order2 = midOrder-1
            
            
        if order1==order2:
            i, j = self.getCodi(order1, cols)
            return matrix[i][j]==target
        return False
            
        
    def getOrder(self, i , j, cols):
        return i*cols+j
    
    def getCodi(self, order, cols):
        i = order//cols
        j = order%cols
        
        return i,j
```



### 240. Search a 2D Matrix II-$solution2

https://leetcode.com/problems/search-a-2d-matrix-ii/description/

#### Solution-divide and conquer

Ref: https://leetcode.com/articles/search-a-2d-matrix-ii/

每一个submatrix，最小值都在左上角，最大值都在右下角, 还是比较麻烦。。。



#### Solution-Search Space Reduction-worth





### 370. Range Addition-$

https://leetcode.com/problems/range-addition/description/

#### Solution-太聪明了

Ref: [https://leetcode.com/problems/range-addition/discuss/84225/Detailed-explanation-if-you-don't-understand-especially-%22put-negative-inc-at-endIndex%2B1%22](https://leetcode.com/problems/range-addition/discuss/84225/Detailed-explanation-if-you-don't-understand-especially-%22put-negative-inc-at-endIndex%2B1%22)

Further reading: https://leetcode.com/articles/a-recursive-approach-to-segment-trees-range-sum-queries-lazy-propagation/



### 79. Word Search

https://leetcode.com/problems/word-search/

#### Solution-backtrack-常规，看comment

Ref: https://leetcode.com/articles/word-search/

> We argue that a more accurate term to summarize the solution would be ***backtracking***, which is a methodology where we mark the current path of exploration, if the path does not lead to a solution, we then revert the change (*i.e.* backtracking) and try another path.
>
> ....
>
> Instead of returning directly once we find a match, we simply *break* out of the loop and do the cleanup before returning.

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.helper(i, j, word, 0, board):
                    return True
                
                
    def helper(self, x, y, word, index, board):
        if index==len(word):
            return True
        if x<0 or x>=len(board) or y<0 or y>=len(board[0]):
            return False
        cur = board[x][y]
        
        if cur!= word[index]:
            return False
        board[x][y] = "#"
        
        res = False
        if self.helper(x+1, y, word, index+1, board) or self.helper(x-1, y, word, index+1, board) or self.helper(x, y+1, word, index+1, board) or self.helper(x, y-1, word, index+1, board):
            res = True #开始我直接在这里就return了，这样会造成side effect
            
        board[x][y] = cur
        return res
```





### 296. Best Meeting Point-$

https://leetcode.com/problems/best-meeting-point/description/

https://leetcode.com/articles/best-meeting-point/

> As long as there is equal number of points to the left and right of the meeting point, the total distance is minimized.

#### Solution-sorting-worth

```java
public int minTotalDistance(int[][] grid) {
    List<Integer> rows = new ArrayList<>();
    List<Integer> cols = new ArrayList<>();
    for (int row = 0; row < grid.length; row++) {
        for (int col = 0; col < grid[0].length; col++) {
            if (grid[row][col] == 1) {
                rows.add(row);
                cols.add(col);
            }
        }
    }
    int row = rows.get(rows.size() / 2);
    Collections.sort(cols);
    int col = cols.get(cols.size() / 2);
    return minDistance1D(rows, row) + minDistance1D(cols, col);
}

private int minDistance1D(List<Integer> points, int origin) {
    int distance = 0;
    for (int point : points) {
        distance += Math.abs(point - origin);
    }
    return distance;
}
```





#### Solution-two pointers to calculate the distance without knowing the median-worth

本质上是一样的，只是算distance时不同

```java
public int minTotalDistance(int[][] grid) {
    List<Integer> rows = collectRows(grid);
    List<Integer> cols = collectCols(grid);
    return minDistance1D(rows) + minDistance1D(cols);
}

private int minDistance1D(List<Integer> points) {
    int distance = 0;
    int i = 0;
    int j = points.size() - 1;
    while (i < j) {
        distance += points.get(j) - points.get(i);
        i++;
        j--;
    }
    return distance;
}

private List<Integer> collectRows(int[][] grid) {
    List<Integer> rows = new ArrayList<>();
    for (int row = 0; row < grid.length; row++) {
        for (int col = 0; col < grid[0].length; col++) {
            if (grid[row][col] == 1) {
                rows.add(row);
            }
        }
    }
    return rows;
}

private List<Integer> collectCols(int[][] grid) {
    List<Integer> cols = new ArrayList<>();
    for (int col = 0; col < grid[0].length; col++) {
        for (int row = 0; row < grid.length; row++) {
            if (grid[row][col] == 1) {
                cols.add(col);
            }
        }
    }
    return cols;
}
```





### 361. Bomb Enemy-$

https://leetcode.com/problems/bomb-enemy/description/

#### Solution-dynamic programming

Ref: https://leetcode.com/problems/bomb-enemy/discuss/83387/Short-O(mn)-time-O(n)-space-solution

https://www.cnblogs.com/grandyang/p/5599289.html





### 317. Shortest Distance from All Buildings-$

https://leetcode.com/problems/shortest-distance-from-all-buildings/description/

#### Solution-bfs

Ref: https://leetcode.com/problems/shortest-distance-from-all-buildings/discuss/76880/36-ms-C%2B%2B-solution

bymyself

```python
from collections import deque
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        sumCount = [[0]*cols for x in range(len(grid))]
        
        level = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    queue = deque()
                    queue.append((i,j))
                    step = 1
                    
                    while queue:
                        length = len(queue)
                        
                        for _ in range(length):
                            x, y = queue.popleft()
                            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                                if x+dx>=0 and x+dx<rows and y+dy>=0 and y+dy<cols and grid[x+dx][y+dy] == level:
                                    grid[x+dx][y+dy]-= 1
                                    sumCount[x+dx][y+dy] += step
                                    queue.append((x+dx, y+dy))
                        step+=1
                    level-=1
        
        minStep = float("inf")
        for i in range(rows):
            for j in range(cols):
                if grid[i][j]==level and sumCount[i][j]>0:
                    minStep = min(minStep, sumCount[i][j])
        if minStep == float("inf"):
            return -1
        return minStep
```





### 302. Smallest Rectangle Enclosing Black Pixels

https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/description/

#### Solution-dfs-slow

#### Solution-binary search，就是brute force加快的感觉

Ref: https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/discuss/75127/C%2B%2BJavaPython-Binary-Search-solution-with-explanation





### 36. Valid Sudoku-$想一下，我第二次做复杂了

https://leetcode.com/problems/valid-sudoku/description/

#### Solution-hash table

Ref: https://leetcode.com/problems/valid-sudoku/discuss/15472/Short%2BSimple-Java-using-Strings

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        exist = set()
        
        for i in range(0, 9):
            for j in range(0, 9):
                if board[i][j]!=".":
                    rowStr = str(i)+"("+board[i][j]+")"
                    colStr = "("+board[i][j]+")"+str(j)
                    blockStr = str(i//3) + "("+board[i][j]+")"+str(j//3)
                    if rowStr in exist or colStr in exist or blockStr in exist:
                        return False
                    exist.add(rowStr)
                    exist.add(colStr)
                    exist.add(blockStr)
                    
                    
        return True
```



### 37. Sudoku Solver-$

https://leetcode.com/problems/sudoku-solver/

#### Solution-backtrack, did@4.22

Ref: https://leetcode.com/problems/sudoku-solver/discuss/15752/Straight-Forward-Java-Solution-Using-Backtracking



36ms by others, heapq这个想法特别好，就从剩余最少的地方开始看，set的加减法

```python
import heapq

class Solution:
    # Does a DFS using a priority queue
    def solve(self, board, heap):
        if len(heap)==0:
            return True
        size1, idy1, idx1, remaining1 = heap[0]
        if size1 == 0:
            #No options, so must be invalid
            return False
        for value in remaining1:
            board[idy1][idx1] = value
            newheap = []
            for size, idy, idx, remaining in heap[1:]:
                matchedrow = idy1==idy
                matchedcol = idx1==idx
                matchedbox = (idy1//3*3+idx1//3)==(idy//3*3+idx//3)
                if (matchedrow or matchedcol or matchedbox) and value in remaining:
                    heapq.heappush(newheap,(size-1,idy, idx, remaining - {value}))
                else:
                    heapq.heappush(newheap, (size, idy, idx, remaining))
            if self.solve(board, newheap):
                return True
        return False
    
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        allnums = {'1','2','3','4','5','6','7','8','9'}
        rows = [set() for i in range(9)]
        cols = [set() for i in range(9)]
        boxes = [set() for i in range(9)]
        todo = []
        for idy, row in enumerate(board):
            for idx, val in enumerate(row):
                if val == '.':
                    todo.append((idy,idx))
                else:
                    rows[idy].add(val)
                    cols[idx].add(val)
                    idbox = idy//3*3+idx//3
                    boxes[idbox].add(val)
        heap = []
        # the following is the key part
        for idy, idx in todo:
            row = rows[idy]
            col = cols[idx]
            box = boxes[idy//3*3+idx//3]
            remaining = allnums - row - col - box
            heapq.heappush(heap, (len(remaining), idy, idx, remaining))

        self.solve(board, heap)
```

