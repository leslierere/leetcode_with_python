### 48. Rotate Image

#### Solution-upside down flip+symmatry flip

Ref: https://leetcode.com/problems/rotate-image/discuss/18872/A-common-method-to-rotate-the-image



#### Solution

Ref: https://leetcode.com/problems/rotate-image/discuss/18884/Seven-Short-Solutions-(1-to-7-lines)

> we can instead do each four-cycle of elements by using three swaps of just two elements.

```python
class Solution:
    def rotate(self, A):
        n = len(A)
        for i in range(n/2):
            for j in range(n-n/2):
                for _ in '123':
                    A[i][j], A[~j][i], i, j = A[~j][i], A[i][j], ~j, ~i
                i = ~j
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





#### Solution

我惊呆了

Ref: https://leetcode.com/problems/spiral-matrix/discuss/20571/1-liner-in-Python-%2B-Ruby





### 59. Spiral Matrix II

https://leetcode.com/problems/spiral-matrix-ii/

#### Solution

just like 54, Solution1

