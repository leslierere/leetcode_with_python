Python可以直接比较object吗？

Counter 对不存在的直接加？





## Notion

### Tree

“a tree is an undirected graph in which any two vertices are connected by *exactly* one path. In other words, any connected graph without simple cycles is a tree.”

Ref: https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts

> (1) A tree is an undirected graph in which any two vertices are
> connected by exactly one path.
>
> (2) Any connected graph who has `n` nodes with `n-1` edges is a tree.
>
> (3) The degree of a vertex of a graph is the number of
> edges incident to the vertex.
>
> (4) A leaf is a vertex of degree 1. An internal vertex is a vertex of
> degree at least 2.
>
> (5) A path graph is a tree with two or more vertices that is not
> branched at all.
>
> (6) A tree is called a rooted tree if one vertex has been designated
> the root.
>
> (7) The height of a rooted tree is the number of edges on the longest
> downward path between root and a leaf.





### Binary search tree (BST)

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.



### Segment Tree





### Return

函数体内部可以用`return`随时返回函数结果；

函数执行完毕也没有`return`语句时，自动`return None`。

函数可以同时返回多个值，但其实就是一个tuple。





### boolean

The following values are considered false in Python:

- `None`
- `False`
- Zero of any numeric type. For example, `0`, `0.0`, `0j`
- Empty sequence. For example, `()`, `[]`, `''`. but `[[]]` would be evaluated to True
- Empty mapping. For example, `{}`
- objects of Classes which has `__bool__()` or `__len()__` method which returns `0` or `False`



### Bit Operation(Python)

位运算的precedence低于加减乘除

**storing negative numbers**, and **two’s complement**. Remember, the leftmost bit of an integer number is called the sign bit. A negative integer number in Java always has its sign bit turned on (i.e. set to 1). A positive integer number always has its sign bit turned off (0). Java uses the two’s complement formula to store negative numbers. <u>To change a number’s sign</u> using two’s complement, ﬂip all the bits, then add 1 (with a byte, for example, that would mean adding 00000001 to the ﬂipped value).

* ~ Binary Ones Complement

  flips all the bits

* & Binary AND

  bits are turned on only if both original bits are turned on

  > Used with 1, it basically masks the value to extract the lowest bit, or in other words will tell you if the value is even or odd.

* |Binary OR

  It copies a bit if it exists in either operand.

* ^ Binary XOR(excluesive or)

  turned on only if exaclty one of the original bits are turned on

* \>\> Binary Right Shift

  Bitwise right shift, it shifts the bits to the right by the specified number of places.
  
  左移一位乘以2
  
  右移一位除以2

```python
a = 60            # 60 = 0011 1100 
b = 13            # 13 = 0000 1101 

c = a & b;        # 12 = 0000 1100

c = a | b;        # 61 = 0011 1101 

c = a ^ b;        # 49 = 0011 0001

c = ~a;           # -61 = 1100 0011

c = a << 2;       # 240 = 1111 0000

c = a >> 2;       # 15 = 0000 1111
```

#### trick

##### 判断奇偶

`if a&1 == 0`, equal zero->even number, otherwise, odd number

##### 变相反数

flip all the bits, then add 1

##### 2的n次方

1<<n



### Catalan number

Definition: 

> 令h ( 0 ) = 1， catalan 数满足递推式：(总之每项是两个（和为n-1）变量的函数的乘积)
>
> **h ( n ) = h ( 0 ) \* h ( n - 1 ) + h ( 1 ) \* h ( n - 2 ) + ... + h ( n - 1 ) \* h ( 0 ) ( n >=1 )**
>
> 例如：
>
> h ( 1 ) = h ( 0 ) * h ( 0 ) = 1
>
> h ( 2 ) = h ( 0 ) * h ( 1 ) + h ( 1 ) * h ( 0 ) = 1 * 1 + 1 * 1 = 2
>
> h ( 3 ) = h ( 0 ) * h ( 2 ) + h ( 1 ) * h ( 1 ) + h ( 2 ) * h ( 0 ) = 1 * 2 + 1 * 1 + 2 * 1 = 5
>
> 卡塔兰数有一个通项公式。
>
> <img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaolecv6wvj30i808qdgm.jpg" alt="image-20200107130949465" style="zoom:33%;" />



### Dynamic Programming

> Dynamic Programming is mainly an optimization over plain [recursion](https://www.geeksforgeeks.org/recursion/). Wherever we see a recursive solution that has repeated calls for same inputs, we can optimize it using Dynamic Programming. The idea is to simply store the results of subproblems, so that we do not have to re-compute them when needed later. This simple optimization reduces time complexities from exponential to polynomial. For example, if we write simple recursive solution for [Fibonacci Numbers](https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/), we get exponential time complexity and if we optimize it by storing solutions of subproblems, time complexity reduces to linear.

![](https://www.geeksforgeeks.org/wp-content/uploads/Dynamic-Programming-1.png)



### Divide and Conquer

Let's follow here a solution template for the divide and conquer problems :

- Define the base case(s).
- Split the problem into subproblems and solve them recursively.
- Merge the solutions for the subproblems to obtain the solution for the original problem.



### priority queue

A priority queue is like a dictionary, it contains entries that each consists of a key and an associated value. However, while a dictionary is used when we want to be able to look up arbitrary key, a priority queue is used to prioritize entries, thus that you can easily access and manipulate the value with the largest/smallest key.

* 一般的operation
  * insert()
  * min()
  * removeMin()

* another operation

  * Bottom-up heap construction

    make complete tree of entries, in any order. Work backward from last internal node to root. When visiting node, bubble its entry down as in removeMin().

    O(n) operation. Reasoning:

    > Ref: G & T, P383
    >
    > Bottom-up construction of a heap with n entries takes O(n) time, assuming two keys can be compared in O(1) time. The primary cost of the construction is due to the down-heap steps performed at each nonleaf position.
    >
    > Let πv denote the path of T from nonleaf node v to its “inorder successor” leaf, that is, the path that starts at v, goes to the right child of v, and then goes down leftward until it reaches a leaf. The total running time of the bottom-up heap construction algorithm is therefore bounded by the sum ∑ v * πv .
    >
    > We claim that the paths **πv for all nonleaf v are edge-disjoint**, and thus the sum of the path lengths is bounded by the number of total edges in the tree, hence O(n).

* Implementation

**Binary heap**(i.e. a complete binary tree whose entries satisfy keep-order property. For example, the key of a child always >= the key of its parent, and in this way, this is a min heap)

To store as array, map treenodes to array indices with level-numbering: level-order traversal with root at index 1. In this way, node i's children are 2i and 2i+1, parent is i//2



### hashtable&dictionary

> A dictionary is a general concept that maps keys to values. There are many ways to implement such a mapping. A hashtable is a specific way to implement a dictionary. 
>
> In python, dictionary is a hash table.

(ref: [https://stackoverflow.com/questions/2061222/what-is-the-true-difference-between-a-dictionary-and-a-hash-table](https://stackoverflow.com/questions/2061222/what-is-the-true-difference-between-a-dictionary-and-a-hash-table), https://mail.python.org/pipermail/python-list/2000-March/031607.html)



### 正负无穷

```python
正无穷：float("inf"); 负无穷：float("-inf")
```

利用 inf 做简单加、乘算术运算仍会得到 inf



### 乘除法

**/** 默认得到float

**//** 舍小数得到int





### 访问限制

https://www.liaoxuefeng.com/wiki/1016959663602400/1017496679217440

> 如果要让内部属性不被外部访问，可以把属性的名称前加上两个下划线`__`，在Python中，实例的变量名如果以`__`开头，就变成了一个私有变量（private）

> 需要注意的是，在Python中，变量名类似`__xxx__`的，也就是以双下划线开头，并且以双下划线结尾的，是特殊变量，特殊变量是可以直接访问的，不是private变量，所以，不能用`__name__`、`__score__`这样的变量名。
>
> 有些时候，你会看到以一个下划线开头的实例变量名，比如`_name`，这样的实例变量外部是可以访问的，但是，按照约定俗成的规定，当你看到这样的变量时，意思就是，“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”。
>
> 双下划线开头的实例变量是不是一定不能从外部访问呢？其实也不是。不能直接访问`__name`是因为Python解释器对外把`__name`变量改成了`_Student__name`，所以，仍然可以通过`_Student__name`来访问`__name`变量



## Function

### str.isalnum()

https://docs.python.org/3/library/stdtypes.html#str.isalnum

Return `True` if all characters in the string are alphanumeric and there is at least one character, `False` otherwise. A character `c` is alphanumeric if one of the following returns `True`: `c.isalpha()`, `c.isdecimal()`, `c.isdigit()`, or `c.isnumeric()`.

### str.isdecimal()[¶](https://docs.python.org/3/library/stdtypes.html#str.isdecimal)

Return `True` if all characters in the string are decimal characters and there is at least one character, `False` otherwise. Decimal characters are those that can be used to form numbers in base 10, e.g. U+0660, ARABIC-INDIC DIGIT ZERO. Formally a decimal character is a character in the Unicode General Category “Nd”.

### str.isdigit()

Return `True` if all characters in the string are digits and there is at least one character, `False` otherwise. Digits include decimal characters and digits that need special handling, such as the compatibility superscript digits. This covers digits which cannot be used to form numbers in base 10, like the Kharosthi numbers. Formally, a digit is a character that has the property value **Numeric_Type=Digit** or **Numeric_Type=Decimal.**

### str.isnumeric()[¶](https://docs.python.org/3/library/stdtypes.html#str.isnumeric)

Return `True` if all characters in the string are numeric characters, and there is at least one character, `False` otherwise. Numeric characters include digit characters, and all characters that have the Unicode numeric value property, e.g. U+2155, VULGAR FRACTION ONE FIFTH. Formally, numeric characters are those with the property value **Numeric_Type=Digit**, **Numeric_Type=Decimal** or **Numeric_Type=Numeric**.

```python
In [28]: my_string = '\u00BC'                                                   
In [29]: my_string.isdecimal()                                                  
Out[29]: False

In [30]: my_string.isdigit()                                                    
Out[30]: False

In [31]: my_string.isnumeric()                                                  
Out[31]: True
```



### str.isalpha()

Return `True` if all characters in the string are alphabetic and there is at least one character, `False` otherwise. Alphabetic characters are those characters defined in the Unicode character database as “Letter”, i.e., those with general category property being one of “Lm”, “Lt”, “Lu”, “Ll”, or “Lo”. Note that this is different from the “Alphabetic” property defined in the Unicode Standard.



### split()

```python
In [21]: s2.split??                                                         
'''
Signature: s2.split(sep=None, maxsplit=-1)
Docstring:
Return a list of the words in the string, using sep as the delimiter string.

sep
  The delimiter according which to split the string.
  None (the default value) means split according to any whitespace,
  and discard empty strings from the result.
maxsplit
  Maximum number of splits to do.
  -1 (the default value) means no limit.
Type:      builtin_function_or_method
'''
# split()和split('')是不一样的
# 只要声明了sep，那么空字符串也会被单独分组
  
In [24]: s1                                                                 
Out[24]: '  subdir1'

In [25]: s2                                                                 
Out[25]: ' subdir1'

In [26]: s1.split(' ')                                                       
Out[26]: ['', '', 'subdir1']

In [27]: s2.split(' ')                                                       
Out[27]: ['', 'subdir1']
  
In [34]: s1.split()                                                         
Out[34]: ['subdir1']

In [35]: s2.split()                                                         
Out[35]: ['subdir1']
```







### join()

join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

The string whose method is called is inserted in between each given string.

The result is returned as a new string.

Example: 

```python
'.'.join(['ab', 'pq', 'rs'])
>>>'ab.pq.rs'
```







### count()

```
list.count(obj)
str.count(sub, start= 0,end=len(string))
#sub -- 搜索的子字符串
#start -- 字符串开始搜索的位置。默认为第一个字符,第一个字符索引值为0。
#end -- 字符串中结束搜索的位置。字符中第一个字符的索引为 0。默认为字符串的最后一个位置。
```



### enumerate()

用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

`enumerate(sequence, [start=0])`



### setdefault()

> https://www.tutorialspoint.com/python/dictionary_setdefault.htm
>
> Python dictionary method **setdefault()** is similar to get(), but will set *dict[key]=default* if key is not already in dict.
>
> Following is the syntax for **setdefault()** method −
>
> ```python
> dict.setdefault(key, default=None)
> ```
>
> Parameters
>
> - **key** − This is the key to be searched.
> - **default** − This is the Value to be returned in case key is not found.
>
> Return Value
>
> This method returns the key value available in the dictionary and if given key is not available then it will return provided default value.



### map()

map(function, iterable,…)

iterable,…可以传入一个或多个序列

python3返回迭代器



### lambda

Say you wanna sort a list of strings by string length, you can first define a function:

```python
def by_length(animal):
  return len(animal)
```

Now you pass `by_length` to `sorted` (YES, functions can be parameters) using the `key` parameter. When you see a parameter being named (as in `key=by_length`), this parameter is called a *keyword* or *named argument*.

```python
animals = ["Fox", "Snake", "Octopus", "Bear"]
print(sorted(animals, key=by_length))
```

The function `by_length` <u>will be called for every value</u> in the animals' list. The `sorted` function will use the returned values to determine the order.

Since the `sort` function just calls the built-in `len` function, you can just use that:

```
print(sorted(animals, key=len))
```

As you may have noticed these sort functions are usually pretty short. There's a way in Python to create/define a function using a special syntax that allows you to write the entire function in one line of code! It's called a **lambda** function. Let's see how to create one.

What if you had a list of numbers as strings and you wanted to sort the list based on the numeric value?

We can create a sort function to help us:

```
values = ["1", "10", "2", "20"] 
def by_value(num):
  return int(num)
print(sorted(values, key=by_value))
```

There is a special shorthand notation that's very useful for passing a function into another function called a *lambda* function

```python
by_value = lambda x: int(x)
values = ["1", "10", "2", "20"]
print(sorted(values, key=by_value))
```

- `lambda` is a keyword
- `x` is the parameter, it can be any legal variable name
- `:` separates the input from the output
- `int(x)` (all the stuff to the right of the `:`) is what value is returned

So everything between `lambda` and the colon is the input and everything after the colon is the output.

Every time you create a function it's visible to the entire module it's created in. However, many of these functions are for "one-time" use and there's no need to keep them around after the sorting has finished. The nice thing about `lambda` functions is that they can be created in-line with the function call:

```python
values = ["1", "10", "2", "20"]
print(sorted(values, key=lambda x: int(x)))
```

In this case, since this `lambda` function has no name, it's referred to as an anonymous function. Anonymous functions were first studied in lambda calculus and that's why the keyword `lambda` is used.





### 列表生成器List Comprehensions

生成[1x1, 2x2, 3x3, ..., 10x10], 把要生成的元素`x * x`放到前面，后面跟`for`循环，for循环后面还可以加上if判断，这样我们就可以筛选出仅偶数的平方：

```python
>>>[x * x for x in range(1, 11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>> [x * x for x in range(1, 11) if x % 2 == 0]
[4, 16, 36, 64, 100]
```

列表生成式也可以使用两个变量来生成list:

```python
>>> d = {'x': 'A', 'y': 'B', 'z': 'C' }
>>> [k + '=' + v for k, v in d.items()]
['y=B', 'x=A', 'z=C']
```

leetcode760:

```python
def numJewelsInStones(self, J: str, S: str) -> int:
        import collections 
        stones = collections.Counter(S)
        return sum([stones[j] for j in J])
```

Ref: https://leetcode.com/problems/jewels-and-stones/discuss/327540/Python-hashmap-esay-solution



### any()&all()

`any()` function returns True if any item in an iterable are true, otherwise it returns False.

`any(*iterable*)`

`all()` returns true if all of the items are True (or if the iterable is empty). All can be thought of as a sequence of AND operations on the provided iterables. It also short circuit the execution i.e. stop the execution as soon as the result is known.





### max()

https://leetcode.com/problems/longest-word-in-dictionary/discuss/186128/O(1)-Space!-NlogN-Solution!-PythonC%2B%2B

### zip()

[https://leetcode.com/problems/sentence-similarity/discuss/109624/Simple-Python-Solution-32ms!](https://leetcode.com/problems/sentence-similarity/discuss/109624/Simple-Python-Solution-32ms!)

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。

我们可以使用 list() 转换来输出列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 ***** 号操作符，可以将元组解压为列表。

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 返回一个对象
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list() 转换为列表
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]

>>> a1, a2 = zip(*zip(a,b))          # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
```





### Ascii-character

```python
# Get the ASCII number of a character
number = ord(char)

# Get the character given by an ASCII number
char = chr(number)
```



### String

String and list both belongs to the [`collections.abc.Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence) ABC, for more about the ABC, see https://docs.python.org/3/library/stdtypes.html#common-sequence-operations.

#### string.find()

```python
str.find(sub[, start[, end]] )
```

- Parameter

  * **sub** - It's the substring to be searched in the str string.

  * **start** and **end** (optional) - substring is searched within `str[start:end]`

- It returns value: 

  * If substring exists inside the string, it returns the index of first occurence of the substring.
  * If substring doesn't exist inside the string, it returns -1.

Index() has the same function and unlike find(), it can also be used by list. However, it will throw an exception if it doesn't find it.

#### string.count()

```python
# S.count(sub[, start[, end]]) -> int
s = 'apple'
s.count('p') 
# Out[12]: 2
```



#### string.isalnum()

Return True if the string is an alpha-numeric string, False otherwise.

类似的还有string.isdigit()检测字符串是否只由数字组成 , string.isalpha(), 检测字符串是否只由字母组成

```python
In [16]: s='a'                                                                  
In [17]: s.isalnum()                                                            
Out[17]: True
```





#### isupper(), islower(), lower(), upper()

```python
In [133]: "9".lower()   # 不会报错                                                             
Out[133]: '9'
```







### set

set()可以对字符串操作

```python
s = 'apple'
set(s) 
# Out[11]: {'a', 'e', 'l', 'p'}
```

数学意义上的交集、并集等操作

```python
# union
In [17]: people = {"Jay", "Idrish", "Archil"}  

In [18]: people|vampires                                                        
Out[18]: {'Archil', 'Arjun', 'Idrish', 'Jay', 'Karan'}
  
# intersection
In [26]: s1                                                                 
Out[26]: {0, 1, 2, 3, 4}

In [27]: s2                                                                 
Out[27]: {3, 4, 5, 6}

In [28]: s1&s2                                                               
Out[28]: {3, 4}

# Difference
In [31]: s1-s2                                                               
Out[31]: {0, 1, 2}
  
# the set of elements in precisely one of s1 or s2  
In [32]: s1^s2                                                               
Out[32]: {0, 1, 2, 5, 6}
  
# s1 is subset of s2, return boolean
s1 <= s2

# s1 is proper subset of s2
s1 < s2
```





### int()

Convert base-2 binary number string to int](https://stackoverflow.com/questions/8928240/convert-base-2-binary-number-string-to-int)

```python
>>> int('11111111', 2)
255
```



#### Covert int to binary number string-bin()



### Reading and Writing Files

Ref: https://docs.python.org/3/tutorial/inputoutput.html

> [`open()`](https://docs.python.org/3/library/functions.html#open) returns a [file object](https://docs.python.org/3/glossary.html#term-file-object), and is most commonly used with two arguments: `open(filename, mode)`.
>
> ```python
> >>> f = open('workfile', 'w')
> ```

Mode: 

* `'r'` when the file will only be read, default mode
*  `'w'` for only writing (an existing file with the same name will be erased)
* `'a'` opens the file for appending; any data written to the file is automatically added to the end.
*  `'r+'` opens the file for both reading and writing. 

> Normally, files are opened in *text mode*, that means, you read and write strings from and to the file, which are <u>encoded in a specific encoding</u>. If encoding is not specified, the default is platform dependent (see [`open()`](https://docs.python.org/3/library/functions.html#open)). `'b'` appended to the mode opens the file in *binary mode*: now the data is read and written in the form of bytes objects. This mode should be used for all files that don’t contain text.

It is good practice to use the [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) keyword when dealing with file objects. The advantage is that the file is properly closed after its suite finishes, even if an exception is raised at some point.

```python
>>> with open('workfile') as f:
...     read_data = f.read()

>>> # We can check that the file has been automatically closed.
>>> f.closed
True
```

If you’re not using the [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) keyword, then you should call `f.close()` to close the file and immediately free up any system resources used by it.



## Module

### Container

#### list

- 排序

  * sorted()

  It returns a new sorted list

  ```python
  >>> sorted([5, 2, 3, 1, 4])
  [1, 2, 3, 4, 5]
  ```

  * [`list.sort()`](https://docs.python.org/3/library/stdtypes.html#list.sort)

  It modifies the list in-place (and returns `None`).

  Another difference is that the `list.sort()` method is only defined for lists. In contrast, the `sorted()`function accepts any iterable.

- `numbers = list(range(1, 6, 2)) #[1, 3, 5]`

- 复制列表

  `list2 = list1[:]`

- reverse()

```python
list.reverse()
```

* reversed()

You can also use the global `reversed` function to reverse items; however, it does not modify its parameter. The `reversed` function also does **not** return a list (it returns an *iterator*):

#### tuple

* 如果要定义一个空的tuple，可以写成`()`

  ```pytho
  >>> t = ()
  >>> t
  ()
  ```

* 只有1个元素的tuple定义时必须加一个逗号`,`，来消除歧义：

  ```python
  >>> t = (1,)
  >>> t
  (1,)
  ```

#### dict

- dict的get()

  如果key不存在，可以返回`None`，或者自己指定的value

  ```python
  >>> d.get('Thomas')
  >>> d.get('Thomas', -1)
  -1
  ```

#### set

set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。要创建一个set，需要提供一个list作为输入集合.

* 将一个字符串变为单个字母的set

```python
s1 = set("QWERTYUIOPqwertyuiop")

print(s1)
#{'T', 'e', 't', 'P', 'O', 'r', 'u', 'Y', 'I', 'Q', 'q', 'y', 'p', 'W', 'R', 'w', 'i', 'E', 'o', 'U'}
```

* set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作：

```python
>>> s1 = set([1, 2, 3])
>>> s2 = set([2, 3, 4])
>>> s1 & s2
{2, 3}
>>> s1 | s2
{1, 2, 3, 4}
```

 

### collection

#### collections.Counter()

Ref: https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter

一个 [`Counter`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter) 是一个 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 的子类，用于计数可哈希对象。它是一个集合，元素像字典键(key)一样存储，它们的计数存储为值。计数可以是任何整数值，包括0和负数。

```python
s='apple'
count = collections.Counter(s) 
In [15]: count                                                                  
Out[15]: Counter({'a': 1, 'p': 2, 'l': 1, 'e': 1})

In [17]: c = collections.Counter({'red': 4, 'blue': 2})                         
In [18]: c                                                                      
Out[18]: Counter({'red': 4, 'blue': 2})
  
In [19]: c =collections.Counter(cats=4, dogs=8)                                 
In [20]: c                                                                      
Out[20]: Counter({'cats': 4, 'dogs': 8})
```



* Counter对象有一个字典接口，如果引用的键没有任何记录，就返回一个0，而不是弹出一个 [`KeyError`](https://docs.python.org/zh-cn/3/library/exceptions.html#KeyError)

  ```python
  >>> c = Counter(['eggs', 'ham'])
  >>> c['bacon']# count of a missing element is zero
  0
  ```

* 使用 `del` 来删除key

* elements()

  返回一个迭代器，每个元素重复计数的个数。元素顺序是任意的。如果一个元素的计数小于1， [`elements()`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter.elements) 就会忽略它。

  ```python
  >>> c = Counter(a=4, b=2, c=0, d=-2)
  >>> sorted(c.elements())
  ['a', 'a', 'a', 'a', 'b', 'b']
  ```

* 通常字典方法都可用于 [`Counter`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter) 对象

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        count = collections.Counter(s)
        l = count.values()#返回每个值个数的一个list
        
        for k, v in enumerate(s):
            if count[v]==1:
                return k
        return -1
```

* 常用方法

```python
In [20]: c                                                                      
Out[20]: Counter({'cats': 4, 'dogs': 8})
  
sum(c.values())                 # total of all counts
c.clear()                       # reset all counts
list(c)                         # list unique elements
In [22]: list(c)                                                                  
Out[22]: ['cats', 'dogs']
  
set(c)                          # convert to a set
In [21]: set(c)                                                                   
Out[21]: {'cats', 'dogs'}
  
dict(c)                         # convert to a regular dictionary
c.items()                       # convert to a list of (elem, cnt) pairs
In [23]: c.items()                                                                
Out[23]: dict_items([('cats', 4), ('dogs', 8)])
  
+c                              # remove zero and negative counts
>>> c = Counter(a=2, b=-4)
>>> +c
Counter({'a': 2})
```

* 数学操作

```python
>>> c = Counter(a=3, b=1)
>>> d = Counter(a=1, b=2)
>>> c + d                       # add two counters together:  c[x] + d[x]
Counter({'a': 4, 'b': 3})
>>> c - d                       # subtract (keeping only positive counts)
Counter({'a': 2})
>>> c & d                       # intersection:  min(c[x], d[x]) 
Counter({'a': 1, 'b': 1})
>>> c | d                       # union:  max(c[x], d[x])
Counter({'a': 3, 'b': 2})
```



#### collections.defaultdict()

* *class* `collections.defaultdict`([*default_factory*[, *...*]])

  返回一个新的类似字典的对象。 [`defaultdict`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict) 是内置 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 类的子类。它重载了一个方法并添加了一个可写的实例变量。其余的功能与 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 类相同，此处不再重复说明。

  第一个参数 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory) 提供了一个初始值。它默认为 `None` 。所有的其他参数都等同与 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 构建器中的参数对待，包括关键词参数。

* 使用 [`list`](https://docs.python.org/zh-cn/3/library/stdtypes.html#list) 作为 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory)

```python
In [24]: s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]    
In [26]: d = collections.defaultdict(list)                                           
In [27]: for k, v in s: 
    ...:     d[k].append(v) 
    ...:                                                                             
In [28]: d                                                                           
Out[28]: defaultdict(list, {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]})
# 当每个键第一次遇见时，它还没有在字典里面；所以条目自动创建，通过 default_factory 方法，并返回一个空的 list 。 list.append() 操作添加值到这个新的列表里。当键再次被存取时，就正常操作， list.append() 添加另一个值到列表中。这个计数比它的等价方法 dict.setdefault() 要快速和简单
```

* 设置 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory) 为 [`int`](https://docs.python.org/zh-cn/3/library/functions.html#int) ，使 [`defaultdict`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict) 在计数方面发挥好的作用（像其他语言中的bag或multiset）

```python
In [32]: d = collections.defaultdict(int)                                            
In [33]: s = 'mississippi'                                                           
In [34]: d = collections.defaultdict(int)                                            

In [35]: for k in s: 
    ...:     d[k] += 1 
    ...:                                                                             
In [36]: d                                                                           
Out[36]: defaultdict(int, {'m': 1, 'i': 4, 's': 4, 'p': 2})
# 当一个字母首次遇到时，它就查询失败，所以 default_factory 调用 int() 来提供一个整数0作为默认值。自增操作然后建立对每个字母的计数。
```

* 设置 [`default_factory`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict.default_factory) 为 [`set`](https://docs.python.org/zh-cn/3/library/stdtypes.html#set) 使 [`defaultdict`](https://docs.python.org/zh-cn/3/library/collections.html#collections.defaultdict) 用于构建字典集合

```python
>>> s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
>>> d = defaultdict(set)
>>> for k, v in s:
...     d[k].add(v)
...
>>> sorted(d.items())
[('blue', {2, 4}), ('red', {1, 3})]
```



#### collections.deque()

https://docs.python.org/zh-cn/3/library/collections.html#collections.deque

*class* `collections.deque([*iterable*[, *maxlen*]])`

双向队列对象

```python
>>> from collections import deque
>>> d = deque('ghi')                 # make a new deque with three items
>>> d.append('j')                    # add a new entry to the right side
>>> d.appendleft('f')                # add a new entry to the left side
>>> d                                # show the representation of the deque
deque(['f', 'g', 'h', 'i', 'j'])

>>> d.pop()                          # return and remove the rightmost item
'j'
>>> d.popleft()                      # return and remove the leftmost item
'f'
>>> list(d)                          # list the contents of the deque
['g', 'h', 'i']
>>> d[0]                             # peek at leftmost item
'g'
>>> d[-1]                            # peek at rightmost item
'i'

>>> list(reversed(d))                # list the contents of a deque in reverse
['i', 'h', 'g']
>>> 'h' in d                         # search the deque
True
>>> d.extend('jkl')                  # add multiple elements at once
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> d.rotate(1)                      # right rotation
>>> d
deque(['l', 'g', 'h', 'i', 'j', 'k'])
>>> d.rotate(-1)                     # left rotation
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])

>>> deque(reversed(d))               # make a new deque in reverse order
deque(['l', 'k', 'j', 'i', 'h', 'g'])
>>> d.clear()                        # empty the deque
>>> d.pop()                          # cannot pop from an empty deque
Traceback (most recent call last):
    File "<pyshell#6>", line 1, in -toplevel-
        d.pop()
IndexError: pop from an empty deque

>>> d.extendleft('abc')              # extendleft() reverses the input order
>>> d
deque(['c', 'b', 'a'])
```





### queue

https://docs.python.org/3/library/queue.html

* *class* `queue.Queue(*maxsize=0*)`

Constructor for a **FIFO** queue. *maxsize* is an integer that sets the upperbound limit on the number of items that can be placed in the queue. Insertion will block once this size has been reached, until queue items are consumed. If *maxsize* is less than or equal to zero, the queue size is infinite.

* *class* `queue.LifoQueue(maxsize=0)`

Constructor for a **LIFO** queue. 

* class `queue.PriorityQueue(maxsize=0)`

Constructor for a priority queue. The lowest valued entries are retrieved first (the lowest valued entry is the one returned by `sorted(list(entries))[0]`).

##### Public methods for [`Queue`](https://docs.python.org/3/library/queue.html#queue.Queue), [`LifoQueue`](https://docs.python.org/3/library/queue.html#queue.LifoQueue), or [`PriorityQueue`](https://docs.python.org/3/library/queue.html#queue.PriorityQueue)

* `Queue.qsize()`

  Return the approximate size of the queue. Note, qsize() > 0 doesn’t guarantee that a subsequent get() will not block, nor will qsize() < maxsize guarantee that put() will not block.

* `Queue.empty()`

  Return `True` if the queue is empty, `False` otherwise.

* `Queue.full()`

  Return `True` if the queue is full, `False` otherwise. 

* `Queue.put(item, block=True, timeout=None)`

  Put item into the queue. If optional args block is true and timeout is None (the default), block if necessary until a free slot is available. If timeout is a positive number, it blocks at most timeout seconds and raises the Full exception if no free slot was available within that time. Otherwise (block is false), put an item on the queue if a free slot is immediately available, else raise the Full exception (timeout is ignored in that case).

* `Queue.get(*block=True*, *timeout=None*)`

  Remove and return an item from the queue. 

* `Queue.join()`

  Blocks until all items in the queue have been gotten and processed.



### heapq

可以参照cs61b-priority queue

***Priority queue in Python, 堆***

> 堆是一种树形数据结构，其中子节点与父节点之间是一种有序关系。最大堆中父节点大于或等于两个子节点，最小堆父节点小于或等于两个子节点。Python的heapq模块实现了一个最小堆。

* 创建堆

  * 用[]初始化

* 已有list转化为heap，[heapify()](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heapify)，heapq.heapify(*list*)

* heapq模块可以接受元组对象，默认元组的第一个元素作为`priority`

* heapq.heappush(*heap*, *item*)

  将 *item* 的值加入 *heap* 中，保持堆的不变性。

* heapq.heappop(*heap*)

  弹出并返回 heap 的最小的元素，保持堆的不变性。如果堆为空，抛出 IndexError 。使用 heap[0] ，可以只访问最小的元素而不弹出它。

* heapq.heappushpop(*heap*, *item*)

  将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用  heappush() 再调用 heappop()运行起来更有效率。

LC378,[Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

```python
from heapq import *

def find_Kth_smallest(matrix, k):
    minHeap = []

    # put the 1st element of each row in the min heap
    # we don't need to push more than 'k' elements in the heap
    for i in range(min(k, len (matrix))):
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



***Real Priority queue in Python***

[`PriorityQueue`](https://docs.python.org/3/library/queue.html#queue.PriorityQueue)







### bisect

Ref: https://docs.python.org/zh-cn/3.6/library/bisect.html

这个模块对有序列表提供了支持，使得他们可以在插入新数据仍然保持有序。

* `bisect.bisect_left(*a*, *x*, *lo=0*, *hi=len(a)*)`

  在 *a* 中找到 *x* 合适的插入点以维持有序。参数 *lo* 和 *hi* 可以被用于确定需要考虑的子集；默认情况下整个列表都会被使用。如果 *x* 已经在 *a* 里存在，那么插入点会在已存在元素之前（也就是左边）。

* `bisect.bisect_right(*a*, *x*, *lo=0*, *hi=len(a)*)`

  Similar to bisect_left(), but returns an insertion point which comes after (to the right of) any existing entries of item in list.

* `bisect.bisect(*a*, *x*, *lo=0*, *hi=len(a)*)`

  Alias for `bisect_right()`

```python
In [39]: import bisect
In [42]: data = [2,4,6,8]                                                            

In [43]: bisect.bisect(data,1)                                                       
Out[43]: 0

In [44]: bisect.bisect(data,4)                                                       
Out[44]: 2

In [45]: bisect.bisect_right(data,4)                                                 
Out[45]: 2

In [46]: bisect.bisect_left(data,4)                                                  
Out[46]: 1
```











## 经典写法

### GCD

```python
def gcd(self, x, y):
  if y==0:
    return x
  else:
    return self.gcd(y, x%y)
```





### Traversal

ref: https://www.youtube.com/watch?v=A6iCX_5xiU4

![image-20200103121202462](https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5wbuomj31c00u0n9b.jpg)

* In a **_preorder_** traversal, you visit each node before recursively visiting its children, which are visited from left to right. The root is visited first.

  A preorder traversal is a natural way to print a directory’s structure. Simply have the method visit() *print each node of the tree*.

```java
class SibTreeNode {
  public void preorder() { 
    this.visit(); if (firstChild != null) { 
      firstChild.preorder(); 
    } 
    if (nextSibling != null) { 
      nextSibling.preorder(); 
    }
  }
}
```

* In a **_postorder_** traversal, you visit each node’s children (in left-to-right order) before the node itself.

```java
public void postorder() {
  if (firstChild != null) { 
    firstChild.postorder(); 
  } 
  this.visit(); 
  if (nextSibling != null) {
    nextSibling.postorder(); 
  }
}
```

​	A postorder traversal visits the nodes in this order.

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1gaol5x3gfzj308y06uaa3.jpg" alt="image-20200102202316343" style="zoom:40%;" />

​	The postorder() code is trickier than it looks. The best way to understand it is to draw a depth-two tree on paper, then pretend you’re the computer and execute the algorithm carefully. Trust me on this. It’s worth your time.

​	A postorder traversal is the *natural way to sum the total disk space used in the root directory and its descendants*. The method visit() sums "this" node’s disk space with the disk space of all its children. In the example above, a postorder traversal would first sum the sizes of the files in hw1/ and hw2/; then it would visit hw/ and sum its two children. The last thing it would compute is the total disk space at the root ˜jrs/61b/, which sums all the files in the tree.

* Binary trees allow for an **_inorder_** traversal: recursively traverse the root’s left subtree (rooted at the left child), then the root itself, then the root’s right subtree.
* In a **_level-order_** traversal, you visit the root, then all the depth-1 nodes (from left to right), then all the depth-2 nodes, et cetera. The level-order traversal of our expression tree is "+ * ^ 3 7 4 2" (which doesn’t mean much).Unlike the three previous traversals, a level-order traversal is not straightforward to define recursively. However, a level-order traversal can be done in O(n) time.***Use a queue, which initially contains only the root***. Then repeat the following steps:
  - Dequeue a node.
  - Visit it.
  - Enqueue its children (in order from left to right). Continue until the queue is empty.

**A final thought**: if you use a stack instead of a queue, and push each node’s children in reverse order--from right to left (so they pop off the stack in order from left to right)--you perform a preorder traversal. Think about why.





## Trick

### 防止0的出现

```python
size = (b-a)//(len(num)-1) or 1
```



### The use of *

Ref: https://zhuanlan.zhihu.com/p/54738146

* 解压(unpack)的功能。zip函数的基本用法如下:

```python
stuff = ['apple','banana','peach']
money = [10, 5, 7]

pair = list(zip(stuff,money))
# pair = [('apple',10),('banana',5),('peach',7)]
```

但是如果我们现在已经有 pair 这个 list 了，希望能够还原成stuff 和 money 两个 list,我们就需要用到*符号:

```python
stuff,money = zip(*pair)
```

* \* 对于迭代对象的作用

```python
def do_something(x,y,z):
    print('x is ', x)
    print('y is ', y)
    print('z is ', z)

list1 = ['a','b','c']

>>> do_something(list1)
Trackback (most recent call last):
   File "<stdin>", line1, in <module>
TypeError: do() missing 2 required positional arguments: 'y' and 'z'

>>> do_something(*list1)
x is a
y is b
z is c
```

* **的用法

对于字典(dict)来说，我们也可以使用`*`运算符，传入key值:

```python
dict1 = {'x':1, 'y':2, 'z':3}

>>> do_something(*dict1)
x is x
y is y
z is z
```

传入 value 值，我们需要使用`**`运算符:

```python
>>> do_somthing(**dict1)
x is 1
y is 2
z is 3
```

**一定要注意的是，此处的用法必须要求函数形参（do_something(x,y,z)， 形参是x, y, z这三个）与字典 key 值一一对应**，请看下面的用法:

```python
dict2 = {'z':1, 'x':2, 'y':3}
>>> do_something(**dict2)
x is 2
y is 3
z is 1

dict3 = {'a':1, 'b':2, 'c':3}
>>> do_something(**dict3)
# TypeError: do_something() missing 1 required positional argument: 'x'
```

* `*args` 和`**kwargs`

`*args`和`**kwargs`是我们经常见到的两个参数，但这两个参数究竟是干什么的可能有些人仍有疑惑。其实它们的用法也很简单，对于一个函数而言，如果你不确定它还有什么待传入的参数，就不妨用一个`*args`(当然你不一定非得叫`args`，但一般都喜欢起这个名字)，它将输入的多余形参以元组形式保存在`args`中:

```python
#两个数的加法
def adder_2(x,y):
  return x+y

#三个数的加法
def adder_3(x,y,z):
  return x+y+z

# 无限个数的加法
def adder_unlimited(*args):
   result = 0
   for num in args:
       result += num
   return result

>>> adder_unlimited(1)
1
>>> adder_unlimited(1,2)
3
>>> adder_unlimited(1,2,3,4)
10

>>> list_num = [1,2,3,4]
>>> adder_unlimited(*list_num)  #活学活用
10
```

`**kwargs`效果则是将输入的未定义的形参及其对应的值存在`kwargs`字典里(例子来源Reference 4):

```python
def intro(**data):
    print("\nData type of argument:",type(data))
    for key, value in data.items():
        print("{} is {}".format(key,value))

>>> intro(Firstname="Sita", Lastname="Sharma", Age=22, Phone=1234567890)

Data type of argument: <class 'dict'>
Firstname is Sita
Lastname is Sharma
Age is 22
Phone is 1234567890
```

因此，对于一个好的 API 来说，应该尽量使用`*args`和`**kwargs`以提高程序稳定性



to be continued......



### The use of comma

Ref: https://leetcode.com/problems/summary-ranges/discuss/63193/6-lines-in-Python

I have these two basic cases:

```python
ranges += [],
r[1:] = n,
```

Why the trailing commas? Because it turns the right hand side into a tuple and I get the same effects as these more common alternatives:

```python
ranges += [[]]
or
ranges.append([])

r[1:] = [n]
```



Without the comma, ...

- `ranges += []` wouldn't add `[]` itself but only its elements, i.e., nothing.
- `r[1:] = n` wouldn't work, because my `n` is not an iterable.

Why do it this way instead of the more common alternatives I showed above? Because it's shorter and faster (according to tests I did a while back).





###  "while ... else" clause

Ref:  [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii)



### is

`is None` *is* ok

Also, when you compare a value to `None`, it is recommended (it is Pythonic to do so) that you use the **`is`** operator and not `==`:

```
if parameter is None:
   # do something
```

`is True` *is* **not** ok

Do NOT use the `is` operator for testing other values (numbers, strings, booleans, etc).



## span lines in Python

n addition to using variables, you can use parentheses to allow your code to span multiple lines (splitting the line with a carriage return (or newline) will result in a syntax error):

```
def will_eat(hungry, price):
  if (price > 1.00 and price < 2.00 
     or hungry == True):
    return True
  return False
```

Note in the above code, parentheses are used to allow the line to be split over more than one line. You can avoid this by using an optional line continuation character '' (the backslash):

```
def will_eat1(price, hungry):
  if price == 0.25 or \
     price == 0.50 or \
     price == 1.00 and hungry == True:
    return True
  return False
```

The above can be simplified by just returning the result of the expression (skipping the entire if statement) as well.



## Math more

mean minimizes total distance for euclidian distance
median minimzes total distance for absolute deviation
mode minimizes distance for indicator function

#### euclidian distance

In [Cartesian coordinates](https://en.wikipedia.org/wiki/Cartesian_coordinates), if **p** = (*p*1, *p*2,..., *p**n*) and **q** = (*q*1, *q*2,..., *q**n*) are two points in [Euclidean *n*-space](https://en.wikipedia.org/wiki/Euclidean_space), then the distance (d) from **p** to **q**, or from **q** to **p** is given by the [Pythagorean formula](https://en.wikipedia.org/wiki/Pythagorean_theorem):[[1\]](https://en.wikipedia.org/wiki/Euclidean_distance#cite_note-Anton-1)

![image-20200421101855162](https://tva1.sinaimg.cn/large/007S8ZIlgy1ge1ssp1mvyj3108086gmi.jpg)

#### indicator function

In [mathematics](https://en.wikipedia.org/wiki/Mathematics), an **indicator function** or a **characteristic function** is a [function](https://en.wikipedia.org/wiki/Function_(mathematics)) defined on a [set](https://en.wikipedia.org/wiki/Set_(mathematics)) *X* that indicates membership of an [element](https://en.wikipedia.org/wiki/Element_(mathematics)) in a [subset](https://en.wikipedia.org/wiki/Subset) *A* of *X*, having the value 1 for all elements of *A* and the value 0 for all elements of *X* not in *A*. It is usually denoted by a symbol 1 or *I*, sometimes in boldface or [blackboard boldface](https://en.wikipedia.org/wiki/Blackboard_bold), with a subscript specifying the subset.

#### minimax

**Without pruning:**

![image-20210413121546057](https://tva1.sinaimg.cn/large/008eGmZEly1gpime8b77cj31c00u04jc.jpg)

Static evaluation: AKA heuristic value of node

Ref: https://en.wikipedia.org/wiki/Minimax#:~:text=The%20heuristic%20value%20is%20a,favorable%20for%20the%20minimizing%20player.

The minimax function returns a heuristic value for [leaf nodes](https://en.wikipedia.org/wiki/Leaf_nodes) (terminal nodes and nodes at the maximum search depth). Non-leaf nodes inherit their value from a descendant leaf node. The heuristic value is a score measuring the favorability of the node for the maximizing player. Hence nodes resulting in a favorable outcome, such as a win, for the maximizing player have higher scores than nodes more favorable for the minimizing player. The heuristic value for terminal (game ending) leaf nodes are scores corresponding to win, loss, or draw, for the maximizing player. For non terminal leaf nodes at the maximum search depth, an evaluation function **estimates** a heuristic value for the node. The quality of this estimate and the search depth determine the quality and accuracy of the final minimax result.



**with pruning**

Alpha, beta keep track of the score either side can achieve assuming the best play from the opponent.

In the initial score, the positive infinity represents the worst possible score for white, the negative infinity represents the worst possible score for black

![image-20210413122018479](https://tva1.sinaimg.cn/large/008eGmZEly1gpimisulcyj31c00u01kx.jpg)



#### Huffman codes

Ref: 

[Robert-Sedgewick,-Kevin-Wayne]-Algorithms,-4th-Ed(z-lib.org)

Jeff

We now examine a data-compression technique that can save a substantial amount of space in natural language ﬁles (and many other kinds of ﬁles). The idea is to abandon the way in which text ﬁles are usually stored: instead of using the usual 7 or 8 bits for each character, we use fewer bits for characters that appear often than for those that appear rarely.

FIrst, we think of *A code associates each character with a bitstring: a symbol table with characters as keys and bitstrings as values.* But we will need delimiters.

The next step is to *take advantage of the fact that delimiters are not needed if no character code is the preﬁx of another.* A code with this property is known as a **preﬁx-free** code. All preﬁx-free codes are uniquely decodable (without needing any delimiters) in this way, so preﬁx-free codes are widely used in practice. Note that ﬁxedlength codes such as 7-bit ASCII are preﬁx-free. Trie representation for preﬁx-free codes. One convenient way to represent a preﬁx-free code is with a trie.

##### Optimal

How do we ﬁnd the trie that leads to the best preﬁx-free code? 

We have observed that high-frequency characters are nearer the root of the tree than lower-frequency characters and are therefore encoded with fewer bits, so this is a good code, but why is it an optimal preﬁx-free code?

To answer this question, we begin by deﬁning the weighted external path length of a tree to be the sum of the weight (associated frequency count) times depth (see page 226) of all of the leaves. For any preﬁx-free code, the length of the encoded bitstring is equal to the weighted external path length of the corresponding trie.







## Notion

### Signature

Two functions have the same signature if their input types and order are the same and their output types are the same.



### Method V.S. function

A method is a function that is owned by an object. The function `len` is NOT a method since you can use it on many types of values (e.g. both lists and strings).



### programming constructs 

All imperative languages (e.g Java, C, JavaScript, etc) have three programming constructs (things that the language allows you to do):

- Sequential Control Flow
- Selection
- Iteration

Since all three determine how the code is run, these programming constructs are also called control flow structures.



### Bytes

A kilobyte is not exactly, as one might expect, of 1000 bytes. Rather, the correct amount is 2^10 i.e. 1024 bytes. 



## Notes

* pay attention to the situation where the input can be null
* able to explain why you should use a specific data structure in your solution