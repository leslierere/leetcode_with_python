### 8.5



#### 28. Implement strStr()

https://leetcode.com/problems/implement-strstr/





#### 14. Longest Common Prefix

https://leetcode.com/problems/longest-common-prefix/



#### 58. Length of Last Word

https://leetcode.com/problems/length-of-last-word/





### 8.8

#### 383. Ransom Note

https://leetcode.com/problems/ransom-note/





#### 344. Reverse String

https://leetcode.com/problems/reverse-string/





#### 151. Reverse Words in a String

https://leetcode.com/problems/reverse-words-in-a-string/

- solution

  尝试O(1) space-not done
  
  

#### 186. Reverse Words in a String II

https://leetcode.com/problems/reverse-words-in-a-string-ii/



#### 345. Reverse Vowels of a String

https://leetcode.com/problems/reverse-vowels-of-a-string/



#### 293. Flip Game-easy

https://leetcode.com/problems/flip-game/



### 8.13

####  294. Flip Game II

https://leetcode.com/problems/flip-game-ii/

* Solution-minimax&backtracking

  ***worth thinking and doing***



### 8.21

#### 49. Group Anagrams

https://leetcode.com/problems/group-anagrams/



#### 249. Group Shifted Strings

https://leetcode.com/problems/group-shifted-strings/



### 8.22

#### 87. Scramble String

https://leetcode.com/problems/scramble-string/

* Solution-recursion

* Solution-DP

  ***worth doing and thinking***




### 8.26

#### 161. One Edit Distance

https://leetcode.com/problems/one-edit-distance/



#### 38. Count and Say

https://leetcode.com/problems/count-and-say/



### 8.29

#### 358. Rearrange String k Distance Apart

https://leetcode.com/problems/rearrange-string-k-distance-apart/

* solution-Priority queue



#### 316. Remove Duplicate Letters

https://leetcode.com/problems/remove-duplicate-letters/

* solution-Stack, greedy



#### 271. Encode and Decode Strings

https://leetcode.com/problems/encode-and-decode-strings/

* Solution-good thoughts



### 8.30

#### 168. Excel Sheet Column Title

https://leetcode.com/problems/excel-sheet-column-title/





#### 171. Excel Sheet Column Number

https://leetcode.com/problems/excel-sheet-column-number/

* Easy



#### 13. Roman to Integer

https://leetcode.com/problems/roman-to-integer/





### 9.14

#### 12. Integer to Roman

https://leetcode.com/problems/integer-to-roman/



#### 273. Integer to English Words

https://leetcode.com/problems/integer-to-english-words/

* solution

  感觉如下分组更好

  ```python
  def words(n):
          if n < 20:
              return to19[n-1:n]
          if n < 100:
              return [tens[n/10-2]] + words(n%10)
          if n < 1000:
              return [to19[n/100-1]] + ['Hundred'] + words(n%100)
          for p, w in enumerate(('Thousand', 'Million', 'Billion'), 1):
              if n < 1000**(p+1):
                  return words(n/1000**p) + [w] + words(n%1000**p)
      return ' '.join(words(num)) or 'Zero'
  ```









### 9.21

#### 247. Strobogrammatic Number II

https://leetcode.com/problems/strobogrammatic-number-ii/

* solution-recursive

  https://leetcode.com/problems/strobogrammatic-number-ii/discuss/67275/Python-recursive-solution-need-some-observation-so-far-97

  > Some observation to the sequence:
  >
  > n == 1: [0, 1, 8]
  >
  > n == 2: [11, 88, 69, 96]
  >
  > 
  >
  > How about n == `3`?
  > => it can be retrieved if you insert `[0, 1, 8]` to the middle of solution of n == `2`
  >
  > 
  >
  > n == `4`?
  > => it can be retrieved if you insert `[11, 88, 69, 96, 00]` to the middle of solution of n == `2`
  >
  > 
  >
  > n == `5`?
  > => it can be retrieved if you insert `[0, 1, 8]` to the middle of solution of n == `4`
  >
  > 
  >
  > the same, for n == `6`, it can be retrieved if you insert `[11, 88, 69, 96, 00]` to the middle of solution of n == `4`

  

#### 248. Strobogrammatic Number III

https://leetcode.com/problems/strobogrammatic-number-iii/



#### 157. Read N Characters Given Read4

https://leetcode.com/problems/read-n-characters-given-read4/



### 9.28

#### 158. Read N Characters Given Read4 II - Call multiple times

https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/

虽然题目烂，还是可以再做的



#### 68. Text Justification

https://leetcode.com/problems/text-justification/

* Solution-可以改进

  >  看了下，发现思想和自己也是一样的。但是这个速度却打败了 100% ，0 ms。考虑了下，差别应该在我的算法里使用了一个叫做 row 的 list 用来保存当前行的单词，用了很多 row.get ( index )，而上边的算法只记录了 left 和 right 下标，取单词直接用的 words 数组。然后尝试着在我之前的算法上改了一下，去掉 row，用两个变量 start 和 end 保存当前行的单词范围。主要是 ( end - start ) 代替了之前的 row.size ( )， words [ start + k ] 代替了之前的 row.get ( k )。
  >
  > 充分说明 list 的读取还是没有数组的直接读取快呀，还有就是要向上边的作者学习，多封装几个函数，思路会更加清晰，代码也会简明。
  >
  > https://leetcode.wang/leetCode-68-Text-Justification.html



### 12.13

#### 65. Valid Number

https://leetcode.com/problems/valid-number/

#### DFA(Deterministic Finite Automaton)

Link: https://leetcode.com/problems/valid-number/discuss/23728/A-simple-solution-in-Python-based-on-DFA

```python
class Solution(object):
  def isNumber(self, s):
      """
      :type s: str
      :rtype: bool
      """
      #define a DFA
      state = [{}, 
              {'blank': 1, 'sign': 2, 'digit':3, '.':4}, 
              {'digit':3, '.':4},
              {'digit':3, '.':5, 'e':6, 'blank':9},
              {'digit':5},
              {'digit':5, 'e':6, 'blank':9},
              {'sign':7, 'digit':8},
              {'digit':8},
              {'digit':8, 'blank':9},
              {'blank':9}]
      currentState = 1
      for c in s:
          if c >= '0' and c <= '9':
              c = 'digit'
          if c == ' ':
              c = 'blank'
          if c in ['+', '-']:
              c = 'sign'
          if c not in state[currentState].keys():
              return False
          currentState = state[currentState][c]
      if currentState not in [3,5,8,9]:
          return False
      return True
```



```java
interface NumberValidate {

	boolean validate(String s);
}

abstract class  NumberValidateTemplate implements NumberValidate{

public boolean validate(String s)
	{
		if (checkStringEmpty(s))
		{
			return false;
		}
		
		s = checkAndProcessHeader(s);
		
		if (s.length() == 0)
		{
			return false;
		}
		
		return doValidate(s);
	}
	
	private boolean checkStringEmpty(String s)
	{
		if (s.equals(""))
		{
			return true;
		}
		
		return false;
	}
	
	private String checkAndProcessHeader(String value)
	{
	    value = value.trim();
	    
		if (value.startsWith("+") || value.startsWith("-"))
		{
			value = value.substring(1);
		}
	
	
		return value;
	}
	
	
	
	protected abstract boolean doValidate(String s);
}

class NumberValidator implements NumberValidate {
	
	private ArrayList<NumberValidate> validators = new ArrayList<NumberValidate>();
	
	public NumberValidator()
	{
		addValidators();
	}

	private  void addValidators()
	{
		NumberValidate nv = new IntegerValidate();
		validators.add(nv);
		
		nv = new FloatValidate();
		validators.add(nv);
		
		nv = new HexValidate();
		validators.add(nv);
		
		nv = new SienceFormatValidate();
		validators.add(nv);
	}
	
	@Override
	public boolean validate(String s)
	{
		for (NumberValidate nv : validators)
		{
			if (nv.validate(s) == true)
			{
				return true;
			}
		}
		
		return false;
	}

	
}

class IntegerValidate extends NumberValidateTemplate{
	
	protected boolean doValidate(String integer)
	{
		for (int i = 0; i < integer.length(); i++)
		{
			if(Character.isDigit(integer.charAt(i)) == false)
			{
				return false;
			}
		}
		
		return true;
	}
}

class HexValidate extends NumberValidateTemplate{

	private char[] valids = new char[] {'a', 'b', 'c', 'd', 'e', 'f'};
	protected boolean doValidate(String hex)
	{
		hex = hex.toLowerCase();
		if (hex.startsWith("0x"))
		{
			hex = hex.substring(2);
		}
		else
		{
		    return false;
		}
		
		for (int i = 0; i < hex.length(); i++)
		{
			if (Character.isDigit(hex.charAt(i)) != true && isValidChar(hex.charAt(i)) != true)
			{
				return false;
			}
		}
		
		return true;
	}
	
	private boolean isValidChar(char c)
	{
		for (int i = 0; i < valids.length; i++)
		{
			if (c == valids[i])
			{
				return true;
			}
		}
		
		return false;
	}
}

class SienceFormatValidate extends NumberValidateTemplate{

protected boolean doValidate(String s)
	{
		s = s.toLowerCase();
		int pos = s.indexOf("e");
		if (pos == -1)
		{
			return false;
		}
		
		if (s.length() == 1)
		{
			return false;
		}
		
		String first = s.substring(0, pos);
		String second = s.substring(pos+1, s.length());
		
		if (validatePartBeforeE(first) == false || validatePartAfterE(second) == false)
		{
			return false;
		}
		
		
		return true;
	}
	
	private boolean validatePartBeforeE(String first)
	{
		if (first.equals("") == true)
		{
			return false;
		}
		
		if (checkHeadAndEndForSpace(first) == false)
		{
			return false;
		}
		
		NumberValidate integerValidate = new IntegerValidate();
		NumberValidate floatValidate = new FloatValidate();
		if (integerValidate.validate(first) == false && floatValidate.validate(first) == false)
		{
			return false;
		}
		
		return true;
	}
	
private boolean checkHeadAndEndForSpace(String part)
	{
		
		if (part.startsWith(" ") ||
				part.endsWith(" "))
		{
			return false;
		}
		
		return true;
	}
	
	private boolean validatePartAfterE(String second)
	{
		if (second.equals("") == true)
		{
			return false;
		}
		
		if (checkHeadAndEndForSpace(second) == false)
		{
			return false;
		}
		
		NumberValidate integerValidate = new IntegerValidate();
		if (integerValidate.validate(second) == false)
		{
			return false;
		}
		
		return true;
	}
}

class FloatValidate extends NumberValidateTemplate{
	
   protected boolean doValidate(String floatVal)
	{
		int pos = floatVal.indexOf(".");
		if (pos == -1)
		{
			return false;
		}
		
		if (floatVal.length() == 1)
		{
			return false;
		}
		
		NumberValidate nv = new IntegerValidate();
		String first = floatVal.substring(0, pos);
		String second = floatVal.substring(pos + 1, floatVal.length());
		
		if (checkFirstPart(first) == true && checkFirstPart(second) == true)
		{
			return true;
		}
		
		return false;
	}
	
	private boolean checkFirstPart(String first)
	{
	    if (first.equals("") == false && checkPart(first) == false)
	    {
	    	return false;
	    }
	    
	    return true;
	}
	
	private boolean checkPart(String part)
	{
	   if (Character.isDigit(part.charAt(0)) == false ||
				Character.isDigit(part.charAt(part.length() - 1)) == false)
		{
			return false;
		}
		
		NumberValidate nv = new IntegerValidate();
		if (nv.validate(part) == false)
		{
			return false;
		}
		
		return true;
	}
}

public class Solution {
    public boolean isNumber(String s) {
        NumberValidate nv = new NumberValidator();

	    return nv.validate(s);
    }
}
```





## 语法

#### split()

不管中间空格有几个都可以分组



```python
# Get the ASCII number of a character
number = ord(char)

# Get the character given by an ASCII number
char = chr(number)
```





#### collections.defaultdict

Usually, a Python dictionary throws a `KeyError` if you try to get an item with a key that is not currently in the dictionary. The `defaultdict` in contrast will simply create any items that you try to access (provided of course they do not exist yet). To create such a "default" item, it calls the function object that you pass to the constructor (more precisely, it's an arbitrary "callable" object, which includes function and type objects). 



#### find()

```python
str.find(sub[, start[, end]] )
```

- Parameter

  * **sub** - It's the substring to be searched in the str string.

  * **start** and **end** (optional) - substring is searched within `str[start:end]`

- It returns value: 

  * If substring exists inside the string, it returns the index of first occurence of the substring.
  * If substring doesn't exist inside the string, it returns -1.

  