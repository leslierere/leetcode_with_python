、建议找个视频食用

### 121. Best Time to Buy and Sell Stock

- At most one transaction

- 直观一点就是，找到一个最小值，然后找到这个值后的最大值，计算当前max，再和res进行取大；继续遍历。

  ```java
  public int maxProfit(int[] prices) {
      int res = 0, min = prices[0];
      for(int price : prices) {
          if(price > min) {
              res = Math.max(res, price - min);
          }else {
              min = price;
          }
      }
      return res;
  }
  ```

- 复杂一点：前一天买，当天卖，连续即可看做`i`天买，`j`天卖；current 取`Math.max(0, cur+profit)`

  ```java
  public int maxProfit(int[] prices) {
      int res = 0, cur = 0;
      for(int i = 1; i < prices.length; i++) {
          cur = Math.max(0, cur + prices[i] - prices[i-1]);
          res = Math.max(cur, res);
      }
      return res;
  }
  ```

  

### 122. Best Time to Buy and Sell Stock II

- Many transactions, sell the stock before you buy again

- 朴素来想，把所有的正向差值都积累起来（和1的思路2相似）

  ```java
  public int maxProfit(int[] prices) {
      int res = 0, cur = 0;
      for(int i = 1; i < prices.length; i++) {
          
      }
      return res;
  }
  ```
  
- dp: 买入状态的最大profit & 卖出状态的最大profit -- ***<u>dp的类型，维护两个array</u>***

  - buy -> sell
  - 当前是买入状态的max profit
  - 当前是卖出状态的max profit
  - 递推关系为何成立
  
  ```java
  public int maxProfit(int[] prices) {
      if(prices.length < 2) return 0;
      int buy = -prices[0];
      int sell = 0;
      int res = 0;
      for(int i = 1; i < prices.length; i++) {
          buy = Math.max(buy, sell - prices[i]);
          sell = Math.max(sell, buy + prices[i]);
          res = Math.max(res, sell);
      }
      return res;
  }
  ```
  
  

### 123. Best Time to Buy and Sell Stock III

- At most two transactions
- Trick ✨
  - 分成两段分别计算！机智啊！
  - 一个是起点固定，一个是终点固定。
    - 先从左往右，记录下每个到位置的最大值是多少
    - 再从右往左，算出到当前位置的最大值和上面的相加计算全局最大值

### 188. Best Time to Buy and Sell Stock IV

- At most k transactions
- `k > n / 2`, then you can make maximum number of transactions.
- `dp[n][k]`: n 天 k 次操作
  - 当天一定有操作 & 全局 max
  - `local[n][k]= Math.max(global[n-1][k-1], local[n-1][k]) + diff`
    - 可以选择昨天买今天卖
    - 或者是很久之前买今天卖，相当于连续买卖，所以交易次数不变  `local[n-1][k]`
  - `global[n][k] = Math.max(global[n-1][k], local[n][k])`
- 优化成一维: `dp[k]` 当前天最多交易k次的最大值
  - `local[k]= global[k-1] + diff`
  - 因为`local[k]用到了global[k-1]`，所以需要暂时保存一下，不能后面直接改动
  - `global[k] = Math.max(global[k], local[k])`

### 309. Best Time to Buy and Sell Stock with Cooldown

- 122-dp的基础上再加一个新的状态`cool`：当前维持不变的最大收益
- `buy` 在计算时需要使用cool来进行递推，因为前一个sell不能直接接上buy

### 714. Best Time to Buy and Sell Stock with Transaction Fee

- 类似122 - dp思路，sell的时候扣除手续费即可

