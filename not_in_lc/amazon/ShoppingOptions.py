class Solution:
    def getNumberOfOptions(self, priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, dollars):
        jeansShoes = self.combine(priceOfJeans, priceOfShoes, dollars)
        skirtsTops = self.combine(priceOfSkirts, priceOfTops, dollars)
        options = 0

        jeansShoes.sort()
        skirtsTops.sort()

        left = 0
        right = len(skirtsTops) - 1
        while left < len(jeansShoes) and right >= 0:
            left_price = jeansShoes[left]
            right_price = skirtsTops[right]
            if left_price + right_price <= dollars:
                options += right + 1
                left += 1
            else:
                right -= 1

        return options

    def combine(self, list1, list2, dollar):
        result = []
        for price1 in list1:
            for price2 in list2:
                if price1 + price2 < dollar: # I think we don't need equal here?
                    result.append(price1+price2)

        return result


if __name__ == '__main__':
    solution = Solution()
    priceOfJeans = [2, 3]
    priceOfShoes = [4]
    priceOfSkirts = [2, 3]
    priceOfTops = [1, 2]
    budgeted = 10
    print(solution.getNumberOfOptions(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, budgeted)) # should be 4
