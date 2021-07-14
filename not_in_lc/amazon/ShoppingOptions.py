class Solution:
    def getNumberOfOptions(self, priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, dollars):
        priceOfJeans.sort()
        priceOfShoes.sort()
        priceOfSkirts.sort()
        priceOfTops.sort()
        options = 0

        for jean in priceOfJeans:
            if jean > dollars:
                break
            value1 = dollars - jean
            for shoe in priceOfShoes:
                if shoe > value1:
                    break
                value2 = value1 - shoe
                for skirt in priceOfSkirts:
                    if skirt > value2:
                        break
                    value3 = value2 - skirt
                    for top in priceOfTops:
                        if top > value3:
                            break
                        options += 1

        return options

if __name__ == '__main__':
    solution = Solution()
    priceOfJeans = [2, 3]
    priceOfShoes = [4]
    priceOfSkirts = [2, 3]
    priceOfTops = [1, 2]
    budgeted = 10
    print(solution.getNumberOfOptions(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, budgeted)) # should be 4
