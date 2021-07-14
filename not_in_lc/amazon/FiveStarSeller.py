import heapq
# precision problem
class Solution:
    def fiveStarReviews(self, productCount, productRatings, threshold):
        threshold = threshold/100
        total_score = 0
        output = 0
        reviews = []
        heapq.heapify(reviews)
        for five_star, total in productRatings:
            score = five_star/total
            total_score += score
            increment = -((five_star+1)/(total+1) - score)
            heapq.heappush(reviews, (increment, five_star, total))

        while total_score < threshold*productCount:
            increment, five_star, total = heapq.heappop(reviews)
            total_score -= increment
            five_star += 1
            total += 1
            increment = -((five_star + 1) / (total + 1) - five_star / total)
            heapq.heappush(reviews, (increment, five_star, total))
            output += 1

        return output


if __name__ == '__main__':
    solution = Solution()
    productCount = 3
    productRatings = [[4, 4], [1, 2], [3, 6]]
    threshold = 77
    print(solution.fiveStarReviews(productCount, productRatings, threshold)) # should be 3