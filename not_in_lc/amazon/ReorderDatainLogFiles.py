class Solution:
    def reorderLogFiles(self, logs):
        letter_logs = []
        digit_logs = []

        for log in logs:
            if self.ifLetterLog(log):
                letter_logs.append(log)
            else:
                digit_logs.append(log)

        letter_logs.sort(key=lambda x: (x.split()[1:], x.split()[0]))

        return letter_logs + digit_logs

    def ifLetterLog(self, log):
        first_space = log.find(" ")
        first_char = log[first_space + 1]
        return first_char.isalpha()

    def ifPrime(self, number):
        if number < 2:
            return False

        n_square = int(number**0.5)
        primes = [True for i in range(number+1)]

        for factor in range(2, n_square+1):
            if primes[factor]:
                value = factor * 2
                while value < number:
                    primes[value] = False
                    value += factor

                if value == number:
                    primes[value] = False
                    break
        return primes[number]

if __name__ == '__main__':
    solution = Solution()
    print(solution.ifPrime(0))
    print(solution.ifPrime(1))
    print(solution.ifPrime(2))
    print(solution.ifPrime(7))
    print(solution.ifPrime(9))