"""
TODO

1. from stock list
2. A) Use either a math.comb for best performing mixes by brute forcing (defined by a rudimentary filterof eps, pe ratio, dividend yield))
3. Return the selected stocks in a usable format (tuple)
"""
import itertools
import math

def combination_stock_selector(stocks, stocks_data, num):
    def evaluate_performance(comb):
        score = 0
        for stock in comb:
            #This metric is very rudimentary and only serves as a baseline
            score += stocks_data[stock]['eps'] * stocks_data[stock]['pe_ratio'] * stocks_data[stock]['dividend_yield']
        return score

    num_stocks = num
    max_perf = 0
    best_comb = None
    for i in range(1, num_stocks + 1):
        for comb in itertools.combinations(stocks, i):
            perf = evaluate_performance(comb)
            if perf > max_perf:
                max_perf = perf
                best_comb = comb
    return best_comb