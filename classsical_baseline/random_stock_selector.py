"""
TODO:

1. Get trading data
2. Choose random from tradadable universe (random number as well)
"""
#MAX Stock is 5
import random
def random_stock_selector(stocks: list[str], stocks_data, prev_selected_stocks: list[str]):

    MAX_STOCKS = 5
    MIN_STOCKS = 1

    if len(prev_selected_stocks) == 1:
        weightToKeep = 0.8
    elif len(prev_selected_stocks) == 2:
        weightToKeep = 0.6
    elif len(prev_selected_stocks) == 3:
        weightToKeep = 0.4
    elif len(prev_selected_stocks) == 4:
        weightToKeep = 0.3
    elif len(prev_selected_stocks) == 5:
        weightToKeep = 0.2

    

    num_stocks = random.randint(MIN_STOCKS, MAX_STOCKS)

    selected_stocks = []
    for n in range(num_stocks):
        if random.random() < weightToKeep:
            selected_stocks.append(random.choice(prev_selected_stocks))
            prev_selected_stocks.remove(selected_stocks[-1])
        else:
            random_stock = random.choice(stocks)
            stocks.remove(random_stock) #no duplicates
            selected_stocks.append(random_stock)


    
    return selected_stocks


    
    





    
    