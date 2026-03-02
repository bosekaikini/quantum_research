"""
TODO:

1. Get trading data
2. Choose random from tradadable universe (random number as well)
"""
#MAX Stock is 5
import random
def random_stock_selector(stocks: list[str], stocks_data, prev_selected_stocks: list[str]) -> tuple[str, ...]:

    MAX_STOCKS = 5
    MIN_STOCKS = 1

    weightToKeep = 0.0
    if len(prev_selected_stocks) == 1:
        weightToKeep = 0.8
    elif len(prev_selected_stocks) == 2:
        weightToKeep = 0.6
    elif len(prev_selected_stocks) == 3:
        weightToKeep = 0.4
    elif len(prev_selected_stocks) == 4:
        weightToKeep = 0.3
    elif len(prev_selected_stocks) >= 5:
        weightToKeep = 0.2

    # Convert to lists to allow .remove() in case tuples are passed in,
    # and to avoid mutating the original passed lists.
    available_stocks = list(stocks)
    available_prev = list(prev_selected_stocks)

    num_stocks = random.randint(MIN_STOCKS, MAX_STOCKS)

    selected_stocks = []
    for n in range(num_stocks):
        if random.random() < weightToKeep and available_prev:
            chosen = random.choice(available_prev)
            selected_stocks.append(chosen)
            available_prev.remove(chosen)
        else:
            if available_stocks:
                random_stock = random.choice(available_stocks)
                available_stocks.remove(random_stock) #no duplicates
                selected_stocks.append(random_stock)

    return tuple(selected_stocks)


    
    





    
    