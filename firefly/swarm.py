from firefly import Firefly
from movement import calculate_movement
from brightness import calculate_brightness

def run_swarm(
    num_fireflies, 
    iterations, 
    metrics, 
    weights, 
    stock_tickers, 
    budget, 
    bounds=(0.0, 1.0)
):
    """
    Orchestrates the Firefly algorithm over a given number of iterations.
    
    Args:
        num_fireflies (int): The number of fireflies in the swarm.
        iterations (int): The number of iterations to run.
        metrics (list): Metric data for the stock universe.
        weights (list): Weightings for those metrics.
        stock_tickers (list): The list of stock tickers.
        budget (float): The budget for the portfolios.
        bounds (tuple): Coordinate boundaries for the firefly [x, y] spawn mapping.
        
    Returns:
        tuple: (best_firefly_object, best_portfolio_strategy)
    """
    
    # 1. Initialize the entire swarm of fireflies
    swarm = [Firefly(metrics, weights, stock_tickers, budget, bounds) for _ in range(num_fireflies)]
        
    # Variables tracking the best global firefly performance
    best_firefly = None
    best_brightness_score = float('-inf')
    
    # 2. Iteration Loop
    for _ in range(iterations):
        for i in range(num_fireflies):
            # Calculate brightness/fitness for firefly i using the external function
            brightness_i = calculate_brightness(swarm[i])
            
            # Check if this is the new global best
            if brightness_i > best_brightness_score:
                best_brightness_score = brightness_i
                best_firefly = swarm[i]
            
            # Compare performance with other fireflies in the swarm
            for j in range(num_fireflies):
                if i != j:
                    brightness_j = calculate_brightness(swarm[j])
                    
                    # If firefly j is brighter than firefly i, firefly i moves towards firefly j
                    if brightness_j > brightness_i:
                        # Call the external movement function to get new [x, y] coordinates
                        new_coordinates = calculate_movement(swarm[i], swarm[j])
                        
                        # Update firefly i's spatial position
                        swarm[i].position = new_coordinates
                        
    # 3. Output the best strategy found over all iterations
    # Returns the actual Firefly object alongside its specific tuple array (the portfolio strategy)
    return best_firefly, best_firefly.portfolio
