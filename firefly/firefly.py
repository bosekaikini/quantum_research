import random

class Firefly:
    def __init__(self, metrics, weights, stock_tickers, budget, bounds=(0.0, 1.0)):
        """
        Initialize the Firefly.
        The top stocks are selected using the provided metrics and weights.
        The position is a completely random 2D spatial coordinate: [x, y].
        """
        self.metrics = metrics
        self.weights = weights
        self.stock_tickers = stock_tickers
        self.budget = budget
        self.bounds = bounds
        
        # The position is purely a random spatial [x, y] coordinate
        self.position = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])]
        
        # Generate and store the portfolio of top 10 stocks automatically
        self.portfolio = self._generate_top_portfolio(top_n=10)

    def _generate_top_portfolio(self, top_n=10):
        """
        Calculates the top 'n' stocks decided by the product of their metric and weighting.
        The metric scores for these top stocks are normalized to determine their final portfolio weights.
        Returns a list of tuples: (string, float) representing the stock ticker and its weight.
        """
        stock_scores = []
        # Calculate score for each stock from metrics and weights
        for i in range(len(self.stock_tickers)):
            # Default to 0 if lists are uneven
            m = self.metrics[i] if i < len(self.metrics) else 0
            w = self.weights[i] if i < len(self.weights) else 0
            
            score = m * w
            stock_scores.append((self.stock_tickers[i], score))
            
        # Sort stocks by their calculated score in descending order
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Isolate the top N stocks
        top_stocks = stock_scores[:top_n]
        
        # Calculate total score from the top stocks to normalize the allocations
        positive_scores = [max(score, 0) for ticker, score in top_stocks]
        total_score = sum(positive_scores)
        
        # Normalize the scores so they act as percentage weights that sum to 1
        normalized_portfolio = []
        for i, (ticker, score) in enumerate(top_stocks):
            p_score = positive_scores[i]
            normalized_weight = (p_score / total_score) if total_score > 0 else 0
            normalized_portfolio.append((ticker, normalized_weight))
            
        return normalized_portfolio
