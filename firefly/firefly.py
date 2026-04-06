import math
import random

class Firefly:
    def __init__(self, metrics, weights, stock_tickers, budget, bounds=(0.0, 1.0), position=None, top_n=None):
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
        self.brightness = 0.0
        default_top_n = max(3, min(10, len(self.stock_tickers) // 3 if len(self.stock_tickers) >= 3 else len(self.stock_tickers)))
        self.top_n = top_n if top_n is not None else default_top_n
        
        # The position is purely a random spatial [x, y] coordinate
        self.position = list(position) if position is not None else [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])]
        
        # Generate and store the portfolio from the top-ranked subset automatically.
        self.portfolio = self._generate_top_portfolio(top_n=self.top_n)

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
            
            position_signal = math.sin((i + 1) * self.position[0] * math.pi) + math.cos((i + 1) * self.position[1] * math.pi)
            base_score = (0.65 * m) + (0.35 * w)
            score = base_score + (0.02 * position_signal)
            stock_scores.append((self.stock_tickers[i], score))
            
        # Sort stocks by their calculated score in descending order
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Isolate the top N stocks
        top_stocks = stock_scores[:top_n]
        
        # Calculate total score from the top stocks to normalize the allocations
        positive_scores = [max(score, 0) for ticker, score in top_stocks]
        total_score = sum(positive_scores)
        avg_positive_score = (sum(positive_scores) / len(positive_scores)) if positive_scores else 0.0

        # Let the algorithm hold cash in weak signal regimes instead of forcing full investment.
        exposure = min(1.0, max(0.2, avg_positive_score / 0.75))
        
        # Normalize the scores so they act as percentage weights that sum to 1
        normalized_portfolio = []
        for i, (ticker, score) in enumerate(top_stocks):
            p_score = positive_scores[i]
            normalized_weight = ((p_score / total_score) * exposure) if total_score > 0 else 0
            normalized_portfolio.append((ticker, normalized_weight))
            
        return normalized_portfolio

    def get_list(self):
        return self.portfolio

    def rebuild_portfolio(self, top_n=None):
        chosen_top_n = self.top_n if top_n is None else top_n
        self.portfolio = self._generate_top_portfolio(top_n=chosen_top_n)
        return self.portfolio
