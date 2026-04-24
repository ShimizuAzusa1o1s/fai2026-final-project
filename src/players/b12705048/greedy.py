class Minimizer():
    """
    A simple greedy agent that always selects the smallest card in its hand.
    """
    def __init__(self, player_idx):
        self.player_idx = player_idx
    
    def action(self, hand, history):
        """
        Returns the minimum value card from the current hand.
        """
        return min(hand)

class Maximizer():
    """
    A simple greedy agent that always selects the largest card in its hand.
    """
    def __init__(self, player_idx):
        self.player_idx = player_idx
    
    def action(self, hand, history):
        """
        Returns the maximum value card from the current hand.
        """
        return max(hand)
