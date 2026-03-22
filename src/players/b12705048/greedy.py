class Minimizer():
    def __init__(self, player_idx):
        self.player_idx = player_idx
    
    def action(self, hand, history):
        return min(hand)

class Maximizer():
    def __init__(self, player_idx):
        self.player_idx = player_idx
    
    def action(self, hand, history):
        return max(hand)
