import math
import random

class RiskEvaluation():
    def __init__(self, player_idx=0):
        self.player_idx = player_idx

    def calculate_p_saved(self, d_under, N_unseen, n_opponents=3):
        n = min(n_opponents, N_unseen)
        if d_under == 0:
            return 0.0
        
        p_zero = 1.0
        for i in range(n):
            if N_unseen - i <= 0:
                break
            p_zero *= max(0, (N_unseen - d_under - i)) / (N_unseen - i)
            
        return 1.0 - p_zero

    def get_bullheads(self, card):
        if card == 55: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def select_action_softmax(self, stats_dict, temperature=1.0, top_k=3):
        # Top-K approach
        sorted_stats = sorted(stats_dict.items(), key=lambda item: item[1])
        top_k_stats = dict(sorted_stats[:top_k])
        
        cards = list(top_k_stats.keys())
        costs = list(top_k_stats.values())
        
        # 1. Apply negative temperature scaling
        # We subtract the min cost before exp() for numerical stability 
        # (prevents math overflow errors if costs are large)
        min_cost = min(costs)
        scaled_values = [-(c - min_cost) / temperature for c in costs]
        
        # 2. Calculate Exponentials
        exps = [math.exp(v) for v in scaled_values]
        sum_exps = sum(exps)
        
        # 3. Create Probability Distribution
        probabilities = [e / sum_exps for e in exps]
        
        # 4. Roulette Wheel Selection
        r = random.random()
        cumulative = 0.0
        for card, prob in zip(cards, probabilities):
            cumulative += prob
            if r <= cumulative:
                return card
                
        return cards[-1] # Fallback

    def action(self, hand, history):
        scores = history.get('scores', [])
        num_players = len(scores) if scores else 4
        opponents = num_players - 1
        
        seen_cards = set()
        
        for c in hand:
            seen_cards.add(c)
            
        board = history.get('board', [])
        row_stats = []
        for i, row in enumerate(board):
            for c in row:
                seen_cards.add(c)
            bh = sum(self.get_bullheads(c) for c in row)
            row_stats.append({
                'index': i,
                'end_card': row[-1],
                'length': len(row),
                'bullheads': bh
            })
            
        history_matrix = history.get('history_matrix', [])
        for round_actions in history_matrix:
            for c in round_actions:
                seen_cards.add(c)
                
        N = 104 - len(seen_cards)
        
        card_costs = {}
        
        for c in hand:
            target_row = None
            max_end = -1
            
            for r in row_stats:
                if r['end_card'] < c and r['end_card'] > max_end:
                    max_end = r['end_card']
                    target_row = r
                    
            if target_row is None:
                sorted_rows = sorted(row_stats, key=lambda x: (x['bullheads'], x['length'], x['index']))
                min_bulls = sorted_rows[0]['bullheads']
                
                unseen_cards = [u for u in range(1, 105) if u not in seen_cards]
                d_under = sum(1 for u in unseen_cards if u < c)
                p_saved = self.calculate_p_saved(d_under, N, opponents)
                
                current_cost = (1.0 - p_saved) * min_bulls
            else:
                L_t = target_row['length']
                e_t = target_row['end_card']
                R_t_bh = target_row['bullheads']
                
                if L_t == 5:
                    current_cost = R_t_bh
                else:
                    k_req = 5 - L_t
                    if k_req > opponents:
                        current_cost = 0.0
                    else:
                        d = 0
                        for x in range(e_t + 1, c):
                            if x <= 104 and x not in seen_cards:
                                d += 1
                                
                        P_break = 0.0
                        total_ways = math.comb(N, opponents)
                        if total_ways > 0:
                            for m in range(k_req, opponents + 1):
                                if d >= m and (N - d) >= (opponents - m):
                                    P_break += (math.comb(d, m) * math.comb(N - d, opponents - m)) / total_ways
                                    
                        current_cost = P_break * R_t_bh
                        
            card_costs[c] = current_cost
                
        return self.select_action_softmax(card_costs)
