import math

class RiskEvaluation():
    def __init__(self, player_idx=0):
        self.player_idx = player_idx

    def get_bullheads(self, card):
        if card == 55: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

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
        
        best_card = None
        min_cost = float('inf')
        
        for c in hand:
            target_row = None
            max_end = -1
            
            for r in row_stats:
                if r['end_card'] < c and r['end_card'] > max_end:
                    max_end = r['end_card']
                    target_row = r
                    
            if target_row is None:
                sorted_rows = sorted(row_stats, key=lambda x: (x['bullheads'], x['length'], x['index']))
                current_cost = sorted_rows[0]['bullheads']
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
                        
            if current_cost < min_cost:
                min_cost = current_cost
                best_card = c
                
        return best_card
