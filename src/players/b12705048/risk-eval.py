class RiskEvaluation():
    def __init__(self, player_idx=0):
        self.player_idx = player_idx
        # Step 0: Precomputation
        self.seen_cards = [False] * 105
        self.comb_table = [[0] * 105 for _ in range(105)]
        for n in range(105):
            self.comb_table[n][0] = 1
            for k in range(1, n + 1):
                self.comb_table[n][k] = self.comb_table[n-1][k-1] + self.comb_table[n-1][k]

    def get_comb(self, n, k):
        if k < 0 or k > n or n < 0 or n >= 105:
            return 0
        return self.comb_table[n][k]

    def get_bullheads(self, card):
        if card % 55 == 0: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def action(self, hand, history):
        # Step 1: State Update
        # Re-initialize for the current state to be secure across game boundaries
        self.seen_cards = [False] * 105
        
        # Mark cards in hand as seen
        for c in hand:
            self.seen_cards[c] = True
            
        # Parse history to identify newly played cards and board
        board = history.get('board', [])
        for row in board:
            for c in row:
                self.seen_cards[c] = True
                
        history_matrix = history.get('history_matrix', [])
        for round_actions in history_matrix:
            for c in round_actions:
                self.seen_cards[c] = True
                
        N = sum(1 for c in range(1, 105) if not self.seen_cards[c])
        scores = history.get('scores', [])
        num_players = len(scores) if scores else 4
        opponents_cards = (num_players - 1) * len(hand)

        # Step 2: Board Evaluation
        lowest_bullhead = float('inf')
        row_stats = []
        
        for row in board:
            bh = sum(self.get_bullheads(c) for c in row)
            L_t = len(row)
            row_stats.append({
                'end_card': row[-1],
                'length': L_t,
                'bullheads': bh
            })
            if bh < lowest_bullhead:
                lowest_bullhead = bh

        # Step 3: Card Evaluation Loop
        best_card = None
        min_expected_penalty = float('inf')

        for c in hand:
            # Find Target Row
            target_row = None
            max_end = -1
            
            for r in row_stats:
                if r['end_card'] < c and r['end_card'] > max_end:
                    max_end = r['end_card']
                    target_row = r
                    
            if target_row is None:
                # Evaluate Low Card
                cost = lowest_bullhead
            else:
                # Evaluate Normal Placement
                k = 5 - target_row['length']
                e_t = target_row['end_card']
                
                # Scan unseen cards between e_t and c
                d = sum(1 for x in range(e_t + 1, c) if not self.seen_cards[x])
                
                if d < k:
                    # Short-circuit optimization: 100% safe
                    return c
                else:
                    total_ways = self.get_comb(N, opponents_cards)
                    P_break = 0.0
                    
                    if total_ways > 0:
                        for i in range(k, min(d, opponents_cards) + 1):
                            ways = self.get_comb(d, i) * self.get_comb(N - d, opponents_cards - i)
                            P_break += ways / total_ways
                            
                    cost = P_break * target_row['bullheads']

            # Update Minimum
            if cost < min_expected_penalty:
                min_expected_penalty = cost
                best_card = c
                
        # Step 4: Execute Action
        return best_card
