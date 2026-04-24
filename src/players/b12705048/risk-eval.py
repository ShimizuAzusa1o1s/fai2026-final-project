import math
import random

class RiskEvaluation():
    """
    A rule-based heuristic agent that assesses the probability of getting penalized.
    It calculates the risk associated with each card considering the likelihood that 
    opponents might under-cut or trigger a 6th-card penalty.
    """
    def __init__(self, player_idx=0):
        self.player_idx = player_idx

    def calculate_p_saved(self, p_under, n_opponents=3):
        """
        Calculates the probability that at least one opponent plays a card 
        lower than our proposed undercut, effectively 'saving' us from taking a row.
        
        Args:
            p_under: probability that an unseen card drawn by opponent is smaller than our card
            n_opponents: number of other players
            
        Returns:
            Probability of being saved from undercut penalty.
        """
        # Using binomial approximation since weights are continuous
        p_zero = (1.0 - p_under) ** n_opponents
        return 1.0 - p_zero

    def get_bullheads(self, card):
        if card == 55: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def select_action_softmax(self, stats_dict, temperature=1.0, top_k=3):
        """
        Samples an action using a Softmax strategy over the top K least risky choices.
        
        Args:
            stats_dict: dict mapping 'card' to 'computed risk or cost'
            temperature: controls how much randomness vs greediness is injected
            top_k: restricts consideration to only the top K safest variations
            
        Returns:
            The selected card.
        """
        # Top-K approach: select the K entries with lowest estimated penalty costs.
        sorted_stats = sorted(stats_dict.items(), key=lambda item: item[1])
        top_k_stats = dict(sorted_stats[:top_k])
        
        cards = list(top_k_stats.keys())
        costs = list(top_k_stats.values())
        
        # 1. Apply negative temperature scaling
        # We subtract the min cost before exp() for numerical stability 
        # (prevents math overflow errors if costs are large)
        min_cost = min(costs)
        scaled_values = [-(c - min_cost) / temperature for c in costs]
        
        # 2. Calculate Exponentials to map costs to weights
        exps = [math.exp(v) for v in scaled_values]
        sum_exps = sum(exps)
        
        # 3. Create Probability Distribution normalizing weights
        probabilities = [e / sum_exps for e in exps]
        
        # 4. Roulette Wheel Selection based on weights
        r = random.random()
        cumulative = 0.0
        for card, prob in zip(cards, probabilities):
            cumulative += prob
            if r <= cumulative:
                return card
                
        return cards[-1] # Fallback in case of precision issues

    def compute_card_weight(self, card, row_stats, min_bulls):
        target_row = None
        max_end = -1
        for r in row_stats:
            if r['end_card'] < card and r['end_card'] > max_end:
                max_end = r['end_card']
                target_row = r
                
        if target_row is None:
            # Under-cut: Forced to take the row with the fewest bullheads
            return 1.0 / (min_bulls / 5.0 + 1.0)
            
        if target_row['length'] == 5:
            # Forced play: Rational opponents avoid it, but might be cornered
            return 0.05
            
        # Tight Plays & Bullhead Dump: Opponents prefer smaller gaps and dumping high bullhead cards
        gap = card - target_row['end_card']
        base_weight = max(0.2, 5.0 / max(1, gap))
        bullhead_multiplier = 1.0 + 0.2 * self.get_bullheads(card)
        return base_weight * bullhead_multiplier

    def action(self, hand, history):
        """
        Analyzes the current board to assign each card in hand an expected penalty structure, and then returns an optimal probabilistic action.
        """
        scores = history.get('scores', [])
        num_players = len(scores) if scores else 4
        opponents = num_players - 1
        
        seen_cards = set()
        
        # 1. Track all cards currently in hand to build the 'seen' set
        for c in hand:
            seen_cards.add(c)
            
        # 2. Extract board state: track seen cards and compute stats for each row
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
            
        # 3. Process past rounds to register all previously played cards
        history_matrix = history.get('history_matrix', [])
        for round_actions in history_matrix:
            for c in round_actions:
                seen_cards.add(c)
                
        N = 104 - len(seen_cards)
        
        # 4. Rational Opponent Modeling: Calculate continuous weights for the unseen cards.
        # We assume rational opponents avoid plays that incur high penalties.
        # First, find the row with the fewest bullheads globally to scale risks for 'undercut' cards.
        sorted_rows_global = sorted(row_stats, key=lambda x: (x['bullheads'], x['length'], x['index']))
        min_bulls_global = sorted_rows_global[0]['bullheads']

        unseen_cards_list = [u for u in range(1, 105) if u not in seen_cards]
        weights = {u: self.compute_card_weight(u, row_stats, min_bulls_global) for u in unseen_cards_list}
        
        # N_eff represents the effective total mass of unseen cards, discounting unlikely plays
        N_eff = sum(weights.values())
        if N_eff == 0:
            N_eff = 1.0  # fallback
        
        # 4.5 Calculate Chaos Penalty (Probability of ANY opponent undercutting)
        min_end_global = min(r['end_card'] for r in row_stats)
        d_under_global = sum(weights[u] for u in unseen_cards_list if u < min_end_global)
        p_under_global = min(d_under_global / N_eff, 1.0)
        p_chaos = 1.0 - ((1.0 - p_under_global) ** opponents)
        
        card_costs = {}
        
        # 5. Evaluate the expected penalty for each card in the player's hand
        for c in hand:
            target_row = None
            max_end = -1
            
            # Find the row this card is nominally supposed to go into
            for r in row_stats:
                if r['end_card'] < c and r['end_card'] > max_end:
                    max_end = r['end_card']
                    target_row = r
                    
            if target_row is None:
                # Scenario A: The card is smaller than all row end-cards (an 'undercut').
                # Our base penalty is taking the row with the minimum bullheads.
                sorted_rows = sorted(row_stats, key=lambda x: (x['bullheads'], x['length'], x['index']))
                min_bulls = sorted_rows[0]['bullheads']
                
                # Effectively count how many unseen smaller cards exist based on opponent rationality
                d_under_eff = sum(weights[u] for u in unseen_cards_list if u < c)
                p_under = min(d_under_eff / N_eff, 1.0)
                
                # Probability that we are saved by an opponent playing an even smaller card
                p_saved = self.calculate_p_saved(p_under, opponents)
                
                # Expected cost: Penalty * Probability of actually taking the penalty
                current_cost = (1.0 - p_saved) * min_bulls
            else:
                # Scenario B: The card naturally falls into an existing row
                L_t = target_row['length']
                e_t = target_row['end_card']
                R_t_bh = target_row['bullheads']
                
                if L_t == 5:
                    # If the row is already 5 cards long, our card strictly triggers a penalty
                    current_cost = R_t_bh
                else:
                    # Row length < 5. Calculate the probability that opponents will play enough cards 
                    # in this specific gap (between the row's end card and our card) to push it to 6 cards.
                    k_req = 5 - L_t
                    if k_req > opponents:
                        # Mathematically impossible for the row to break this turn purely by normal placement
                        P_break = 0.0
                    else:
                        # Find the effective probability that a single opponent plays into this gap
                        d_eff = sum(weights[x] for x in unseen_cards_list if e_t < x < c)
                        p_break = min(d_eff / N_eff, 1.0)
                        
                        # Use Binomial PMF to compute probability of at least k_req opponents playing into this gap
                        P_break = 0.0
                        for m in range(k_req, opponents + 1):
                            P_break += math.comb(opponents, m) * (p_break ** m) * ((1.0 - p_break) ** (opponents - m))
                                    
                    # Add chaos penalty caused by board instability (cascading undercuts)
                    # and an arbitrary shift in the board state
                    chaos_penalty = p_chaos * min_bulls_global * 0.5
                    current_cost = P_break * R_t_bh + chaos_penalty
                        
            card_costs[c] = current_cost
                
        # 6. Apply Softmax to probabilistically select a card prioritizing lower expected penalties
        return self.select_action_softmax(card_costs)
