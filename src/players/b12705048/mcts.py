import time
import random
import math

class MCTS():
    """
    Monte Carlo Tree Search (MCTS) agent for the card game.
    Uses determinization to guess hidden cards and simulates random playouts 
    to evaluate the expected penalty of each possible action.
    """
    def __init__(self, player_idx):
        self.player_idx = player_idx
        # Time budget per turn to ensure play stays within limits
        self.time_limit = 0.85 
        self.total_cards = set(range(1, 105))
        self.seen_cards = set()
        self.prior_weight = 3.0

    def _get_bullheads(self, card):
        if card == 55: return 7
        elif card % 11 == 0: return 5
        elif card % 10 == 0: return 3
        elif card % 5 == 0: return 2
        else: return 1

    def _extract_row_stats(self, board):
        stats = []
        for i, row in enumerate(board):
            stats.append({
                "index": i,
                "end_card": row[-1],
                "length": len(row),
                "bullheads": sum(self._get_bullheads(c) for c in row)
            })
        return stats

    def _compute_card_weight(self, card, row_stats, min_bulls):
        target_row = None
        max_end = -1
        for r in row_stats:
            if r["end_card"] < card and r["end_card"] > max_end:
                max_end = r["end_card"]
                target_row = r
                
        if target_row is None:
            return 1.0 / (min_bulls / 5.0 + 1.0)
            
        if target_row["length"] == 5:
            return 0.05
            
        gap = card - target_row["end_card"]
        base_weight = max(0.2, 5.0 / max(1, gap))
        bullhead_multiplier = 1.0 + 0.2 * self._get_bullheads(card)
        return base_weight * bullhead_multiplier

    def _get_card_weights(self, unseen_cards, board):
        row_stats = self._extract_row_stats(board)
        min_bulls = min(r["bullheads"] for r in row_stats)
        weights = []
        for u in unseen_cards:
            w = self._compute_card_weight(u, row_stats, min_bulls)
            weights.append(max(w, 1e-6))
        return weights

    def _calculate_p_saved(self, p_under, n_opponents=3):
        return 1.0 - (1.0 - p_under) ** n_opponents

    def _estimate_card_cost(self, card, row_stats, min_bulls, weights_dict, N_eff):
        target_row = None
        max_end = -1
        for r in row_stats:
            if r["end_card"] < card and r["end_card"] > max_end:
                max_end = r["end_card"]
                target_row = r
                
        if target_row is None:
            d_under_eff = sum(w for u, w in weights_dict.items() if u < card)
            p_under = min(d_under_eff / N_eff, 1.0)
            p_saved = self._calculate_p_saved(p_under, 3)
            return (1.0 - p_saved) * min_bulls
        else:
            if target_row["length"] == 5:
                return target_row["bullheads"]
            return 0.0

    def _static_risk(self, card, board):
        row_stats = self._extract_row_stats(board)
        min_bulls = min(r["bullheads"] for r in row_stats)
        unseen = list(self.total_cards - self.seen_cards)
        weights_dict = {u: max(self._compute_card_weight(u, row_stats, min_bulls), 1e-6) for u in unseen}
        N_eff = sum(weights_dict.values()) or 1.0
        return self._estimate_card_cost(card, row_stats, min_bulls, weights_dict, N_eff)

    def _rollout_pick(self, hand, board):
        if not hand: return None
        row_stats = self._extract_row_stats(board)
        min_bulls = min(r["bullheads"] for r in row_stats)
        unseen = list(self.total_cards - self.seen_cards)
        weights_dict = {u: max(self._compute_card_weight(u, row_stats, min_bulls), 1e-6) for u in unseen}
        N_eff = sum(weights_dict.values()) or 1.0

        costs = {c: self._estimate_card_cost(c, row_stats, min_bulls, weights_dict, N_eff) for c in hand}
        min_cost = min(costs.values())
        exps = {c: math.exp(-(v - min_cost)) for c, v in costs.items()}
        total = sum(exps.values())
        probs = {c: e / total for c, e in exps.items()}

        r = random.random()
        cumulative = 0.0
        for c, p in probs.items():
            cumulative += p
            if r <= cumulative:
                return c
        return list(hand)[-1]

    def _chaos_surcharge(self, board):
        row_stats = self._extract_row_stats(board)
        min_end = min(r["end_card"] for r in row_stats)
        unseen = [u for u in range(1, 105) if u not in self.seen_cards]
        min_bulls = min(r["bullheads"] for r in row_stats)
        
        weights_dict = {u: max(self._compute_card_weight(u, row_stats, min_bulls), 1e-6) for u in unseen}
        N_eff = sum(weights_dict.values()) or 1.0
        
        p_under = sum(weights_dict[u] for u in unseen if u < min_end) / N_eff
        p_chaos = 1 - (1 - min(p_under, 1.0)) ** 3
        return p_chaos * min_bulls * 0.5

    def _weighted_sample_without_replacement(self, population, weights, k):
        scores = []
        for item, w in zip(population, weights):
            u = random.random()
            if u == 0.0: u = 1e-10
            if u == 1.0: u = 1.0 - 1e-10
            key = -math.log(u) / w
            scores.append((key, item))
        scores.sort(key=lambda x: x[0])
        return [item for key, item in scores[:k]]

    def _simulate_round(self, my_card, my_hand, opp_hands, board):
        my_penalty = 0
        current_my_hand = list(my_hand)
        current_opp_hands = [list(h) for h in opp_hands]
        
        turns_left = len(current_my_hand) + 1 
        
        for turn in range(turns_left):
            played_cards = []
            if turn == 0:
                played_cards.append((my_card, "me"))
            else:
                chosen_card = self._rollout_pick(current_my_hand, board)
                current_my_hand.remove(chosen_card)
                played_cards.append((chosen_card, "me"))
                
            for i, opp_h in enumerate(current_opp_hands):
                opp_card = self._rollout_pick(opp_h, board)
                opp_h.remove(opp_card)
                played_cards.append((opp_card, f"opp_{i}"))
                
            played_cards.sort(key=lambda x: x[0])
            
            for card, owner in played_cards:
                valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                
                if not valid_rows:
                    best_row_idx = -1
                    best_cost = (float("inf"), float("inf"), -1)
                    
                    for idx, row in enumerate(board):
                        bullheads = sum(self._get_bullheads(c) for c in row)
                        length = len(row)
                        cost = (bullheads, length, idx)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_row_idx = idx
                            
                    if owner == "me":
                        my_penalty += best_cost[0]
                        
                    board[best_row_idx] = [card]
                else:
                    target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                    target_row.append(card)
                    
                    if len(target_row) == 6:
                        if owner == "me":
                            my_penalty += sum(self._get_bullheads(c) for c in target_row[:5])
                        board[target_row_idx] = [card]
                        
            my_penalty += self._chaos_surcharge(board)
                        
        return my_penalty

    def action(self, hand, history):
        start_time = time.perf_counter()
        board = history.get("board", []) if isinstance(history, dict) else history[-1]
        
        self.seen_cards.update(hand)
        for row in board:
            self.seen_cards.update(row)
            
        if isinstance(history, dict):
            history_matrix = history.get("history_matrix", [])
            for round_actions in history_matrix:
                for c in round_actions:
                    self.seen_cards.add(c)
            
        unseen_cards = list(self.total_cards - self.seen_cards)
        
        stats = {
            c: {
                "penalty": self._static_risk(c, board) * self.prior_weight,
                "visits": self.prior_weight
            }
            for c in hand
        }
        hand_size = len(hand)
        
        while time.perf_counter() - start_time < self.time_limit:
            weights = self._get_card_weights(unseen_cards, board)
            determinized = self._weighted_sample_without_replacement(unseen_cards, weights, len(unseen_cards))
            
            h = hand_size
            opp_hands = [[], [], []]
            for i, card in enumerate(determinized[:3*h]):
                opp_hands[i % 3].append(card)
            
            for c in hand:
                board_copy = [row[:] for row in board]
                remaining_hand = [card for card in hand if card != c]
                opp_copy = [opp[:] for opp in opp_hands]
                
                penalty = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                stats[c]["penalty"] += penalty
                stats[c]["visits"] += 1
                
        best_card = min(stats.keys(), key=lambda k: stats[k]["penalty"] / max(1, stats[k]["visits"]))
        return best_card
