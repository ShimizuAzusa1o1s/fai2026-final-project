class HumanHeuristicPlayer():
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def _get_row_bullheads(self, row):
        score = 0
        for i in row:
            if i % 55 == 0:
                score += 7
            elif i % 11 == 0:
                score += 5
            elif i % 10 == 0:
                score += 3
            elif i % 5 == 0:
                score += 2
            else:
                score += 1
        return score

    def _row_tension(self, row_stats):
        """
        Returns a danger score per row.
        Length 5 = certain death, length 4 = high risk.
        """
        tension = {}
        for r in row_stats:
            if r['length'] == 5:
                tension[r['index']] = float('inf')
            elif r['length'] == 4:
                tension[r['index']] = r['bullheads'] * 2.0
            else:
                tension[r['index']] = r['bullheads'] * (r['length'] / 5.0)
        return tension

    def _find_safe_gaps(self, hand, row_stats, seen_cards):
        """
        For each card in hand, check if the gap between the target row's
        end card and our card is 'owned' (no unseen cards exist in the gap).
        """
        safe_plays = []
        for c in hand:
            target = max(
                (r for r in row_stats if r['end_card'] < c),
                key=lambda r: r['end_card'],
                default=None
            )
            if target is None:
                continue
            gap_cards = [u for u in range(target['end_card'] + 1, c)
                         if u not in seen_cards and u not in hand]
            if len(gap_cards) == 0:
                safe_plays.append(c)
        return safe_plays

    def _should_undercut(self, hand, row_stats):
        """
        Returns the best undercut card if deliberate undercutting is strategically
        better than any normal placement.
        """
        sorted_rows = sorted(row_stats, key=lambda r: (r['bullheads'], r['length']))
        cheapest = sorted_rows[0]

        # Only consider undercutting if cost is low and board is dangerous
        high_tension_rows = sum(1 for r in row_stats if r['length'] >= 4)
        undercut_candidates = [c for c in hand
                               if c < min(r['end_card'] for r in row_stats)]

        if cheapest['bullheads'] <= 3 and high_tension_rows >= 3:
            if undercut_candidates:
                # Play the largest undercut card — minimizes wasted card value
                return max(undercut_candidates)
        return None

    def _card_danger_zone(self, card, row_stats):
        """
        Penalizes median cards that are likely to collide with opponent plays.
        """
        ends = sorted(r['end_card'] for r in row_stats)
        # Cards that fall just above a row end in a crowded range are most dangerous
        for end in ends:
            if end < card < end + 15:  # tight gap = high collision risk
                return True
        return False

    def _late_game_strategy(self, hand, seen_cards, row_stats):
        """
        In late game, reason about which remaining cards are dangerous
        by elimination rather than probability.
        """
        unseen = [c for c in range(1, 105) if c not in seen_cards]

        certain_safe = []
        certain_danger = []

        for c in hand:
            target = max(
                (r for r in row_stats if r['end_card'] < c),
                key=lambda r: r['end_card'],
                default=None
            )
            if target is None:
                continue
            # Cards that could land before ours in the same row
            interlopers = [u for u in unseen if target['end_card'] < u < c]
            needed_to_fill = 5 - target['length']

            if len(interlopers) < needed_to_fill:
                certain_safe.append(c)
            elif len(interlopers) >= needed_to_fill and target['length'] >= 4:
                certain_danger.append(c)

        return certain_safe, certain_danger

    def _row_priority(self, row_stats):
        """
        Rank rows by how badly you want to avoid triggering them.
        Higher score = avoid more urgently.
        """
        return sorted(
            row_stats,
            key=lambda r: (r['bullheads'] / max(1, 6 - r['length'])),
            reverse=True
        )

    def action(self, hand, history):
        board = history.get('board', [])
        
        seen_cards = set()
        for p_action in history.get('history_matrix', []):
            if isinstance(p_action, dict):
                for card in p_action.values():
                    seen_cards.add(card)
            elif isinstance(p_action, list) or isinstance(p_action, tuple):
                 for val in p_action:
                      if isinstance(val, tuple):
                          seen_cards.add(val[0])
                      else:
                          seen_cards.add(val)
        
        for row in board:
            for card in row:
                seen_cards.add(card)
                
        row_stats = []
        for i, row in enumerate(board):
            row_stats.append({
                'index': i,
                'length': len(row),
                'end_card': row[-1],
                'bullheads': self._get_row_bullheads(row)
            })

        # Priority 1: Late game (<=3 cards)
        if len(hand) <= 3:
            certain_safe, certain_danger = self._late_game_strategy(hand, seen_cards, row_stats)
            if certain_safe:
                return min(certain_safe) # Return smallest safe card to conserve large ones
            
            candidates = set(hand) - set(certain_danger)
            if candidates:
                hand_to_consider = list(candidates)
            else:
                hand_to_consider = hand
        else:
            hand_to_consider = hand

        # Priority 2: Safe gap exists
        safe_gaps = self._find_safe_gaps(hand_to_consider, row_stats, seen_cards)
        if safe_gaps:
            return min(safe_gaps)

        # Priority 3: Deliberate undercut if cheap
        undercut_card = self._should_undercut(hand_to_consider, row_stats)
        if undercut_card is not None:
             return undercut_card

        # Priority 4: Avoid top priority dangerous row
        if row_stats:
            avoidance_order = self._row_priority(row_stats)
            top_danger_row_idx = avoidance_order[0]['index']
            
            non_fatal_candidates = []
            for c in hand_to_consider:
                target = max(
                    (r for r in row_stats if r['end_card'] < c),
                    key=lambda r: r['end_card'],
                    default=None
                )
                if target is None:
                    non_fatal_candidates.append(c)
                elif target['index'] != top_danger_row_idx:
                    non_fatal_candidates.append(c)
            
            if non_fatal_candidates:
                candidates = non_fatal_candidates
            else:
                candidates = hand_to_consider
        else:
            candidates = hand_to_consider
            
        # Priority 5: Dump median cards
        median_candidates = [c for c in candidates if self._card_danger_zone(c, row_stats)]
        if median_candidates:
            candidates = median_candidates
            
        # Priority 6: Prefer lowest row tension target
        tension = self._row_tension(row_stats)
        best_card = None
        best_tension = float('inf')
        
        for c in candidates:
            target = max(
                (r for r in row_stats if r['end_card'] < c),
                key=lambda r: r['end_card'],
                default=None
            )
            if target is None:
                # Undercutting consequence: tension based on cheapest row
                cheapest = sorted(row_stats, key=lambda r: (r['bullheads'], r['length']))[0]
                t = cheapest['bullheads']
            else:
                t = tension[target['index']]
                
            # Tie breaker: larger card is slightly better if same tension, 
            # or just take the one that minimizes objective
            if t < best_tension:
                best_tension = t
                best_card = c
            elif t == best_tension:
                if best_card is None or c < best_card:
                    best_card = c
                    
        if best_card is None:
            return min(hand)
        
        return best_card