"""
Expert Heuristic Player Module

This module implements a sophisticated heuristic-based player that mimics human-like strategic
decision-making for a card game. The player uses multiple prioritized strategies to select
cards, including risk assessment, safe placement detection, and late-game analysis.

The core philosophy is to make strategic decisions based on:
1. Game state analysis (board rows, bullhead values, visible cards)
2. Risk assessment (row tension, collision probability)
3. Multi-level decision hierarchy (safe gaps -> undercutting -> tension minimization)
"""


class HumanHeuristicPlayer():
    """
    A heuristic-based player that uses expert-level strategic reasoning to decide which card to play.
    
    This player analyzes the board state and employs multiple decision-making strategies:
    - Calculates row "tension" (danger level based on bullheads and row length)
    - Identifies "safe gaps" where cards can be placed without risk
    - Evaluates undercutting opportunities when cost is low
    - Uses late-game analysis for endgame decisions
    - Prioritizes rows by danger level
    
    The decision process follows a strict priority order to minimize risk while adapting
    to different game phases.
    """
    
    def __init__(self, player_idx):
        """
        Initialize the heuristic player.
        
        Args:
            player_idx (int): The player's index in the game (0-3)
        """
        self.player_idx = player_idx

    def _get_row_bullheads(self, row):
        """
        Calculate the total bullhead penalty for a given row.
        
        Bullheads are penalty points with special values:
        - Card 55: 7 bullheads (special card, highest penalty)
        - Cards divisible by 11: 5 bullheads
        - Cards divisible by 10: 3 bullheads
        - Cards divisible by 5: 2 bullheads
        - All other cards: 1 bullhead
        
        Args:
            row (list): A list of card values in a row
            
        Returns:
            int: Total bullhead score for the row
        """
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
        Calculate a "tension" score for each row indicating how dangerous it is.
        
        Tension represents the risk of triggering a row (being forced to take its bullheads).
        The scoring system is:
        - Length 5: Infinite tension (critical - will definitely trigger on 6th card)
        - Length 4: Bullheads * 2.0 (very high risk)
        - Length < 4: Bullheads * (length / 5.0) (scaled by how close to 6 cards)
        
        Args:
            row_stats (list): List of dicts containing row metadata:
                - index: row position
                - length: number of cards in the row
                - end_card: highest card value in the row
                - bullheads: total bullhead penalty for the row
                
        Returns:
            dict: Mapping from row index to tension score
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
        Identify cards that can be safely placed by finding "safe gaps" between rows and opponent cards.
        
        A safe gap exists when a card can be placed in a row without risk of collision with
        unseen opponent cards AND without triggering the 6-card rule. Specifically, for each 
        card in hand, we check if:
        1. The target row has fewer than 5 cards (to avoid immediate 6-card penalty)
        2. There are no unseen cards that could land in the gap between the target row's 
           end and our card (to avoid collision with opponent plays)
        
        If no unseen cards exist in the gap AND the target row is not full, the placement 
        is considered safe.
        
        Logic:
        - Find the highest row that our card can follow (row.end_card < our_card)
        - Verify the target row has < 5 cards (not about to trigger 6-card rule)
        - Check all integers between that row's end and our card
        - If all gap integers are either seen or in our hand, the play is safe
        
        Args:
            hand (list): Cards available to play
            row_stats (list): List of row metadata dicts
            seen_cards (set): All cards that have been revealed (visible on board + played)
            
        Returns:
            list: Cards from hand that have safe gaps (guaranteed safe placement)
        """
        safe_plays = []
        for c in hand:
            # Find the row with the highest end card that is still less than our card
            # AND ensure it doesn't have 5 cards (which would trigger 6-card rule immediately)
            target = max(
                (r for r in row_stats if r['end_card'] < c and r['length'] < 5),
                key=lambda r: r['end_card'],
                default=None
            )
            if target is None:
                continue
            # Identify unseen cards in the gap between row end and our card
            gap_cards = [u for u in range(target['end_card'] + 1, c)
                         if u not in seen_cards and u not in hand]
            # If gap contains no unseen cards, this is a safe play
            if len(gap_cards) == 0:
                safe_plays.append(c)
        return safe_plays

    def _should_undercut(self, hand, row_stats):
        """
        Evaluate whether deliberate undercutting is strategically beneficial.
        
        Undercutting means playing a card smaller than all row ends, forcing you to take
        a row's bullheads. This is only desirable when:
        1. The cheapest row has low bullhead cost (3 or less)
        2. The board has multiple dangerous rows (3+ rows with 4+ cards)
        3. We have undercut candidates available
        
        In such situations, deliberately taking a cheap row is better than risking collision
        with opponent cards in a crowded state.
        
        Args:
            hand (list): Available cards to play
            row_stats (list): List of row metadata dicts
            
        Returns:
            int or None: The largest undercut card if undercutting is recommended,
                        None otherwise (play normally instead)
        """
        # Sort rows by cost (bullheads first, then length as tiebreaker)
        sorted_rows = sorted(row_stats, key=lambda r: (r['bullheads'], r['length']))
        cheapest = sorted_rows[0]

        # Count how many rows have high danger (4+ cards)
        high_tension_rows = sum(1 for r in row_stats if r['length'] >= 4)
        # Find cards smaller than all row ends (undercut candidates)
        undercut_candidates = [c for c in hand
                               if c < min(r['end_card'] for r in row_stats)]

        # Only undercut if board is crowded and cheapest row is affordable
        if cheapest['bullheads'] <= 3 and high_tension_rows >= 3:
            if undercut_candidates:
                # Play the largest undercut card to minimize wasted card value
                return max(undercut_candidates)
        return None

    def _card_danger_zone(self, card, row_stats):
        """
        Determine if a card falls in a collision-prone danger zone.
        
        Median-range cards (those in the middle of the playable range) are more likely
        to collide with opponent cards. This method identifies cards that:
        - Fall just above a row end (within 15 units)
        - Are in a "tight gap" where opponent cards are likely
        
        These cards are risky to keep and should be dumped if possible.
        
        Args:
            card (int): The card value to evaluate
            row_stats (list): List of row metadata dicts
            
        Returns:
            bool: True if card is in a danger zone, False if relatively safe
        """
        ends = sorted(r['end_card'] for r in row_stats)
        # Cards that fall just above a row end in a crowded range are most dangerous
        # (high collision risk with opponent cards)
        for end in ends:
            if end < card < end + 15:  # tight gap = high collision risk
                return True
        return False

    def _late_game_strategy(self, hand, seen_cards, row_stats):
        """
        Employ special analysis for endgame situations (3 or fewer cards remaining).
        
        In the late game, instead of probabilistic reasoning, we can use logical deduction.
        We know exactly how many cards remain unseen and how many "interlopers" (unseen cards
        that could land before our card) are possible in each row.
        
        A card is "certain safe" if fewer interlopers exist than needed to fill the row.
        A card is "certain danger" if the row has 4+ cards and enough interlopers exist.
        
        Args:
            hand (list): Remaining cards in player's hand
            seen_cards (set): All cards that have been revealed
            row_stats (list): List of row metadata dicts
            
        Returns:
            tuple: (certain_safe_cards, certain_danger_cards) - two lists of card values
        """
        unseen = [c for c in range(1, 105) if c not in seen_cards]

        certain_safe = []
        certain_danger = []

        for c in hand:
            # Find target row that would receive this card
            target = max(
                (r for r in row_stats if r['end_card'] < c),
                key=lambda r: r['end_card'],
                default=None
            )
            if target is None:
                continue
            # Count unseen cards that would land before ours in the same row
            interlopers = [u for u in unseen if target['end_card'] < u < c]
            # How many more cards needed to complete the row (trigger it)
            needed_to_fill = 5 - target['length']

            # Safe if not enough interlopers to force us to take the row
            if len(interlopers) < needed_to_fill:
                certain_safe.append(c)
            # Danger if many interlopers and row is already nearly full
            elif len(interlopers) >= needed_to_fill and target['length'] >= 4:
                certain_danger.append(c)

        return certain_safe, certain_danger

    def _row_priority(self, row_stats):
        """
        Rank rows by how urgently they should be avoided.
        
        Rows with high bullhead penalties relative to available space (5 - length)
        are prioritized for avoidance. The priority metric is:
        (bullheads / available_space), with higher scores = higher avoidance priority.
        
        Args:
            row_stats (list): List of row metadata dicts
            
        Returns:
            list: Row stats sorted by danger priority (highest priority first)
        """
        return sorted(
            row_stats,
            key=lambda r: (r['bullheads'] / max(1, 6 - r['length'])),
            reverse=True
        )

    def action(self, hand, history):
        """
        Main decision-making method that selects the best card to play using a hierarchical
        strategy evaluation system.
        
        Decision hierarchy:
        1. Late game (<=3 cards): Use logical deduction to identify safe and danger cards
        2. Safe gaps: Play cards with guaranteed collision-free placement
        3. Deliberate undercut: If board is crowded and a cheap row exists, undercut strategically
        4. Avoid top priority row: Try not to add to the most dangerous row
        5. Dump median cards: Prioritize removing high-collision-risk cards
        6. Minimize row tension: Among remaining candidates, pick the card targeting lowest tension
        
        Args:
            hand (list): Cards available to play
            history (dict): Game state containing board, history_matrix, and seen cards
            
        Returns:
            int: The selected card to play
        """
        board = history.get('board', [])
        
        # Build set of all seen cards (on board and in history)
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
                
        # Build row statistics for board analysis
        row_stats = []
        for i, row in enumerate(board):
            row_stats.append({
                'index': i,
                'length': len(row),
                'end_card': row[-1],
                'bullheads': self._get_row_bullheads(row)
            })

        # PRIORITY 1: Late game (<=3 cards remaining)
        # Use logical deduction instead of probability-based reasoning
        if len(hand) <= 3:
            certain_safe, certain_danger = self._late_game_strategy(hand, seen_cards, row_stats)
            if certain_safe:
                return min(certain_safe)  # Return smallest safe card to conserve large ones
            
            # Exclude certain danger cards from consideration if possible
            candidates = set(hand) - set(certain_danger)
            if candidates:
                hand_to_consider = list(candidates)
            else:
                hand_to_consider = hand
        else:
            hand_to_consider = hand

        # PRIORITY 2: Safe gap placement
        # These are guaranteed safe plays with no risk of collision
        safe_gaps = self._find_safe_gaps(hand_to_consider, row_stats, seen_cards)
        if safe_gaps:
            return min(safe_gaps)

        # PRIORITY 3: Deliberate undercutting
        # Only when board is crowded and a cheap row exists
        undercut_card = self._should_undercut(hand_to_consider, row_stats)
        if undercut_card is not None:
             return undercut_card

        # PRIORITY 4: Avoid the most dangerous row
        # Try to avoid adding cards to the row we most want to avoid triggering
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
            
        # PRIORITY 5: Dump median cards (danger zone candidates)
        # These cards are likely to collide and should be eliminated if possible
        median_candidates = [c for c in candidates if self._card_danger_zone(c, row_stats)]
        if median_candidates:
            candidates = median_candidates
            
        # PRIORITY 6: Minimize row tension
        # Among remaining candidates, select the card that targets the row with lowest tension
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
                # Undercutting case: tension based on cheapest row
                cheapest = sorted(row_stats, key=lambda r: (r['bullheads'], r['length']))[0]
                t = cheapest['bullheads']
            else:
                t = tension[target['index']]
                
            # Tiebreaker: prefer smaller card if same tension (save larger cards for later)
            if t < best_tension:
                best_tension = t
                best_card = c
            elif t == best_tension:
                if best_card is None or c < best_card:
                    best_card = c
                    
        if best_card is None:
            return min(hand)
        
        return best_card