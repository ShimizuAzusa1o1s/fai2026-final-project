import numpy as np

def get_topological_gaps(board):
    """
    Returns the sorted tails of the 4 rows on the board.
    """
    row_ends = np.array([row[-1] for row in board])
    return np.sort(row_ends)

def assign_card_to_bucket(card, sorted_row_ends):
    """
    Assigns a card to one of the 5 topological buckets defined by the sorted row tails.
    """
    if card < sorted_row_ends[0]: return 0
    elif card < sorted_row_ends[1]: return 1
    elif card < sorted_row_ends[2]: return 2
    elif card < sorted_row_ends[3]: return 3
    else: return 4

def get_gap_capacities(sorted_row_ends, unseen_cards):
    """
    Calculates the number of available unplayed cards that fit into each of the 5 buckets.
    """
    capacities = np.zeros(5, dtype=np.int32)
    for card in unseen_cards:
        bucket = assign_card_to_bucket(card, sorted_row_ends)
        capacities[bucket] += 1
    return capacities
