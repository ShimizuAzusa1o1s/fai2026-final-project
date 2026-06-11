// Compilation command:
//     g++ -O3 -shared -fPIC -fopenmp -o fast_engine.so fast_engine.cpp

#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>

typedef unsigned __int128 bitboard;

extern "C" {

/**
 * @brief Main entry point for the vectorized C++ Monte Carlo simulation engine.
 * 
 * This function resolves a batch of Monte Carlo simulations for a set of candidate cards
 * using Successive Halving and a multi-policy rollout sequence. It parallelizes the work
 * across multiple CPU threads using OpenMP.
 * 
 * @param n_turns Number of turns remaining in the current round.
 * @param player_idx The seat index (0-3) of the FlatMC agent.
 * @param uniform_ratio Probability of using a purely uniform random rollout policy.
 * @param minmax_ratio Probability of using the Min-Max rollout policy (Exploitation).
 * @param tau Temperature for the Softmax safety heuristic (Exploration).
 * @param orig_tails Array [4] containing the last card of each row on the current board.
 * @param orig_lengths Array [4] containing the number of cards in each row.
 * @param orig_rbulls Array [4] containing the current bullhead penalty total for each row.
 * @param bullhead_lookup Array [105] providing O(1) lookup for the bullheads of any card.
 * @param card_log_weights Array [3][105] containing neural net log-probabilities for each opponent's cards.
 * @param S_scores Array [105] containing the deterministic heuristic safety scores.
 * @param unseen_cards Array [n_unseen] containing cards not currently visible to the agent.
 * @param n_unseen Number of unseen cards.
 * @param my_hand Array [n_turns] containing the agent's current hand.
 * @param candidates Array [num_cand] containing the candidate cards currently surviving Successive Halving.
 * @param budget Array [num_cand] containing the number of simulations allocated to each candidate.
 * @param num_cand Number of active candidates.
 * @param seed Random seed for thread-safe RNG generation.
 * @param out_stats_penalty Array [num_cand] to output the cumulative penalty accumulated by each candidate.
 * @param out_stats_visits Array [num_cand] to output the number of simulations actually run for each candidate.
 */
void resolve_batch_with_sampling(
    int n_turns,
    int player_idx,
    float uniform_ratio,
    float minmax_ratio,
    float tau,
    int eval_method,
    const int* orig_tails,          // [4]
    const int* orig_lengths,        // [4]
    const int* orig_rbulls,         // [4]
    const int* bullhead_lookup,     // [105]
    const float* card_log_weights,  // [3, 105]
    const float* S_scores,          // [105]
    const int* unseen_cards,        // [n_unseen]
    int n_unseen,
    const int* my_hand,             // [n_turns]
    const int* candidates,          // [num_cand]
    const int* budget,              // [num_cand]
    int num_cand,
    int seed,
    double* out_stats_penalty,      // [num_cand]
    int* out_stats_visits           // [num_cand]
) {
    // Determine the indices of the 3 opponents.
    int opp_indices[3];
    int idx = 0;
    for(int i = 0; i < 4; i++) {
        if(i != player_idx) opp_indices[idx++] = i;
    }

    // Parallelize the candidate evaluations. Each thread handles a subset of the candidate cards.
    #pragma omp parallel for
    for (int c_idx = 0; c_idx < num_cand; ++c_idx) {
        int cand_card = candidates[c_idx];
        int sims = budget[c_idx];
        
        // Skip if this candidate was allocated 0 simulations (e.g. eliminated).
        if (sims == 0) {
            out_stats_penalty[c_idx] = 0;
            out_stats_visits[c_idx] = 0;
            continue;
        }
        
        // Initialize thread-safe, reproducible random number generator.
        std::mt19937 rng(seed + c_idx + omp_get_thread_num() * 12345);
        std::uniform_real_distribution<float> unif(1e-8f, 1.0f - 1e-8f);
        
        double total_penalty = 0;
        
        // Determine the remainder of the agent's hand after playing the candidate card.
        int my_rest[10];
        int n_my_rest = 0;
        for(int i = 0; i < n_turns; i++) {
            if(my_hand[i] != cand_card) {
                my_rest[n_my_rest++] = my_hand[i];
            }
        }
        
        // Run the allocated number of Monte Carlo simulations for this candidate.
        for (int b = 0; b < sims; ++b) {
            int opp_hands[3][10];
            bitboard available_mask = 0;
            
            // Reset the available deck to contain only the strictly unseen cards.
            for(int i = 0; i < n_unseen; i++) available_mask |= ((bitboard)1 << unseen_cards[i]);
            
            // ================================================================
            // PHASE 4B: NEURAL DETERMINIZATION VIA GUMBEL-MAX
            // ================================================================
            // Sample opponent hands without replacement using the Gumbel-Max
            // trick to predict plausible opponent cards.
            // ================================================================
            for(int opp = 0; opp < 3; ++opp) {
                struct CardScore { int card; float score; };
                CardScore scores[105];
                int valid_cnt = 0;
                
                // For each available card, sample a Gumbel-perturbed weight.
                // This correctly samples from the NN categorical distribution without replacement.
                uint64_t low = (uint64_t)available_mask;
                uint64_t high = (uint64_t)(available_mask >> 64);
                
                while(low) {
                    int i = __builtin_ctzll(low);
                    float u = unif(rng);
                    float gumbel = -std::log(-std::log(u));
                    float w = card_log_weights[opp * 105 + i];
                    if (w > -1e8f) { // Ignore cards explicitly ruled out by the NN
                        scores[valid_cnt].card = i;
                        scores[valid_cnt].score = w + gumbel;
                        valid_cnt++;
                    }
                    low &= low - 1;
                }
                while(high) {
                    int i = __builtin_ctzll(high) + 64;
                    float u = unif(rng);
                    float gumbel = -std::log(-std::log(u));
                    float w = card_log_weights[opp * 105 + i];
                    if (w > -1e8f) { // Ignore cards explicitly ruled out by the NN
                        scores[valid_cnt].card = i;
                        scores[valid_cnt].score = w + gumbel;
                        valid_cnt++;
                    }
                    high &= high - 1;
                }
                
                // Sort by the Gumbel-perturbed score to select the highest likelihood cards.
                std::sort(scores, scores + valid_cnt, [](const CardScore& a, const CardScore& b) {
                    return a.score > b.score;
                });
                
                // Mark the selected n_turns cards as unavailable for subsequent opponents.
                for(int t = 0; t < n_turns; ++t) {
                    available_mask &= ~((bitboard)1 << scores[t].card);
                }
                
                // ================================================================
                // PHASE 4C: OPPONENT PLAY-ORDER (SOFTMAX POLICY)
                // ================================================================
                // Determine the play order for each opponent's hand using
                // Softmax(S/τ) for exploration or Min-Max sequence for exploitation.
                // ================================================================
                float policy_u = unif(rng);
                struct CardPlayScore { int card; float score; };
                CardPlayScore play_scores[10];
                
                if (policy_u < uniform_ratio) {
                    // Policy 1: Uniform Random Rollout
                    for(int t = 0; t < n_turns; ++t) {
                        play_scores[t].card = scores[t].card;
                        play_scores[t].score = unif(rng);
                    }
                } else if (policy_u < uniform_ratio + minmax_ratio) {
                    // Policy 2: Min-Max Sequence (Exploitation Policy)
                    // Plays alternating smallest and largest cards to create pressure and play safely.
                    std::sort(scores, scores + n_turns, [](const CardScore& a, const CardScore& b) {
                        return a.card < b.card;
                    });
                    int left = 0;
                    int right = n_turns - 1;
                    for(int t = 0; t < n_turns; ++t) {
                        int c = (unif(rng) > 0.5f) ? scores[left++].card : scores[right--].card;
                        play_scores[t].card = c;
                        play_scores[t].score = (float)(n_turns - t);
                    }
                } else {
                    // Policy 3: Softmax-Safety Heuristic (Exploration Policy)
                    // Sample play order according to deterministic safety scores S(c), modulated by tau.
                    for(int t = 0; t < n_turns; ++t) {
                        int c = scores[t].card;
                        play_scores[t].card = c;
                        float u = unif(rng);
                        float gumbel = -std::log(-std::log(u));
                        play_scores[t].score = (S_scores[c] / tau) + gumbel;
                    }
                }
                
                // Sort the hand by the selected policy scores to define the play order.
                std::sort(play_scores, play_scores + n_turns, [](const CardPlayScore& a, const CardPlayScore& b) {
                    return a.score > b.score;
                });
                
                for(int t = 0; t < n_turns; ++t) {
                    opp_hands[opp][t] = play_scores[t].card;
                }
            }
            
            // ================================================================
            // PHASE 4D: ASSIGN OUR CANDIDATE CARDS
            // ================================================================
            // Lock in the candidate card as our first play, then choose the
            // rollout sequence for the remaining cards in hand.
            // ================================================================
            int my_play[10];
            my_play[0] = cand_card;
            
            if (n_my_rest > 0) {
                float policy_u = unif(rng);
                struct CardPlayScore { int card; float score; };
                CardPlayScore play_scores[10];
                
                if (policy_u < uniform_ratio) {
                    for(int t = 0; t < n_my_rest; ++t) {
                        play_scores[t].card = my_rest[t];
                        play_scores[t].score = unif(rng);
                    }
                } else if (policy_u < uniform_ratio + minmax_ratio) {
                    std::sort(my_rest, my_rest + n_my_rest);
                    int left = 0;
                    int right = n_my_rest - 1;
                    for(int t = 0; t < n_my_rest; ++t) {
                        int c = (unif(rng) > 0.5f) ? my_rest[left++] : my_rest[right--];
                        play_scores[t].card = c;
                        play_scores[t].score = (float)(n_my_rest - t);
                    }
                } else {
                    for(int t = 0; t < n_my_rest; ++t) {
                        int c = my_rest[t];
                        play_scores[t].card = c;
                        float u = unif(rng);
                        float gumbel = -std::log(-std::log(u));
                        play_scores[t].score = (S_scores[c] / tau) + gumbel;
                    }
                }
                
                std::sort(play_scores, play_scores + n_my_rest, [](const CardPlayScore& a, const CardPlayScore& b) {
                    return a.score > b.score;
                });
                
                for(int t = 0; t < n_my_rest; ++t) {
                    my_play[1+t] = play_scores[t].card;
                }
            }
            
            // Initialize board state for this simulation
            int tails[4], lengths[4], rbulls[4];
            for(int i = 0; i < 4; ++i) {
                tails[i] = orig_tails[i];
                lengths[i] = orig_lengths[i];
                rbulls[i] = orig_rbulls[i];
            }
            
            int player_penalties[4] = {0, 0, 0, 0};
            
            // ================================================================
            // PHASE 5: SIMD BATCH SIMULATION
            // ================================================================
            // Simulate all n_turns tricks for the game native array-wise.
            // Implements exact 6 Nimmt! placement rules.
            // ================================================================
            for (int t = 0; t < n_turns; ++t) {
                struct PlayedCard { int card; int player; };
                PlayedCard turn_cards[4];
                
                // Collect the 4 cards played this turn.
                turn_cards[0].card = my_play[t];
                turn_cards[0].player = player_idx;
                turn_cards[1].card = opp_hands[0][t];
                turn_cards[1].player = opp_indices[0];
                turn_cards[2].card = opp_hands[1][t];
                turn_cards[2].player = opp_indices[1];
                turn_cards[3].card = opp_hands[2][t];
                turn_cards[3].player = opp_indices[2];
                
                // Rule 1: Cards are resolved in ascending order (lowest card goes first).
                std::sort(turn_cards, turn_cards + 4, [](const PlayedCard& a, const PlayedCard& b) {
                    return a.card < b.card;
                });
                
                for (int i = 0; i < 4; ++i) {
                    int c = turn_cards[i].card;
                    int p = turn_cards[i].player;
                    
                    int target_row = -1;
                    int max_valid_tail = -1;
                    
                    // Rule 2: A card must be placed on the row ending with the highest number that is lower than the card.
                    for (int r = 0; r < 4; ++r) {
                        if (tails[r] < c && tails[r] > max_valid_tail) {
                            max_valid_tail = tails[r];
                            target_row = r;
                        }
                    }
                    
                    if (target_row == -1) {
                        // Rule 3: Undercut. If a card is lower than all row tails, the player takes a row.
                        // We use a deterministic heuristic to pick the cheapest row.
                        int best_row = 0;
                        long long min_score = 1000000;
                        for (int r = 0; r < 4; ++r) {
                            long long score = (long long)rbulls[r] * 1000 + lengths[r] * 10 + r;
                            if (score < min_score) {
                                min_score = score;
                                best_row = r;
                            }
                        }
                        target_row = best_row;
                        
                        // Apply penalty and reset the chosen row.
                        player_penalties[p] += rbulls[target_row];
                        lengths[target_row] = 1;
                        tails[target_row] = c;
                        rbulls[target_row] = bullhead_lookup[c];
                    } else if (lengths[target_row] == 5) {
                        // Rule 4: Row is full. If a card is the 6th in a row, the player takes the 5 existing cards.
                        player_penalties[p] += rbulls[target_row];
                        lengths[target_row] = 1;
                        tails[target_row] = c;
                        rbulls[target_row] = bullhead_lookup[c];
                    } else {
                        // Safe placement. Card is added to the row.
                        lengths[target_row] += 1;
                        tails[target_row] = c;
                        rbulls[target_row] += bullhead_lookup[c];
                    }
                }
            }
            
            // ================================================================
            // PHASE 6: STAT AGGREGATION
            // ================================================================
            // Aggregate penalties across simulations for this candidate card
            // based on the configured evaluation method.
            // ================================================================
            int my_pen = player_penalties[player_idx];
            if (eval_method == 0) { // Avg Penalty
                total_penalty += my_pen;
            } else if (eval_method == 1) { // Win Rate
                int min_pen = player_penalties[0];
                for(int i=1; i<4; i++) if(player_penalties[i] < min_pen) min_pen = player_penalties[i];
                if (my_pen == min_pen) {
                    total_penalty -= 1.0;
                }
            } else if (eval_method == 2) { // Avg Rank
                int rank = 1;
                for(int i=0; i<4; i++) {
                    if (i != player_idx && player_penalties[i] < my_pen) rank++;
                }
                total_penalty += rank;
            } else if (eval_method == 3) { // CVaR (Not strictly implemented, fallback to avg penalty)
                total_penalty += my_pen;
            }
        }
        
        // Output final simulation results.
        out_stats_penalty[c_idx] = total_penalty;
        out_stats_visits[c_idx] = sims;
    }
}

} // extern "C"
