#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>

extern "C" {

void resolve_batch_with_sampling(
    int n_turns,
    int player_idx,
    float uniform_ratio,
    float minmax_ratio,
    float tau,
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
    int opp_indices[3];
    int idx = 0;
    for(int i = 0; i < 4; i++) {
        if(i != player_idx) opp_indices[idx++] = i;
    }

    #pragma omp parallel for
    for (int c_idx = 0; c_idx < num_cand; ++c_idx) {
        int cand_card = candidates[c_idx];
        int sims = budget[c_idx];
        
        if (sims == 0) {
            out_stats_penalty[c_idx] = 0;
            out_stats_visits[c_idx] = 0;
            continue;
        }
        
        std::mt19937 rng(seed + c_idx + omp_get_thread_num() * 12345);
        std::uniform_real_distribution<float> unif(1e-8f, 1.0f - 1e-8f);
        
        double total_penalty = 0;
        
        int my_rest[10];
        int n_my_rest = 0;
        for(int i = 0; i < n_turns; i++) {
            if(my_hand[i] != cand_card) {
                my_rest[n_my_rest++] = my_hand[i];
            }
        }
        
        for (int b = 0; b < sims; ++b) {
            int opp_hands[3][10];
            bool available[105];
            for(int i = 0; i < 105; i++) available[i] = false;
            for(int i = 0; i < n_unseen; i++) available[unseen_cards[i]] = true;
            
            for(int opp = 0; opp < 3; ++opp) {
                struct CardScore { int card; float score; };
                CardScore scores[105];
                int valid_cnt = 0;
                
                for(int i = 0; i < 105; ++i) {
                    if (available[i]) {
                        float u = unif(rng);
                        float gumbel = -std::log(-std::log(u));
                        float w = card_log_weights[opp * 105 + i];
                        if (w > -1e8f) {
                            scores[valid_cnt].card = i;
                            scores[valid_cnt].score = w + gumbel;
                            valid_cnt++;
                        }
                    }
                }
                
                std::sort(scores, scores + valid_cnt, [](const CardScore& a, const CardScore& b) {
                    return a.score > b.score;
                });
                
                for(int t = 0; t < n_turns; ++t) {
                    available[scores[t].card] = false;
                }
                
                float policy_u = unif(rng);
                struct CardPlayScore { int card; float score; };
                CardPlayScore play_scores[10];
                
                if (policy_u < uniform_ratio) {
                    for(int t = 0; t < n_turns; ++t) {
                        play_scores[t].card = scores[t].card;
                        play_scores[t].score = unif(rng);
                    }
                } else if (policy_u < uniform_ratio + minmax_ratio) {
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
                    for(int t = 0; t < n_turns; ++t) {
                        int c = scores[t].card;
                        play_scores[t].card = c;
                        float u = unif(rng);
                        float gumbel = -std::log(-std::log(u));
                        play_scores[t].score = (S_scores[c] / tau) + gumbel;
                    }
                }
                
                std::sort(play_scores, play_scores + n_turns, [](const CardPlayScore& a, const CardPlayScore& b) {
                    return a.score > b.score;
                });
                
                for(int t = 0; t < n_turns; ++t) {
                    opp_hands[opp][t] = play_scores[t].card;
                }
            }
            
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
            
            int tails[4], lengths[4], rbulls[4];
            for(int i = 0; i < 4; ++i) {
                tails[i] = orig_tails[i];
                lengths[i] = orig_lengths[i];
                rbulls[i] = orig_rbulls[i];
            }
            
            int my_penalty = 0;
            
            for (int t = 0; t < n_turns; ++t) {
                struct PlayedCard { int card; int player; };
                PlayedCard turn_cards[4];
                
                turn_cards[0].card = my_play[t];
                turn_cards[0].player = player_idx;
                turn_cards[1].card = opp_hands[0][t];
                turn_cards[1].player = opp_indices[0];
                turn_cards[2].card = opp_hands[1][t];
                turn_cards[2].player = opp_indices[1];
                turn_cards[3].card = opp_hands[2][t];
                turn_cards[3].player = opp_indices[2];
                
                std::sort(turn_cards, turn_cards + 4, [](const PlayedCard& a, const PlayedCard& b) {
                    return a.card < b.card;
                });
                
                for (int i = 0; i < 4; ++i) {
                    int c = turn_cards[i].card;
                    int p = turn_cards[i].player;
                    
                    int target_row = -1;
                    int max_valid_tail = -1;
                    
                    for (int r = 0; r < 4; ++r) {
                        if (tails[r] < c && tails[r] > max_valid_tail) {
                            max_valid_tail = tails[r];
                            target_row = r;
                        }
                    }
                    
                    if (target_row == -1) {
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
                        
                        if (p == player_idx) my_penalty += rbulls[target_row];
                        lengths[target_row] = 1;
                        tails[target_row] = c;
                        rbulls[target_row] = bullhead_lookup[c];
                    } else if (lengths[target_row] == 5) {
                        if (p == player_idx) my_penalty += rbulls[target_row];
                        lengths[target_row] = 1;
                        tails[target_row] = c;
                        rbulls[target_row] = bullhead_lookup[c];
                    } else {
                        lengths[target_row] += 1;
                        tails[target_row] = c;
                        rbulls[target_row] += bullhead_lookup[c];
                    }
                }
            }
            total_penalty += my_penalty;
        }
        
        out_stats_penalty[c_idx] = total_penalty;
        out_stats_visits[c_idx] = sims;
    }
}

} // extern "C"
