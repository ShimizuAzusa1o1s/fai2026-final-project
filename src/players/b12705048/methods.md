# 2026FAI Final Project - 6 Nimmt! Agent

This document aims to record the methods that has been attempted to be implemented into the 6 Nimmt! agent.

## Method Tried

### 1. Greedy

Simply plays the max/min-card in the hand. Ignore to the game environment.

An interesting observation is that the Maximizer plays stronger then the Minimizer. Yet, the performance of the greedy agents cannot even over-perform Baseline 1.

### 2. Risk Evaluation

For each round, the agent tries to use simple combination formula to calculate the probabilty that any cards "under-cut" the target hand card, and obtain the sixth-card penalty.

This agent successfully touch the bottom of the Baseline set, but still fails to play well. The main problem is that the agent looks at the "expected risk", ignoring the fact that the opponent's hand is not visible, and not even try to guess their hand for more information.

### 3. Flat Monti Carlo Search

For each round, the agent runs Monti Carlo simulation only on the next-layer, and perform random rollout for the final result. By inspecting the card with the lowest penalty throughout the simulation, we could play that "optimal" card.

The performance of the FlatMC is extraordinarily well. It manages to outperform all the 10 baselines currently published. Hence, it currently act as a strong baseline model for the model-based soluition.

A huge amount of variants are based on this naive version of Monti Carlo search, as you will see in the following section. Their same problem is that they act pretty bad in the start of the game, since no clear opponent modeling and hands evaluation involves.

## Method Promising

### 1. Monti Carlo Search Variants

As stated in the previous section, we have implemented a considerable amount of variants, but most of them fails to out-perform their original version. Since I have no guarantee that these variants are weaker, we will list all of them in the following section.

#### (1) Offensive Variant

The original FlatMC agent only tracks the agent's own penalty. In this variant, we may attempt to prioritize hurting our opponent when falling behind, by adding reward of making opponents take the penalty to the evaluation function.

Despite acting slightly better in the 6 Nimmt! public tournament (credit to kennyfs), we do not have a solid proof on why this variant works. More study is needed for this variant.

In the prototype of this variant, we do not limit the penalty to only from the sixth-row in the specific row the agent is playing. This makes the information obtained from sampling remains unfiltered, which is a possible improvement for the agent. Also, since there still exists a huge amount of information in the game, other features may also be included in the evaluation.

#### (2) Perfect Information Variant

The original FlatMC randomize the sampling for every simulation. The idea of this agent is to hasten the sampling seed by using the same random sampling from the very beginning. In other words, we are manipulating the "determinization" for Monti Carlo simulation in such imperfect information game.

However, the fact is that due to poor opponent sampling (complete random), the simulation result tends to overfit on the noisy pre-determined result.

The possible improvement is then obvious that we may need to train a sophisticated model(s) to support the simulation with a more solid and sound game setup.

#### (3) Truncated Variant

The original FlatMC performs the random rollout to the very last of the game. Since in the start of the game, rollout to the last phase of the game seems like a non-sense, the idea is to shorten the simulation to generate more samples.

However, such truncation instead makes the agent's vision too shallow, and it fails to act better than its original version.

Currently, my expected solution to this issue is to throw away the penalty-based evaluation in the chaos opening. Instead, hand-evaluation based method can be used in the first round to grab the advantages by trimming the hands, adding pressure to the board, or even make use of opponent's save move.

There are two approached in my mind to achieve this. One is the expert system we mentioned in the previous section. The empirical moves can sometimes help the opening, just like playing chess, where you would like to get yourself a better position. Another is to ensemble a few simple models, which are responsible for the opening evaluation mentioned, respectively.

#### (4) Vectorized Variant

The original FlatMC utilize native Python-style array operations. In this version, we exploit NumPy optimization to speed up the simulation process, sharing the similar ideas to the previous variants.

This version actually aims to achieve a slightly better score in the 6 Nimmt! public tournament. However, the improvements is far lower than expected. This almost provides that simulation speed helps the performance nothing after a certain threshold, and we are almost doomed to rank up the agent without opponent modeling.

However, it is still possible that we do not find the most suitable architecture for such specialized computation. The trick of NumPy optimization still leaves for the future us to resolve.

### 2. Expert System

For each round, the agent evaluates its hand with a policy-hierarchy system, including the following policies from highest to lowest priority:

1. Are we in late game? Use card counting, ignore heuristics.
2. Does a certain-safe gap exist in our hand? Play it.
3. Is the board mostly high-tension? Consider deliberate undercut if cheap.
4. Filter out cards landing in top-priority dangerous rows.
5. Among remaining candidates, prefer dumping median cards.
6. Among equal candidates, prefer lowest row-tension target.

The idea here is actually fantastic. However, most of the strategy are quite inaccurate in the real tournament, and hence it can bearly beats any of the baseline. To make the strategy more specific, and push this method further, we may have to move on to the decision tree method. By extracting the playing style of the FlatMC, we may obtain a better strategy that actually helps.

*Note: This idea is under implementation. A detailed report will be provided in the following section.*

### 3. AlphaZero

AlphaZero is a famous algorithm that mastered perfect information games like Chess and Go by combining Monte Carlo Tree Search (MCTS) with deep neural networks. The idea here is to train a model to evaluate the board state and guide the search, rather than relying on pure random rollouts to the end of the game.

For 6 Nimmt!, the agent would learn by playing millions of games against itself (self-play). The neural network would eventually develop an "intuition" for which cards are safe to play and how dangerous a board is. However, a major challenge is that AlphaZero requires perfect information. Since we cannot see the opponents' hands in 6 Nimmt!, we would need to adapt the algorithm, perhaps by randomly guessing the hidden cards before running the search.

### 4. Deep CFR

Counterfactual Regret Minimization (CFR) is a powerful algorithm specifically designed for games with hidden information, like Poker. Instead of just trying to maximize an immediate score, the agent learns by looking back at its past mistakes and calculating the "regret" for not playing a different card.

Since the game tree of 6 Nimmt! is way too huge to calculate exactly, Deep CFR uses neural networks to approximate these regrets and predict the best overall strategy. This approach could help the agent find a highly robust playing style that is very hard for opponents to exploit. It is a very promising direction for handling the hidden cards and simultaneous turns in 6 Nimmt!, though it requires a huge amount of computing power to train.
