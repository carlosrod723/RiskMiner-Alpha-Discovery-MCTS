# RiskMiner: Alpha Discovery via Risk-Seeking MCTS

## Introduction
This project implements the framework described in the research paper "RiskMiner: Discovering Formulaic Alphas via Risk-Seeking Monte Carlo Tree Search (MCTS)". The implementation focuses on automatically discovering formulaic alphas - mathematical formulas that transform raw stock data into predictive trading signals - using an innovative combination of Monte Carlo Tree Search and risk-seeking policy optimization.

## Theoretical Background

### Reinforcement Learning in Alpha Discovery
Traditional alpha discovery methods often struggle with sparse rewards and inefficient exploration of the vast solution space. Reinforcement Learning (RL) provides a framework where the alpha discovery process can be modeled as a Markov Decision Process (MDP). However, conventional RL approaches focus on maximizing average performance, which may not be optimal for alpha discovery where identifying exceptional predictive signals is crucial.

### Monte Carlo Tree Search (MCTS)
MCTS is a search algorithm that efficiently explores large, discrete solution spaces. It combines:
- Tree Policy: Selection and expansion of nodes based on the UCB1 formula
- Default Policy: Simulation of random actions to evaluate positions
- Backpropagation: Updating node statistics based on simulation results

In my implementation, MCTS is used to systematically explore the space of possible alpha formulas while maintaining a balance between exploration and exploitation.

### Risk-Seeking Policy Optimization
Unlike traditional approaches that optimize for average performance, RiskMiner implements a risk-seeking policy that focuses on discovering high-reward alphas. This is achieved through:
- Quantile-based optimization
- Dynamic reward scaling
- Adaptive exploration parameters

## Implementation Details

### Alpha Pool Management
The system maintains a dynamic pool of discovered alphas, implementing:
- Diversity maintenance through mutual information coefficients
- Performance-based pruning
- Dynamic weight adjustment

### Key Components
1. **MCTS Implementation**
   - UCB1-based node selection
   - Formula generation using operator trees
   - Backpropagation with quantile-based rewards

2. **Feature Engineering**
   - Technical indicators
   - Financial ratios
   - Market microstructure features

3. **Evaluation Framework**
   - Time-series cross-validation
   - Information Coefficient (IC) calculation
   - Backtesting infrastructure

## Results and Performance

### Top Discovered Alphas
The implementation discovered several promising alpha factors:
1. `Dividend_Yield - ev_daily` (IC: 0.0131)
2. `CORREL_30 - ev_daily` (IC: 0.0131)
3. `evebit / ATR_14` (IC: 0.0104)

### Performance Metrics
- **Mean IC**: Measures the predictive power of discovered alphas
- **IC Standard Deviation**: Evaluates consistency across different market conditions
- **Cross-validation Results**: Demonstrates robustness and generalizability

## Key Innovations from Original Paper

1. **Reward-Dense MDP**
   - Structured intermediate rewards
   - Enhanced learning stability
   - Improved alpha synergy discovery

2. **Risk-Seeking Optimization**
   - Focus on best-case performance
   - Efficient exploration of high-potential areas
   - Balance between exploration and exploitation

3. **Dynamic Alpha Pool**
   - Composite alpha construction
   - Diversity maintenance
   - Performance-based pruning

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- PyTorch
- Optuna
- Matplotlib
- Seaborn
- Jupyter

## References
[1] "RiskMiner: Discovering Formulaic Alphas via Risk-Seeking Monte Carlo Tree Search (MCTS)"

## License
This project is licensed under the MIT License.
