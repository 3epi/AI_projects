# AI Course Projects - CA2: Genetic Algorithm & Minimax Game

## Overview

This repository contains the implementation for Computer Assignment 2 (CA2) of the AI course, which consists of two main parts:

1. **Genetic Algorithm** for function approximation using Fourier series
2. **Minimax Algorithm** implementation for the Pentago board game

## Project Structure

```
CA2/
├── CA2-Files (2).ipynb                    # Main implementation notebook
├── minimax without prune 100 games output.ipynb  # Minimax performance analysis
├── AI_CA2_810102303_report.pdf           # Project report
├── AI_CA2_810102303.rar                  # Complete project archive
├── AI-S04-CA2_2.pdf                      # Assignment description
└── Readme.md                             # This file
```

## Part 1: Genetic Algorithm for Function Approximation

### Overview
Implementation of a genetic algorithm to approximate various mathematical functions using Fourier series coefficients.

### Features
- **Target Functions**: Support for multiple function types including:
  - Trigonometric functions (sin_cos, complex_fourier)
  - Polynomial functions (linear, quadratic, cubic, polynomial)
  - Special functions (gaussian, square_wave, sawtooth)

- **Selection Methods**:
  - Roulette wheel selection
  - Rank-based selection
  - Best fitness selection

- **Crossover Operations**:
  - Single-point crossover
  - Multi-point crossover
  - Uniform crossover

- **Mutation**: Gaussian mutation with configurable rate and strength

### Key Parameters
```python
numCoeffs = 41          # Number of Fourier coefficients
populationSize = 100    # Population size
generations = 300       # Number of generations
mutationRate = 0.15     # Mutation probability
functionRange = (-π, π) # Function domain
sampleCount = 100       # Number of sample points
```

### Usage
```python
mainloop(sel='roulette', ctype='single', mutationRate=0.15)
mainloop(sel='rank', ctype='multi', mutationRate=0.15)
mainloop(sel='best', ctype='uni', mutationRate=0.15)
```

## Part 2: Minimax Algorithm for Pentago

### Overview
Implementation of the Minimax algorithm with alpha-beta pruning for playing the Pentago board game.

### Game Features
- **6x6 board** divided into four 3x3 blocks
- **Two-phase moves**: Place piece + rotate block
- **Win condition**: 5 pieces in a row (horizontal, vertical, or diagonal)
- **Interactive UI** using Pygame
- **AI opponent** with configurable difficulty

### Algorithm Features
- **Minimax with Alpha-Beta Pruning**: Efficient game tree search
- **Transposition Table**: Memoization for performance optimization
- **Heuristic Evaluation**: Custom evaluation function for non-terminal states
- **Configurable Depth**: Adjustable search depth for difficulty tuning

### Heuristic Function
The evaluation function considers:
- Lines of 2, 3, and 4 consecutive pieces
- Weighted scoring system (4-in-a-row: 10 points, 3-in-a-row: 3 points, 2-in-a-row: 1 point)
- Difference between player and opponent scores

### Usage
```python
game = PentagoGame(ui=True, print=True, depth=3)
result = game.play()
```

### Minimax Pentago
1. Set `ui=True` for interactive gameplay
2. Set `ui=False` for automated performance testing
3. Adjust `depth` parameter to change AI difficulty

## Results

### Genetic Algorithm
- Successfully approximates various mathematical functions
- Visualization shows convergence over generations
- Different selection and crossover methods show varying performance

### Minimax Game
- AI demonstrates strategic gameplay
- Alpha-beta pruning significantly improves performance
- Configurable difficulty through search depth

## Performance Metrics
- **Genetic Algorithm**: MSE (Mean Squared Error) tracking
- **Minimax**: Nodes visited, computation time, win/loss statistics

