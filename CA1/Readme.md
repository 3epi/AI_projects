# AI Course Projects - CA1: Search Algorithms for Sokoban Puzzle

## Overview

This repository contains the complete implementation for Computer Assignment 1 (CA1) of the AI course, featuring four different search algorithms to solve Sokoban-style puzzle games. The project includes algorithmic implementations, performance analysis, and an interactive GUI for gameplay.

## Project Structure

```
CA1/
├── notebook.ipynb                    # Main implementation and analysis notebook
├── game.py                          # Core game logic and mechanics
├── gui.py                           # Interactive GUI using PyRay
├── A1_CA1_810102303.zip            # Complete project submission
├── AI-S04-CA1.pdf                  # Assignment description
├── Readme.md                        # This file
├── __pycache__/                     # Python cache files
└── assets/
    ├── maps/                        # Game maps (10 different puzzles)
    │   ├── map1.txt → map10.txt
    │   └── solutions.txt
    ├── sounds/                      # Audio assets
    │   ├── move.wav
    │   └── music.mp3
    └── sprites/                     # Visual assets
        ├── box/                     # Box sprites (8 variants)
        ├── environment/             # Wall textures
        ├── goal/                    # Goal position sprites
        ├── player/                  # Player character sprites
        └── portal/                  # Portal sprites
```

## Game Description

### Sokoban-Style Puzzle Game
- **Objective**: Push all boxes (B) to their corresponding goal positions (G)
- **Player**: Human character (H) that can move and push boxes
- **Elements**:
  - **W**: Walls (impassable obstacles)
  - **H**: Human player
  - **B**: Boxes to be moved
  - **P**: Portals for teleportation
  - **G**: Goal positions
  - **.**: Empty spaces

### Game Mechanics
- Player can move in 4 directions: Up (U), Down (D), Left (L), Right (R)
- Boxes can be pushed but not pulled
- Goal is achieved when all boxes are placed on goal positions
- Portals provide teleportation functionality

## Implemented Search Algorithms ✅

### 1. Breadth-First Search (BFS)
```python
def solver_bfs(game_map):
    game = Game(game_map)
    initial_player_pos = game.get_player_position()
    initial_boxes = game.get_box_locations()
    goals = sorted(game.get_goal_locations())
    
    queue = deque([(initial_player_pos, tuple(sorted(initial_boxes)), [])])
    visited = {(initial_player_pos, tuple(sorted(initial_boxes)))}
    # ...
```

**Features:**
- **Queue-based exploration**: Uses `collections.deque` for FIFO processing
- **State representation**: (player_position, sorted_box_positions)
- **Cycle detection**: Visited states tracking
- **Optimal solution**: Guarantees shortest path

### 2. Depth-First Search (DFS)
```python
def solver_dfs(game_map, max_depth=1000):
    game = Game(game_map)
    stack = [(initial_player_pos, tuple(sorted(initial_boxes)), [])]
    visited = {(initial_player_pos, tuple(sorted(initial_boxes)))}
    # ...
```

**Features:**
- **Stack-based exploration**: LIFO processing for depth-first traversal
- **Depth limiting**: Maximum depth of 1000 to prevent infinite loops
- **Direction ordering**: Reversed direction order for consistent exploration
- **Memory efficient**: Lower space complexity than BFS

### 3. Iterative Deepening Search (IDS)
```python
def solver_ids(game_map, max_depth=100):
    total_visited_count = 0
    for depth_limit in range(1, max_depth + 1):
        result, visited_count = depth_limited_search(game_map, depth_limit)
        total_visited_count += visited_count
        if result is not None:
            return result, total_visited_count
```

**Features:**
- **Progressive depth limits**: Searches depths 1 to 100 iteratively
- **Depth-limited search**: Helper function with depth constraints
- **Optimal solution**: Combines benefits of BFS and DFS
- **Space efficient**: Linear space complexity

### 4. A* Search - **IMPLEMENTED**
```python
def solver_astar(game_map, heuristic_func=heuristic, weight=1):
    game = Game(game_map)
    open_set = []
    heapq.heappush(open_set, (initial_h_score, counter, 0, initial_player_pos, 
                             tuple(sorted(initial_boxes)), []))
    # ...
```

**Heuristic Function:**
```python
def heuristic(game):
    boxes = game.get_box_locations()
    goals = game.get_goal_locations()
    
    boxes_sorted = sorted(boxes)
    goals_sorted = sorted(goals)
    
    total_distance = 0
    for i in range(len(boxes_sorted)):
        total_distance += manhattan_distance(boxes_sorted[i], goals_sorted[i])
    
    return total_distance
```

**Features:**
- **Priority queue**: Uses `heapq` for efficient best-first search
- **Manhattan distance heuristic**: Admissible heuristic for optimal solutions
- **Weighted A***: Configurable weight parameter for speed vs optimality trade-off
- **Tie-breaking**: Counter to ensure consistent ordering

## Core Game API

### Game Class Methods
```python
game.get_box_locations()           # Returns list of box positions
game.get_goal_locations()          # Returns list of goal positions  
game.get_player_position()         # Returns player position tuple
game.is_game_won()                 # Checks if puzzle is solved

game.set_player_position(pos)      # Set player position
game.set_box_positions(boxes)      # Set box positions

game.apply_move(direction)         # Apply single move ('U','D','L','R')
game.apply_moves(move_list)        # Apply sequence of moves
game.display_map()                 # Print current game state
```

## Performance Analysis & Results

### Benchmark Configuration
```python
SOLVERS = {
    "BFS": solver_bfs,
    "DFS": solver_dfs,
    "IDS": solver_ids,
    "A*": solver_astar
}
```

### Performance Optimizations
- **Algorithm-specific skipping**: Complex maps skipped for slow algorithms
- **DFS**: Skips maps 7+ due to potential non-termination
- **BFS**: Skips map 9 due to memory constraints
- **IDS**: Selective map testing for efficiency

### Weighted A* Analysis
```python
def solve_Astarweighted():
    weights = [1.2, 1.5, 2.0]
    # Tests weighted A* with different heuristic weights
```

**Weight Effects:**
- **w=1.0**: Optimal A* (admissible)
- **w>1.0**: Faster search, potentially suboptimal solutions
- **Trade-off**: Solution quality vs computation time

## Interactive GUI Features

### Game Interface ([`gui.py`](CA1/gui.py))
- **Map Selection**: Choose from 10 different puzzle maps
- **Algorithm Selection**: Select which search algorithm to use
- **Visual Gameplay**: Real-time visualization with sprites and animations
- **Audio**: Background music and sound effects
- **Controls**: Keyboard navigation and interaction

## Map Collection & Complexity

### Available Maps (10 total)
- **Simple puzzles**: Maps 1-3 (quick solving for all algorithms)
- **Medium complexity**: Maps 4-6 (moderate search space)
- **Complex puzzles**: Maps 7-10 (challenging for uninformed search)

### Example Map Format:
```
W   P1  H   W   W   W   W
W   W   W   G1  W   W   W
W   W   W   B1  W   W   W
W   G2  B2  .   P1  W   W
W   W   W   B3  W   W   W
W   W   W   G3  W   W   W
W   W   W   W   W   W   W
```

## Algorithm Implementation Details

### State Representation
- **State**: `(player_position, tuple(sorted(box_positions)))`
- **Sorting**: Ensures consistent state comparison
- **Tuple conversion**: Enables hashing for visited set

### Search Optimizations
1. **Early goal checking**: Check win condition before state expansion
2. **Visited state tracking**: Prevent cycles and redundant exploration
3. **Direction ordering**: Consistent move ordering for reproducible results
4. **Memory management**: Efficient data structures for large search spaces

### Heuristic Design (A*)
- **Admissible**: Never overestimates true cost
- **Consistent**: h(n) ≤ c(n,n') + h(n') for monotonicity
- **Manhattan distance**: Sum of L1 distances from boxes to goals
- **Optimal assignment**: Pairs boxes with goals in sorted order

## Performance Results

### Algorithm Comparison
| Algorithm | Completeness | Optimality | Time Complexity | Space Complexity |
|-----------|-------------|------------|-----------------|------------------|
| **BFS** | ✅ Complete | ✅ Optimal | O(b^d) | O(b^d) |
| **DFS** | ❌ Not complete | ❌ Not optimal | O(b^m) | O(bm) |
| **IDS** | ✅ Complete | ✅ Optimal | O(b^d) | O(bd) |
| **A** | ✅ Complete | ✅ Optimal* | O(b^d) | O(b^d) |

**Optimal with admissible heuristic**

### Practical Performance
- **Simple maps (1-3)**: All algorithms solve quickly
- **Medium maps (4-6)**: A* shows clear advantage
- **Complex maps (7-10)**: Only A* and IDS remain practical
- **Memory usage**: BFS hits memory limits on map 9


## Key Features & Innovations

### Technical Implementations
- **Efficient state representation**: Sorted tuples for consistent hashing
- **Memory optimization**: Strategic algorithm skipping for large maps
- **Robust search**: Depth limiting and cycle detection
- **Comprehensive testing**: All algorithms on all feasible maps

### Educational Value
- **Algorithm comparison**: Clear demonstration of search trade-offs
- **Heuristic design**: Practical admissible heuristic implementation
- **Performance analysis**: Real-world algorithm behavior study
- **Interactive learning**: GUI for hands-on exploration

## Files Description
- [`notebook.ipynb`](CA1/notebook.ipynb): Complete implementation with all 4 search algorithms
- [`game.py`](CA1/game.py): Core game mechanics and API
- [`gui.py`](CA1/gui.py): Interactive GUI with PyRay
- [`assets/maps/`](CA1/assets/maps/): 10 puzzle maps of varying complexity
- [`A1_CA1_810102303.zip`](CA1/A1_CA1_810102303.zip): Complete project submission

## Learning Outcomes

### Algorithmic Understanding
- **Search strategy comparison**: Uninformed vs informed search
- **Complexity analysis**: Time and space trade-offs in practice
- **Heuristic design**: Creating admissible and effective heuristics
- **Optimization techniques**: State representation and search pruning

### Implementation Skills
- **Python data structures**: Queues, stacks, heaps, sets
- **Algorithm optimization**: Memory management and performance tuning
- **Game state management**: Position tracking and move validation
- **Performance measurement**: Timing and state counting

**Note**: This project demonstrates the practical application of fundamental AI search algorithms, showing how theoretical concepts translate to real-world problem-solving scenarios with measurable performance differences.