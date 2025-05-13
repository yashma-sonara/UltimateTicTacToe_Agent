
from utils import State, Action

from typing import Tuple, List, Optional
import numpy as np
import time


class Student_ai:
    def __init__(self):
        """Instantiates your agent.
        """
        self.max_depth = 3
        self.time_limit = 2.9  
        self.start_time = 0

        self.weights = {
            'win_local': 100,          
            'center_local': 3,        
            'corner_local': 2,         
            'winning_pattern_local': 5,   
            'center_global': 5,      
            'corner_global': 3,       
            'winning_pattern_global': 10,   
        }

        # Patterns for detecting two in a row with the third cell empty
        self.winning_patterns = [
            # Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]

        self.transposition_table = {}
        self.killer_moves = [[] for _ in range(10)]
        self.history_table = {}

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        state = state.invert()
        #return state.get_random_valid_action()
        self.killer_moves = [[] for _ in range(10)]
        
        self.start_time = time.time()
        valid_actions = state.get_all_valid_actions()

        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        total_pieces = np.sum(state.board > 0)
        if total_pieces <= 1:
            # Prioritize center of center board
            center_move = (1, 1, 1, 1)
            if center_move in valid_actions:
                return center_move
            
            # Prioritize centers of corner boards
            for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                corner_center = (i, j, 1, 1)
                if corner_center in valid_actions:
                    return corner_center
        
        # Check for immediate winning moves
        for action in valid_actions:
            next_state = state.clone()
            next_state = next_state.change_state(action)
            if next_state.is_terminal() and next_state.terminal_utility() == 1:
                return action
        
        # Check for moves that win a local board
        for action in valid_actions:
            i, j, k, l = action
            next_state = state.clone()
            next_state = next_state.change_state(action)
            if next_state.local_board_status[i, j] == 1:  # Player 1 wins local board
                return action
            
        # Block opponent from winning a local board
        for action in valid_actions:
            i, j, k, l = action
            if self.is_blocking_move(state.board[i, j], k, l):
                return action
                
        # Check for fork-creating moves (moves that create multiple winning paths)
        for action in valid_actions:
            i, j, k, l = action
            next_state = state.clone()
            next_state = next_state.change_state(action)
            if self.creates_fork(next_state.board[i, j], 1, k, l):  
                return action

        best_action = valid_actions[0]
        current_depth = 1

        while current_depth <= self.max_depth and time.time() - self.start_time < self.time_limit:
            best_value = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            ordered_actions = self.order_moves(state, valid_actions, current_depth)

            for action in ordered_actions:
                next_state = state.clone()
                next_state= next_state.change_state(action)

                if next_state.is_terminal() and next_state.terminal_utility() == 1:
                    return action
                
                value = self.min_value(next_state, current_depth - 1, alpha, beta)

                if value > best_value:
                    action_str = str(action)
                    self.history_table[action_str] = self.history_table.get(action_str, 0) + (1 << current_depth)
                    best_value = value
                    best_action = action
                
                alpha = max(alpha, best_value)

                if time.time() - self.start_time > self.time_limit:
                    break
            current_depth += 1
            if time.time() - self.start_time > self.time_limit:
                    break
        return best_action
    
    def order_moves(self, state, valid_actions:List[Tuple[int, int, int, int]], depth: int) -> List[Tuple[int, int, int, int]]:
        action_scores = []

        for action in valid_actions:
            row, col, next_row, next_col = action
            score = 0

            if action in self.killer_moves[depth]:
                score += 1000 
            
            action_str = str(action)
            if action_str in self.history_table:
                score += self.history_table[action_str]

            next_state = state.clone()
            next_state = next_state.change_state(action)
            if next_state.local_board_status[row, col] == 1:
                score += 2000 

            if self.is_blocking_move(state.board[row, col], next_row, next_col):
                score += 1500 
            
            if self.creates_fork(state.board[row, col], 1, next_row, next_col):
                score += 1200 

            
            if next_row == 1 and next_col == 1:
                score += 50
            elif (next_row == 0 or next_row == 2) and (next_col == 0 or next_col == 2):
                score += 30

            if next_state.local_board_status[next_row, next_col] != 0:
                score -= 20

            action_scores.append((action, score))
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return [action for action, _ in action_scores]

    def is_cutoff(self, state, depth: int) -> bool:
        if time.time() - self.start_time > self.time_limit:
            return True
        
        if state.is_terminal():
            return True
            
        if depth <= 0:
            return True
        
        return False
    
    def max_value(self, state, depth: int, alpha: float, beta: float) -> float:
        if self.is_cutoff(state, depth):
            if state.is_terminal():
                return state.terminal_utility()
            return self.evaluate(state, True)
    
        
        valid_actions = state.get_all_valid_actions()
        ordered_actions =  self.order_moves(state, valid_actions, depth)

        first_child = True
        value = float('-inf')
        
        for action in ordered_actions:
            next_state = state.clone()
            next_state = next_state.change_state(action)

            if first_child:
                child_value = self.min_value(next_state , depth - 1, alpha, beta)
                first_child = False
            else:

                child_value = self.min_value(next_state, depth - 1, alpha, alpha + 0.00001)
                if alpha < child_value:
                    child_value = self.min_value(next_state , depth - 1, alpha, beta)
            
            value = max(value, child_value)
            if value >= beta:  # Beta cutoff
                # Record killer move
                if action not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, action)
                    if len(self.killer_moves[depth]) > 2:  # Keep only 2 killer moves
                        self.killer_moves[depth].pop()
                break

            alpha = max(alpha, value)

            
            if time.time() - self.start_time > self.time_limit:
                break
        
        return value

    def min_value(self, state, depth: int, alpha: float, beta: float) -> float:
        
        if self.is_cutoff(state, depth):
            if state.is_terminal():
                return state.terminal_utility()
            return self.evaluate(state, False)
        
        first_child = True

        value = float('inf')
        valid_actions = state.get_all_valid_actions()
        ordered_actions =  self.order_moves(state, valid_actions, depth)
        
        for action in ordered_actions:
            next_state = state.clone()
            next_state = next_state.change_state(action)

            if first_child:
                child_value = self.max_value(next_state, depth - 1, alpha, beta)
                first_child =False
            else:
                child_value = self.max_value(next_state, depth - 1, beta - 0.00001, beta)
                if child_value < beta:
                    child_value = self.max_value(next_state, depth -1 , alpha, beta)

            value = min(value, child_value)
            if value <= alpha:  # Alpha cutoff
                # Record killer move
                if action not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, action)
                    if len(self.killer_moves[depth]) > 2:  # Keep only 2 killer moves
                        self.killer_moves[depth].pop()
                break
            beta = min(beta, value)
            
            if time.time() - self.start_time > self.time_limit:
                break
        
        return value

    
        
    def evaluate(self, state, is_player_1: bool) -> float:
        player = 1 if is_player_1 else 2
        opponent = 2 if is_player_1 else 1
        sign = 1 if player == 1 else -1

        score = 0

        for i in range(3):
            for j in range(3):
                if state.local_board_status[i, j] == player:
                    score += self.weights['win_local'] * sign

                    if i == 1 and j == 1:
                        score += self.weights['center_global'] * sign
                    elif (i == 0 or i == 2) and (j ==0 or j == 2):
                        score += self.weights['corner_global'] * sign
                elif state.local_board_status[i, j] == opponent:
                    score -= self.weights['win_local'] * sign

                    if i == 1 and j ==1:
                        score -= self.weights['center_global'] * sign
                    elif (i == 0 or i == 2) and (j == 0 or j == 2):
                        score -= self.weights['corner_global'] * sign
                elif state.local_board_status[i, j] == 0:  
                    local_score = self.evaluate_local_board(state.board[i, j], player, opponent)
                    score += local_score * sign
        score += self.evaluate_global_patterns(state, player, opponent) * sign
        return score
    
    def evaluate_local_board(self, local_board, player: int, opponent: int) -> float:
        # eval a single local board
        score = 0

        # center pos
        if local_board[1,1] == player:
            score += self.weights['center_local']
        elif local_board[1, 1] == opponent:
            score -= self.weights['center_local']

        #corner pos
        for i, j in [(0,0), (0,2), (2,0), (2,2)]:
            if local_board[i, j] == player:
                score += self.weights['corner_local']
            elif local_board[i,j] == opponent:
                score -= self.weights['corner_local']

        #check 2 in a row
        score += self.count_winning_patterns(local_board, player) * self.weights['winning_pattern_local']
        score -= self.count_winning_patterns(local_board, opponent) * self.weights['winning_pattern_local']

        if self.count_fork_patterns(local_board, player) > 0:
            score += 8 
        if self.count_fork_patterns(local_board, opponent) > 0:
            score -= 8 

        return score
    
    def count_winning_patterns(self, board, player: int) -> int:
        count = 0
        for pattern in self.winning_patterns:
            player_count = 0
            empty_count = 0

            for i,j in pattern:
                if board[i,j] == player:
                    player_count += 1
                elif board[i,j] == 0:
                    empty_count += 1

            if player_count == 2 and empty_count == 1:
                count += 1
        return count
    
    def evaluate_global_patterns(self, state, player: int, opponent: int) -> float:
        score = 0
        board_status = state.local_board_status

        for pattern in self.winning_patterns:
            player_wins = sum(1 for i, j in pattern if board_status[i, j] == player)
            empty_boards = sum(1 for i, j in pattern if board_status[i, j] == 0)
            
            if player_wins == 2 and empty_boards == 1:
                score += self.weights['winning_pattern_global']
            
            opponent_wins = sum(1 for i, j in pattern if board_status[i, j] == opponent)
            
            if opponent_wins == 2 and empty_boards == 1:
                score -= self.weights['winning_pattern_global']
        return score
    
    def is_blocking_move(self, local_board, row: int, col: int) -> bool:
        opponent = 2  

        for pattern in self.winning_patterns:
            if (row, col) in pattern:
                opponent_count = 0
                empty_count = 0
                
                for i, j in pattern:
                    if local_board[i, j] == opponent:
                        opponent_count += 1
                    elif local_board[i, j] == 0:
                        empty_count += 1
                
                if opponent_count == 2 and empty_count == 1:
                    return True
        
        return False
    
    def creates_fork(self, local_board, player, row, col) -> bool:
       
        temp_board = local_board.copy()
        temp_board[row, col] = player
        
   
        winning_paths = 0
        
        for pattern in self.winning_patterns:
            player_count = 0
            empty_count = 0
            
            for i, j in pattern:
                if temp_board[i, j] == player:
                    player_count += 1
                elif temp_board[i, j] == 0:
                    empty_count += 1
            
            if player_count == 2 and empty_count == 1:
                winning_paths += 1
        
        return winning_paths >= 2
    
    def count_fork_patterns(self, board, player: int) -> int:
      
        # Track potential winning lines
        potential_lines = []
        
        for pattern in self.winning_patterns:
            player_count = 0
            empty_positions = []
            
            for i, j in pattern:
                if board[i, j] == player:
                    player_count += 1
                elif board[i, j] == 0:
                    empty_positions.append((i, j))
            
            if player_count == 1 and len(empty_positions) == 2:
                potential_lines.append(empty_positions)
        
        all_empty_positions = [pos for line in potential_lines for pos in line]
        fork_positions = {}
        
        for pos in all_empty_positions:
            if pos in fork_positions:
                fork_positions[pos] += 1
            else:
                fork_positions[pos] = 1
        
        return sum(1 for count in fork_positions.values() if count >= 2)




        



