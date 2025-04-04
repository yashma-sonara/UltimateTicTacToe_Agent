import numpy as np
import time
from utils import State, Action

class StudentAgent:
    def __init__(self, max_depth=4):
        """
        Initialize the agent with minimax search and enhanced strategic features.
        """
        self.max_depth = max_depth
        self.time_limit = 2.9  # Slightly under the 3-second limit
        self.start_time = 0
        
        # Transposition table for caching positions
        self.transposition_table = {}
        
        # Win patterns for feature extraction
        self.win_patterns = [
            # Rows
            [(0,0), (0,1), (0,2)],
            [(1,0), (1,1), (1,2)],
            [(2,0), (2,1), (2,2)],
            # Columns
            [(0,0), (1,0), (2,0)],
            [(0,1), (1,1), (2,1)],
            [(0,2), (1,2), (2,2)],
            # Diagonals
            [(0,0), (1,1), (2,2)],
            [(0,2), (1,1), (2,0)]
        ]

        self.center_board = (1, 1)
        self.corner_boards = [(0, 0), (0, 2), (2, 0), (2, 2)]
        self.edge_boards = [(0, 1), (1, 0), (1, 2), (2, 1)]
    
        
        # Weights from the trained model with enhanced features
        self.weights = np.array([0.00000000, -0.01366829, 0.00429636, 0.40386716, -0.42831084, 
                               -0.03889868, 0.05326573, 0.48687491, -0.46963722, 0.01053483, 
                               -0.00373828, 0.35223290, -0.35051805, 0.01017058, -0.00085430, 
                               0.39548324, -0.41333263, -0.04549722, 0.04030073, 0.45999820, 
                               -0.46732841, -0.00670549, 0.01437739, 0.41931465, -0.43597088, 
                               -0.04551689, 0.02088586, 0.50129658, -0.47768339, -0.04798010, 
                               0.04550070, 0.49821079, -0.44909268, -0.04203213, 0.05382382, 
                               -0.00926669, 0.14537709, -0.08066332, 0.00597842, 0.03577999, 
                               -0.04454026, 0.08633282, -0.08439606, 0.00144309, -0.01663572, 
                               0.05191251, -0.04635791, 0.00000000, 0.00000000, 0.00000000, 
                               0.00000000, -0.09874369, 0.02009556, -0.00292050, 0.08420047, 
                               -0.09637487, 0.00982704, -0.00862328])
        
        # Intercept from the trained model
        self.intercept =  0.018418474686929885

        if self.weights[37] < 0:
            self.weights[37] = 0.1  # Force a moderate positive value
        else:
            self.weights[37] *= 1.5

        highest_pattern_features = [7, 27, 31]  # These have weights > 0.48
        medium_pattern_features = [3, 15, 19]   # These have weights 0.39-0.46

        # Apply different boosts based on impact
        for idx in highest_pattern_features:
            if idx < len(self.weights):
                self.weights[idx] *= 2.0  # Increase by 100%
                
        for idx in medium_pattern_features:
            if idx < len(self.weights):
                self.weights[idx] *= 1.8  # Keep your current 80% increase
            
        # Free move features (features 47-50)
        # Since these are zeros in your model, let's add explicit values
        if len(self.weights) > 50:
            self.weights[47] = 0.35  # Explicit free move advantage
            self.weights[48] = 0.18  # Strategic boards available in free move
            self.weights[49] = 0.25  # Winning moves available in free move
            self.weights[50] = 0.30  # Meta-win potential in free move
            
        self.weights[47] = 0.4   # Basic free move indicator (higher priority)
        self.weights[48] = 0.15  # Strategic boards available
        self.weights[49] = 0.3   # Winning moves available (moderately high)
        self.weights[50] = 0.25
    
    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board."""
        #state = state.invert()

        self.start_time = time.time()
        
        total_filled = np.sum(state.board != 0)
        if total_filled <= 6:  # First few moves
            opening_move = self.get_opening_move(state)
            if opening_move and state.is_valid_action(opening_move):
                return opening_move

       
        
        # Clear transposition table for fresh search
        self.transposition_table = {}
        
        # Get all valid actions
        valid_actions = state.get_all_valid_actions()
        
        # Only one move? Return it immediately
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Initial move ordering for better pruning
        action_scores = []
        for action in valid_actions:
            next_state = state.change_state(action)
            score = self.quick_evaluate(next_state)
            action_scores.append((action, score))
        
        # Sort actions by score in descending order
        sorted_actions = [a for a, s in sorted(action_scores, key=lambda x: x[1], reverse=True)]
        
        # Iterative deepening search
        best_action = sorted_actions[0]  # Fallback to first sorted action
        current_depth = 1
        
        while current_depth <= self.max_depth and time.time() - self.start_time < self.time_limit:
            # Search at current depth
            action, value = self.minimax(state, current_depth, float('-inf'), float('inf'), True)
            
            if action is not None:  # Found a valid move within time
                best_action = action
                current_depth += 1
            else:  # Out of time, use best action so far
                break
        
        return best_action
    
    def minimax(self, state, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning
        Returns (best_action, value) tuple
        """
        # Check if time is running out
        if time.time() - self.start_time > self.time_limit * 0.9:
            return None, 0
            
        # Terminal state check
        if state.is_terminal():
            return None, self.get_terminal_value(state)
        
        # Depth limit reached
        if depth == 0:
            return None, self.evaluate_state(state)
        
        # Check transposition table
        state_hash = self.hash_state(state)
        if state_hash in self.transposition_table and self.transposition_table[state_hash]['depth'] >= depth:
            return None, self.transposition_table[state_hash]['value']
        
        valid_actions = state.get_all_valid_actions()
        
        # Try to improve move ordering for better pruning
        action_scores = []
        for action in valid_actions:
            next_state = state.change_state(action)
            score = self.quick_evaluate(next_state)
            action_scores.append((action, score))
        
        # Sort actions by score for better pruning
        if maximizing_player:
            sorted_actions = [a for a, s in sorted(action_scores, key=lambda x: x[1], reverse=True)]
        else:
            sorted_actions = [a for a, s in sorted(action_scores, key=lambda x: x[1])]
        
        best_action = None
        
        if maximizing_player:
            best_value = float('-inf')
            
            for action in sorted_actions:
                next_state = state.change_state(action)
                _, value = self.minimax(next_state, depth - 1, alpha, beta, False)
                
                if value > best_value:
                    best_value = value
                    best_action = action
                
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
                
                # Check time limit frequently
                if time.time() - self.start_time > self.time_limit * 0.9:
                    break
        else:
            best_value = float('inf')
            
            for action in sorted_actions:
                next_state = state.change_state(action)
                _, value = self.minimax(next_state, depth - 1, alpha, beta, True)
                
                if value < best_value:
                    best_value = value
                    best_action = action
                
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
                
                # Check time limit frequently
                if time.time() - self.start_time > self.time_limit * 0.9:
                    break
        
        # Save to transposition table
        self.transposition_table[state_hash] = {'depth': depth, 'value': best_value}
        return best_action, best_value
    
    def hash_state(self, state):
        """Create a hash for storing states in the transposition table"""
        # For better performance, use a more compact representation
        board_str = ""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        board_str += str(state.board[i][j][k][l])
        
        return hash((board_str, state.fill_num))
    
    def get_terminal_value(self, state):
        """Returns the value of a terminal state"""
        utility = state.terminal_utility()
        
        # Map 0.0, 0.5, 1.0 to -1.0, 0.0, 1.0 for Player 1's perspective
        if utility == 1.0:  # Player 1 wins
            return 1.0
        elif utility == 0.0:  # Player 2 wins
            return -1.0
        else:  # Draw
            return 0.0
    
    def extract_enhanced_features(self, state):
        """Extract strategic features from the state with enhanced center board and free move analysis"""
        features = []
        
        # Bias term
        features.append(1.0)
        
        # Get board data
        local_board_status = state.local_board_status
        board = state.board
        
        # --- FEATURE GROUP 1: META-BOARD PATTERN FEATURES ---
        
        # For each win pattern on meta-board
        for pattern in self.win_patterns:
            p1_count = 0
            p2_count = 0
            empty_count = 0
            
            for i, j in pattern:
                cell = local_board_status[i][j]
                if cell == 1:
                    p1_count += 1
                elif cell == 2:
                    p2_count += 1
                elif cell == 0:
                    empty_count += 1
            
            # Add basic counts
            features.append(p1_count)
            features.append(p2_count)
            
            # Add weighted pattern values - patterns with more cells are more valuable
            if p1_count > 0 and p2_count == 0:
                if p1_count == 1:
                    features.append(0.1)  # One in a row
                elif p1_count == 2:
                    features.append(0.8)  # Two in a row - much more valuable
                else:
                    features.append(1.0)  # Complete row (should be game over)
            else:
                features.append(0.0)
                
            if p2_count > 0 and p1_count == 0:
                if p2_count == 1:
                    features.append(0.1)  # One in a row
                elif p2_count == 2:
                    features.append(0.8)  # Two in a row - much more valuable
                else:
                    features.append(1.0)  # Complete row (should be game over)
            else:
                features.append(0.0)
        
        # --- FEATURE GROUP 2: BOARD CONTROL FEATURES ---
        
        # Count boards won by each player
        p1_boards = np.sum(local_board_status == 1)
        p2_boards = np.sum(local_board_status == 2)
        features.append(p1_boards)
        features.append(p2_boards)
        
        # Count drawn boards
        drawn_boards = np.sum(local_board_status == 3)
        features.append(drawn_boards)
        
        # Ratio of won boards (to capture relative advantage)
        total_decided = p1_boards + p2_boards
        if total_decided > 0:
            features.append((p1_boards - p2_boards) / total_decided)
        else:
            features.append(0.0)
        
        # --- FEATURE GROUP 3: ENHANCED CENTER BOARD ANALYSIS ---
        
        # Basic center board ownership
        if local_board_status[self.center_board] == 1:
            features.append(1.0)  # Player 1 owns center
        elif local_board_status[self.center_board] == 2:
            features.append(-1.0)  # Player 2 owns center
        else:
            features.append(0.0)  # Nobody owns center
        
        # Enhanced center board analysis when it's still active
        if local_board_status[self.center_board] == 0:
            center_board_arr = board[self.center_board]
            
            # Count pieces in center board
            p1_center_pieces = np.sum(center_board_arr == 1)
            p2_center_pieces = np.sum(center_board_arr == 2)
            features.append(p1_center_pieces - p2_center_pieces)  # Piece advantage
            
            # Check for potential wins in center board
            p1_potential_wins = 0
            p2_potential_wins = 0
            
            for pattern in self.win_patterns:
                p1_pattern = 0
                p2_pattern = 0
                empty_cells = 0
                
                for k, l in pattern:
                    cell = center_board_arr[k][l]
                    if cell == 1:
                        p1_pattern += 1
                    elif cell == 2:
                        p2_pattern += 1
                    else:
                        empty_cells += 1
                
                # Count potential win patterns (2 in a row with empty cell)
                if p1_pattern == 2 and empty_cells == 1:
                    p1_potential_wins += 1
                if p2_pattern == 2 and empty_cells == 1:
                    p2_potential_wins += 1
            
            features.append(p1_potential_wins)  # P1 near-wins in center
            features.append(p2_potential_wins)  # P2 near-wins in center
        else:
            # Center board is already claimed or drawn
            features.append(0.0)  # No piece advantage
            features.append(0.0)  # No P1 potential wins
            features.append(0.0)  # No P2 potential wins
        
        # Check if center is part of potential meta-board win
        center_part_of_p1_potential = 0
        center_part_of_p2_potential = 0
        
        for pattern in self.win_patterns:
            if self.center_board in pattern:
                p1_count = 0
                p2_count = 0
                
                for i, j in pattern:
                    if local_board_status[i][j] == 1:
                        p1_count += 1
                    elif local_board_status[i][j] == 2:
                        p2_count += 1
                
                # Check if this pattern could still be completed
                p1_can_complete = (p1_count > 0 and p2_count == 0)
                p2_can_complete = (p2_count > 0 and p1_count == 0)
                
                if p1_can_complete:
                    center_part_of_p1_potential += 1
                if p2_can_complete:
                    center_part_of_p2_potential += 1
        
        features.append(center_part_of_p1_potential)
        features.append(center_part_of_p2_potential)
        
        # --- FEATURE GROUP 4: STRATEGIC BOARD POSITION FEATURES ---
        
        # Corner boards value
        corner_value = 0
        for i, j in self.corner_boards:
            if local_board_status[i][j] == 1:
                corner_value += 1
            elif local_board_status[i][j] == 2:
                corner_value -= 1
        features.append(corner_value)
        
        # Edge boards value
        edge_value = 0
        for i, j in self.edge_boards:
            if local_board_status[i][j] == 1:
                edge_value += 1
            elif local_board_status[i][j] == 2:
                edge_value -= 1
        features.append(edge_value)
        
        # --- FEATURE GROUP 5: LOCAL BOARD TACTICAL FEATURES ---
        
        # Count almost-complete local boards (two in a row)
        p1_local_two = 0
        p2_local_two = 0
        
        for i in range(3):
            for j in range(3):
                if local_board_status[i][j] == 0:  # Only active boards
                    local_board = board[i][j]
                    
                    for pattern in self.win_patterns:
                        p1_count = 0
                        p2_count = 0
                        empty_count = 0
                        
                        for k, l in pattern:
                            cell = local_board[k][l]
                            if cell == 1:
                                p1_count += 1
                            elif cell == 2:
                                p2_count += 1
                            elif cell == 0:
                                empty_count += 1
                        
                        # Count almost-won patterns
                        if p1_count == 2 and empty_count == 1:
                            p1_local_two += 1
                        if p2_count == 2 and empty_count == 1:
                            p2_local_two += 1
        
        features.append(p1_local_two)
        features.append(p2_local_two)
        
        # --- FEATURE GROUP 6: ENHANCED FREE MOVE ANALYSIS ---
        
        # Determine if next player has a free move or is forced to a specific board
        has_free_move = False
        forced_board_position = None
        
        if hasattr(state, 'prev_local_action') and state.prev_local_action is not None:
            prev_k, prev_l = state.prev_local_action
            
            if 0 <= prev_k < 3 and 0 <= prev_l < 3:
                forced_board_position = (prev_k, prev_l)
                
                # If the target board is not active (won/drawn/full), player has free move
                if local_board_status[prev_k][prev_l] != 0:
                    has_free_move = True
        
        # Basic free move indicator
        features.append(1.0 if has_free_move else 0.0)
        
        # Enhanced free move analysis
        if has_free_move:
            # Count available strategic boards that are still active
            available_strategic_boards = 0
            available_winning_moves = 0
            
            for i in range(3):
                for j in range(3):
                    if local_board_status[i][j] == 0:  # Active board
                        # Count strategic positions
                        if (i, j) == self.center_board:
                            available_strategic_boards += 2  # Center has double value
                        elif (i, j) in self.corner_boards:
                            available_strategic_boards += 1  # Corners are valuable
                        
                        # Count potential winning moves on these boards
                        local_board = board[i][j]
                        for pattern in self.win_patterns:
                            p1_count = 0
                            empty_pos = None
                            
                            for k, l in pattern:
                                cell = local_board[k][l]
                                if cell == 1:
                                    p1_count += 1
                                elif cell == 0:
                                    empty_pos = (k, l)
                            
                            # If player can win with one move
                            if p1_count == 2 and empty_pos is not None:
                                available_winning_moves += 1
                                break  # Only count one winning move per board
            
            features.append(available_strategic_boards / 9.0)  # Normalize
            features.append(available_winning_moves / 9.0)  # Normalize
            
            # Strategic value of free move in the meta-game
            meta_win_potential = 0
            for pattern in self.win_patterns:
                p1_count = 0
                empty_boards = []
                
                for i, j in pattern:
                    if local_board_status[i][j] == 1:
                        p1_count += 1
                    elif local_board_status[i][j] == 0:
                        empty_boards.append((i, j))
                
                # If player already has one board in this pattern and can target others
                if p1_count == 1 and empty_boards:
                    meta_win_potential += 1
            
            features.append(meta_win_potential / 8.0)  # Normalize
        else:
            # No free move
            features.append(0.0)  # No strategic boards available
            features.append(0.0)  # No winning moves available
            features.append(0.0)  # No meta-win potential
        
        # --- FEATURE GROUP 7: ENHANCED FORCED MOVE ANALYSIS ---
        
        # Analyze the quality of the forced board (if no free move)
        if not has_free_move and forced_board_position is not None:
            i, j = forced_board_position
            
            if 0 <= i < 3 and 0 <= j < 3 and local_board_status[i][j] == 0:
                local_board = board[i][j]
                
                # Check if opponent has immediate winning moves
                p2_winning_moves = 0
                for pattern in self.win_patterns:
                    p2_count = 0
                    empty_pos = None
                    
                    for k, l in pattern:
                        cell = local_board[k][l]
                        if cell == 2:
                            p2_count += 1
                        elif cell == 0:
                            empty_pos = (k, l)
                    
                    if p2_count == 2 and empty_pos is not None:
                        p2_winning_moves += 1
                
                # Check player's potential in this board
                p1_potential = 0
                for pattern in self.win_patterns:
                    p1_count = 0
                    empty_count = 0
                    
                    for k, l in pattern:
                        cell = local_board[k][l]
                        if cell == 1:
                            p1_count += 1
                        elif cell == 0:
                            empty_count += 1
                    
                    if p1_count > 0 and p2_count == 0 and empty_count > 0:
                        if p1_count == 1:
                            p1_potential += 0.1
                        elif p1_count == 2:
                            p1_potential += 0.5
                
                # Strategic value of forced board position
                strategic_value = 0
                if (i, j) == self.center_board:
                    strategic_value = 2  # Center has highest value
                elif (i, j) in self.corner_boards:
                    strategic_value = 1  # Corners have good value
                
                # Check how forcing this board impacts the meta-game
                meta_impact = 0
                for pattern in self.win_patterns:
                    if (i, j) in pattern:
                        p1_pattern = 0
                        p2_pattern = 0
                        
                        for pos in pattern:
                            if pos != (i, j):  # Other positions in pattern
                                board_status = local_board_status[pos]
                                if board_status == 1:
                                    p1_pattern += 1
                                elif board_status == 2:
                                    p2_pattern += 1
                        
                        # If winning this board would make a 2-in-a-row or win
                        if p1_pattern > 0 and p2_pattern == 0:
                            meta_impact += p1_pattern
                        elif p2_pattern > 0 and p1_pattern == 0:
                            meta_impact -= p2_pattern
                
                features.append(p2_winning_moves / 8.0)  # Normalize threat level
                features.append(p1_potential)  # Player potential 
                features.append(strategic_value / 2.0)  # Normalize strategic value
                features.append(meta_impact / 2.0)  # Normalize meta impact
            else:
                features.append(0.0)
                features.append(0.0)
                features.append(0.0)
                features.append(0.0)
        else:
            features.append(0.0)
            features.append(0.0)
            features.append(0.0)
            features.append(0.0)
        
        # --- FEATURE GROUP 8: GAME PHASE AND TEMPO FEATURES ---
        
        # Game phase (early, mid, late)
        total_filled = np.sum(board != 0)
        features.append(total_filled / 81.0)  # Normalized game progress
        
        # Move tempo - who has the initiative?
        # If player has more two-in-rows than opponent, they have better tempo
        tempo = (p1_local_two - p2_local_two) / 10.0 if (p1_local_two + p2_local_two) > 0 else 0
        features.append(tempo)
        
        # Freedom factor - how many choices does the player have?
        if not has_free_move and forced_board_position is not None:
            i, j = forced_board_position
            if 0 <= i < 3 and 0 <= j < 3 and local_board_status[i][j] == 0:
                # Count empty cells in the forced board
                empty_count = np.sum(board[i][j] == 0)
                features.append(empty_count / 9.0)  # Normalized freedom
            else:
                features.append(1.0)  # Complete freedom to choose any board
        else:
            features.append(1.0)  # Free choice
        
        return np.array(features)
    
    def evaluate_local_board_quality(self, local_board):
        """Evaluate how favorable a local board position is for Player 1"""
        p1_advantage = 0.0
        
        # Check each win pattern
        for pattern in self.win_patterns:
            p1_count = 0
            p2_count = 0
            empty_count = 0
            
            for k, l in pattern:
                cell = local_board[k][l]
                if cell == 1:
                    p1_count += 1
                elif cell == 2:
                    p2_count += 1
                elif cell == 0:
                    empty_count += 1
            
            # Calculate advantage for this pattern
            if p1_count > 0 and p2_count == 0:
                if p1_count == 1:
                    p1_advantage += 0.1
                elif p1_count == 2:
                    p1_advantage += 0.5
            
            if p2_count > 0 and p1_count == 0:
                if p2_count == 1:
                    p1_advantage -= 0.1
                elif p2_count == 2:
                    p1_advantage -= 0.5
        
        return p1_advantage
    
    def evaluate_state(self, state):
        """
        Evaluate a state using the learned weights with enhanced features
        """
        # Extract enhanced features
        features = self.extract_enhanced_features(state)
        
        # Calculate weighted sum (only use the number of weights we have)
        if len(features) > len(self.weights):
            features = features[:len(self.weights)]
        
        evaluation = np.dot(features, self.weights) + self.intercept
        
        # Add bonus for immediate tactical threats
        evaluation += self.evaluate_tactical_threats(state) * 0.3
        
        # Ensure evaluation is within bounds [-1, 1]
        return np.clip(evaluation, -1.0, 1.0)
        
    def evaluate_tactical_threats(self, state):
        """Evaluate immediate tactical threats on the meta-board"""
        bonus = 0.0
        local_board_status = state.local_board_status
        
        # Check for two-in-a-row on the meta-board
        for pattern in self.win_patterns:
            p1_count = 0
            p2_count = 0
            empty_pos = None
            
            for i, j in pattern:
                cell = local_board_status[i][j]
                if cell == 1:
                    p1_count += 1
                elif cell == 2:
                    p2_count += 1
                elif cell == 0:
                    empty_pos = (i, j)
            
            # Two in a row with third position open
            if p1_count == 2 and p2_count == 0 and empty_pos:
                bonus += 0.2
            
            if p2_count == 2 and p1_count == 0 and empty_pos:
                bonus -= 0.2
        
        return bonus
    
    def quick_evaluate(self, state):
        """
        Quick evaluation for move ordering
        Simplified version of full evaluation for speed
        """
        local_board_status = state.local_board_status
        
        # Count boards won by each player
        p1_boards = np.sum(local_board_status == 1)
        p2_boards = np.sum(local_board_status == 2)
        
        # Check for two-in-a-row patterns on meta-board
        p1_two_in_row = 0
        p2_two_in_row = 0
        
        for pattern in self.win_patterns:
            p1_count = 0
            p2_count = 0
            
            for i, j in pattern:
                cell = local_board_status[i][j]
                if cell == 1:
                    p1_count += 1
                elif cell == 2:
                    p2_count += 1
            
            # Only count nearly complete patterns
            if p1_count == 2 and p2_count == 0:
                p1_two_in_row += 1
            
            if p2_count == 2 and p1_count == 0:
                p2_two_in_row += 1
        
        # Check center control
        center_value = 0
        if local_board_status[1, 1] == 1:
            center_value = 0.2
        elif local_board_status[1, 1] == 2:
            center_value = -0.2
        
        # Combine factors with weights from our trained model
        # Using the highest weight values from our actual model
        board_factor = 0.25 * (p1_boards - p2_boards)  # Using our new positive weight
        pattern_factor = 0.44 * (p1_two_in_row - p2_two_in_row)  # Using original weight
        
        return board_factor + pattern_factor + center_value

    def get_opening_move(self, state):
        """Return a strategic opening move if applicable"""
        # Empty board - play center of center board
        if np.sum(state.board != 0) == 0:
            return (1, 1, 1, 1)
        
        # First move response
        if np.sum(state.board != 0) == 1:
            # Find where the opponent played
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            if state.board[i][j][k][l] == 2:
                                # If they played in center, take corner
                                if (i,j,k,l) == (1,1,1,1):
                                    return (1, 1, 0, 0)  # Top-left corner of center board
                                # If they sent us to center board, take center
                                if (k,l) == (1,1):
                                    return (1, 1, 1, 1)
        
        return None
