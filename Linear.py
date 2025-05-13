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
        
        # Weights from the trained model with enhanced features
        self.weights = np.array([0.0, -0.01866939910691078, 0.006075848470799264, 
                                0.4267182083301891, -0.44714504251496195, 0.003733991901971591, 
                                0.015585162731198808, 0.43182278719735995, -0.4130971572680618, 
                                0.0028229165085288015, 0.002333341285093204, 0.3838973696689166, 
                                -0.3773637999862251, 0.0037953331124183417, 0.0033816502501223065, 
                                0.42093355721581377, -0.43348390500152145, -0.0027602956245603187, 
                                0.0007708340236098652, 0.3964216697949532, -0.4054068799981872, 
                                -0.013147528261756776, 0.019841868292203798, 0.44508399275884103, 
                                -0.46239215104259573, -0.0034033969981267552, -0.01939972667233872, 
                                0.4455866987183704, -0.4174738702502634, -0.004795564227667128, 
                                0.0062087973124694724, 0.4411492351112429, -0.3913780397174879, 
                                -0.012112490706170292, 0.023994352656465347, -0.005476183281036877, 
                                0.21464568566021042, 0.008572170312882574, -0.012152372504386598, 
                                -0.03252664109445973, 0.05237348211621291, -0.04681270321788112, 
                                0.03270540307660656, 0.025646432566436107, -0.02905523901025117, 
                                -0.08235297231181155, 0.00991861852947861, 0.03194227155274275])
        
        # Intercept from the trained model
        self.intercept = -0.042351531839322436

        for i in range(8):  # 8 win patterns
            p1_two_idx = 4*i + 3  # Index for P1 two-in-row pattern value
            p2_two_idx = 4*i + 4  # Index for P2 two-in-row pattern value
            
            if p1_two_idx < len(self.weights):
                self.weights[p1_two_idx] *= 1.5  # Increase importance
            if p2_two_idx < len(self.weights):
                self.weights[p2_two_idx] *= 1.5  # Increase importance
        
        # Board control (indexes 36-38)
        center_board_idx = 36  # Center board control
        if center_board_idx < len(self.weights):
            self.weights[center_board_idx] *= 2.0  # Center board is critical
        
        # Direct board count differential (indexes ~35-36)
        board_count_idx = 35  # Check if this is correct for your feature ordering
        if board_count_idx < len(self.weights):
            self.weights[board_count_idx] *= 1.3
    
    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board."""
        #state = state.invert()
        self.start_time = time.time()
        
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
        """Extract more strategic features from the state"""
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
        
        # --- FEATURE GROUP 3: STRATEGIC BOARD POSITION FEATURES ---
        
        # Center and corner control
        center_board = (1, 1)
        corner_boards = [(0, 0), (0, 2), (2, 0), (2, 2)]
        edge_boards = [(0, 1), (1, 0), (1, 2), (2, 1)]
        
        # Center board value
        if local_board_status[center_board] == 1:
            features.append(1)
        elif local_board_status[center_board] == 2:
            features.append(-1)
        else:
            features.append(0)
        
        # Corner boards value
        corner_value = 0
        for i, j in corner_boards:
            if local_board_status[i][j] == 1:
                corner_value += 1
            elif local_board_status[i][j] == 2:
                corner_value -= 1
        features.append(corner_value)
        
        # Edge boards value
        edge_value = 0
        for i, j in edge_boards:
            if local_board_status[i][j] == 1:
                edge_value += 1
            elif local_board_status[i][j] == 2:
                edge_value -= 1
        features.append(edge_value)
        
        # --- FEATURE GROUP 4: LOCAL BOARD TACTICAL FEATURES ---
        
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
        
        # --- FEATURE GROUP 5: NEXT MOVE STRATEGIC FEATURES ---
        
        # Next board to play on features
        if hasattr(state, 'prev_local_action') and state.prev_local_action is not None:
            prev_k, prev_l = state.prev_local_action
            
            # Is the next board active, won, or drawn?
            if local_board_status[prev_k][prev_l] == 0:
                features.append(1.0)  # Active - forced move
                
                # Evaluate quality of the forced local board
                forced_board_value = self.evaluate_local_board_quality(board[prev_k][prev_l])
                features.append(forced_board_value)
            else:
                features.append(0.0)  # Free choice of board
                features.append(0.0)  # No forced board value
                
            # Strategic value of next board position
            if (prev_k, prev_l) == center_board:
                features.append(0.5)  # Center has highest value
            elif (prev_k, prev_l) in corner_boards:
                features.append(0.3)  # Corners have good value
            elif (prev_k, prev_l) in edge_boards:
                features.append(0.2)  # Edges have less value
            else:
                features.append(0.0)  # Shouldn't happen
        else:
            # First move or no previous action
            features.append(0.0)  # Free choice
            features.append(0.0)  # No forced board
            features.append(0.0)  # No specific position
        
        # --- FEATURE GROUP 6: GAME PHASE AND TEMPO FEATURES ---
        
        # Game phase (early, mid, late)
        total_filled = np.sum(board != 0)
        features.append(total_filled / 81.0)  # Normalized game progress
        
        # Move tempo - who has the initiative?
        # If player has more two-in-rows than opponent, they have better tempo
        tempo = (p1_local_two - p2_local_two) / 10.0 if (p1_local_two + p2_local_two) > 0 else 0
        features.append(tempo)
        
        # Freedom factor - how many choices does the player have?
        if hasattr(state, 'prev_local_action') and state.prev_local_action is not None:
            prev_k, prev_l = state.prev_local_action
            if local_board_status[prev_k][prev_l] == 0:
                # Count empty cells in the forced board
                empty_count = np.sum(board[prev_k][prev_l] == 0)
                features.append(empty_count / 9.0)  # Normalized freedom
            else:
                features.append(1.0)  # Complete freedom to choose any board
        else:
            features.append(1.0)  # Free choice at start of game
        
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
        
        # Ensure evaluation is within bounds [-1, 1]
        return np.clip(evaluation, -1.0, 1.0)
    
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
        board_factor = 0.21 * (p1_boards - p2_boards)
        pattern_factor = 0.44 * (p1_two_in_row - p2_two_in_row)
        
        return board_factor + pattern_factor + center_value
