import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils import State, load_data

class UTTTFeatureExtractor:
    """Extract features from UTTT state for linear regression model"""
    def __init__(self):
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
    
    def extract_features(self, state):
        """Extract strategic features from the state"""
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
        
        # Center board value
        if local_board_status[self.center_board] == 1:
            features.append(1)
        elif local_board_status[self.center_board] == 2:
            features.append(-1)
        else:
            features.append(0)
        
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
            if 0 <= prev_k < 3 and 0 <= prev_l < 3 and local_board_status[prev_k][prev_l] == 0:
                features.append(1.0)  # Active - forced move
                
                # Evaluate quality of the forced local board
                forced_board_value = self.evaluate_local_board_quality(board[prev_k][prev_l])
                features.append(forced_board_value)
            else:
                features.append(0.0)  # Free choice of board
                features.append(0.0)  # No forced board value
                
            # Strategic value of next board position
            if (prev_k, prev_l) == self.center_board:
                features.append(0.5)  # Center has highest value
            elif (prev_k, prev_l) in self.corner_boards:
                features.append(0.3)  # Corners have good value
            elif (prev_k, prev_l) in self.edge_boards:
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
            if 0 <= prev_k < 3 and 0 <= prev_l < 3 and local_board_status[prev_k][prev_l] == 0:
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


def train_linear_model():
    """Train a linear regression model to predict UTTT board evaluations"""
    # Load the dataset
    data = load_data()
    
    # Feature extraction
    feature_extractor = UTTTFeatureExtractor()
    
    # Prepare training data
    X = []
    y = []
    
    print("Extracting features from dataset...")
    for state, value in data:
        features = feature_extractor.extract_features(state)
        X.append(features)
        y.append(value)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Check feature dimensions
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Number of samples: {len(y)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models with different regularization strengths
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=0.01)": Ridge(alpha=0.01),
        "Ridge(alpha=0.1)": Ridge(alpha=0.1),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
    }
    
    best_model = None
    best_score = -float('inf')
    best_model_name = ""
    
    print("\nTraining models with different regularizations:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name}:")
        print(f"  Mean Squared Error: {mse:.6f}")
        print(f"  R² Score: {r2:.6f}")
        
        # Track best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with R² score: {best_score:.6f}")
    
    # Get feature importance for best model
    coefficients = best_model.coef_
    intercept = best_model.intercept_
    
    print(f"\nIntercept: {intercept:.6f}")
    
    # Format coefficients as numpy array initialization code
    coef_str = "np.array(["
    for i, coef in enumerate(coefficients):
        if i % 5 == 0 and i > 0:
            coef_str += "\n                                "
        coef_str += f"{coef:.18f}, "
    coef_str = coef_str.rstrip(", ") + "])"
    
    print("\nModel Coefficients (for use in your agent):")
    print(f"self.weights = {coef_str}")
    print(f"self.intercept = {intercept}")
    
    # Plot feature importance
    feature_importance = np.abs(coefficients)
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    top_n = 20  # Show top 20 features
    top_features = sorted_idx[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), feature_importance[top_features], align='center')
    plt.yticks(range(top_n), [f"Feature {i}" for i in top_features])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.savefig('prediction_quality.png')
    
    # Save model weights for manual tweaking
    with open("linear_model_weights.txt", "w") as f:
        f.write(f"Intercept: {intercept}\n\n")
        f.write("Weights:\n")
        for i, coef in enumerate(coefficients):
            f.write(f"Feature {i}: {coef:.8f}\n")
    
    print("\nModel weights saved to linear_model_weights.txt")
    print("Visualizations saved as feature_importance.png and prediction_quality.png")
    
    # Feature engineering insights
    print("\nFeature Engineering Insights:")
    print("----------------------------")
    
    # Identify highly weighted features for two-in-row patterns
    two_in_row_indices = []
    for i in range(8):  # 8 win patterns
        p1_two_idx = 4*i + 3  # Index for P1 two-in-row pattern value
        p2_two_idx = 4*i + 4  # Index for P2 two-in-row pattern value
        
        if p1_two_idx < len(coefficients) and p2_two_idx < len(coefficients):
            two_in_row_indices.extend([p1_two_idx, p2_two_idx])
    
    print(f"Two-in-row pattern coefficients (indices {two_in_row_indices}):")
    for idx in two_in_row_indices:
        if idx < len(coefficients):
            print(f"  Feature {idx}: {coefficients[idx]:.6f}")
    
    # Center board control (typical index 36)
    center_board_idx = 36  # Adjust if your feature ordering is different
    if center_board_idx < len(coefficients):
        print(f"\nCenter board control (Feature {center_board_idx}): {coefficients[center_board_idx]:.6f}")
    
    # Board count differential (typical index 35)
    board_count_idx = 35  # Adjust if your feature ordering is different
    if board_count_idx < len(coefficients):
        print(f"Board count differential (Feature {board_count_idx}): {coefficients[board_count_idx]:.6f}")
    
    return best_model, X, y


if __name__ == "__main__":
    train_linear_model()
