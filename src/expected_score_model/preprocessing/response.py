import numpy as np

# Define a mapping of final states to result categories
result_map = {
    'goal': 'goal',
    'behind': 'behind',
    'turnover': 'miss',
    'rushed': 'miss',
    'outOfBounds': 'miss',
    'ballUpCall': 'miss',
    'endQuarter': 'miss',
    'rushedOpp': 'miss',
}

# Function to create a result based on the final state
def create_result(final_state):
    """
    Replace the final state with the corresponding result category.
    
    Args:
        final_state (str): The final state of the play.
        
    Returns:
        str: The result category corresponding to the final state.
    """
    return final_state.replace(result_map)

# Define a mapping of result categories to scores
score_map = {
    'goal': 6,
    'behind': 1,
    'miss': 0,
}

# Function to create a score based on the result
def create_score(result):
    """
    Replace the result category with the corresponding score.
    
    Args:
        result (str): The result category of the play.
        
    Returns:
        int: The score corresponding to the result category.
    """
    return result.replace(score_map)

# Function to create the expected score response
def create_expected_score_response(X):
    """
    Create the expected score response by calculating the result and score.
    
    Args:
        X (dict): A dictionary containing the final state of the play.
        
    Returns:
        dict: The updated dictionary with result and score.
    """
    X['result'] = create_result(X['Final_State'])
    X['score'] = create_score(X['result'])
    X['miss'] = np.where(X['result'] == 'miss', 1, 0)
    X['behind'] = np.where(X['result'] == 'behind', 1, 0)
    X['goal'] = np.where(X['result'] == 'goal', 1, 0)
    
    return X
