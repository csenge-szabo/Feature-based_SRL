
import numpy as np
from get_data import read_data


def extract_dependency_features(sentence):
    """
    Extracts dependency-based features for each token in a given sentence.
  
    Parameters:
    - sentence (dict): A dictionary representing a single sentence. It must contain the following keys:
      - 'PRED_TOKEN_ID': The token ID of the predicate (main verb) of the sentence.
      - 'FEATURES': A list of dictionaries, each representing a token in the sentence with its features.
        Each token's features should include 'TOKEN_ID', 'DEPHEAD' (dependency head ID), and 'DEPREL' (dependency relation).

    Returns:
    - dict: The modified sentence dictionary including new keys in each token's feature dictionary:
      - 'DEPENDENCY_HEAD_TOKEN': The head token of the current token in the dependency tree, or 'ROOT' if the token is the root.
      - 'DEPENDENCY_PATH': A list representing the path of dependency relations from the token to the predicate.
      - 'DEPENDENCY_DISTANCE': The distance from the token to the predicate in the dependency tree.

    """
    # Find ancestors of the predicate
    current_id = int(sentence['PRED_TOKEN_ID'])
    pred_ancestors = []
    
    while True:
        pred_ancestors.append(current_id)
        head_id = int(sentence['FEATURES'][current_id-1]['DEPHEAD'])
        if head_id == 0:  # Reached root or loop
            break
        current_id = head_id  # Move to the dependency head for next iteration
    
    # Process each token to extract features
    for i, token_feature in enumerate(sentence['FEATURES']):
        token_id = token_feature['TOKEN_ID']

         # Find the lemma embedding corresponding to the head ID (if it exists)
        head_id = token_feature['DEPHEAD']
        head_token = sentence['FEATURES'][int(head_id)-1]['LEMMA'] if head_id != '0' else 'ROOT'

        dependency_path = []
        current_id = int(token_id)
        distance = 0

        # Go from token to ancestor of the predicate
        while distance < len(sentence['FEATURES']):
            distance += 1  
            head_id = int(sentence['FEATURES'][current_id-1]['DEPHEAD'])
            if head_id in pred_ancestors or head_id == current_id:  # Reached pred ancestors or loop
                break
            
            dependency_path.append(f"up {sentence['FEATURES'][current_id-1]['DEPREL']}")
            current_id = head_id  # Move to the dependency head for next iteration
        
        # Extend the path going down to the predicate
        for anc_id in pred_ancestors:
            if head_id == anc_id:
                break
            dependency_path.append(f"down {sentence['FEATURES'][anc_id-1]['DEPREL']}")

        # Store features for this token
        sentence['FEATURES'][i]['DEPENDENCY_HEAD_TOKEN'] = head_token
        sentence['FEATURES'][i]['DEPENDENCY_PATH'] = ' '.join(dependency_path),
        sentence['FEATURES'][i]['DEPENDENCY_DISTANCE'] = distance

        
    return sentence

if __name__ == "__main__":
    file_type = 'train'
    sentences = read_data(file_type)
    sentence_with_dep = extract_dependency_features(sentences[9])
    print(sentence_with_dep)

