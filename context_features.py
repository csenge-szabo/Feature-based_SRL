import pandas as pd
from get_data import read_data
from tqdm import tqdm

def extract_features(file_path):
    """
    Extracts features from a file containing annotated sentences.
    Parameters:
    - file_path (str): The path to the file containing the annotated sentences.
    Returns:
    - pandas.DataFrame: A DataFrame where each row contains the features of a token: 
      'Token', 'Token-Predicate Distance', 'Relative Position'.
    """
    features = []

    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence = []
        predicate_index = None
        for line in file:
            if line.strip() == '':
                 if current_sentence: 
                    
                    # Calculate the token distance from the predicate for each token
                    for i, token_data in enumerate(current_sentence):
                        token_word = token_data['TO']
                        if predicate_index is not None:
                            distance = abs(i - predicate_index)
                            if i < predicate_index:
                                relative_pos = 'L'
                            elif i > predicate_index:
                                relative_pos = 'R'
                            else:
                                relative_pos = 0
                        else:
                            distance = 0
                            relative_pos = 0
                        features.append((token_word, distance, relative_pos))
                    current_sentence = []
                    predicate_index = None
            else:
                columns = line.strip().split('\t')
                if len(columns) > 10 and columns[0].isdigit():  # Check if it's a token line
                    # Append token data including the predicate marked in column 11
                    current_sentence.append(columns)
                    if columns[10] != '_':
                        predicate_index = len(current_sentence) - 1

    # Each token's feature is a row in the DataFrame
    return pd.DataFrame(features, columns=['Token', 'Token-Predicate Distance', 'Relative Position'])

def extract_pred_features(sentence):
    """
    Enhances a sentence dictionary with predicate-related features.
    Parameters:
    - sentence (dict): A dictionary representing a sentence, where 'PRED_TOKEN_ID' is the index of the predicate token and 
      'FEATURES' is a list of dictionaries, each representing token features.
    Returns:
    - dict: The same sentence dictionary input, but updated to include 'PRED_DISTANCE', 'RELATIVE_POS', and 'PRED_ID' for each token.
    """

    predicate_index = int(sentence['PRED_TOKEN_ID'])                
    
    # Calculate the token distance from the predicate for each token
    for i, _ in enumerate(sentence['FEATURES']):

        distance = abs(i - predicate_index)
        if i < predicate_index:
            relative_pos = 'L'
        elif i > predicate_index:
            relative_pos = 'R'
        else:
            relative_pos = 0
       
        sentence['FEATURES'][i]['PRED_DISTANCE'] = distance
        sentence['FEATURES'][i]['RELATIVE_POS'] = relative_pos
        sentence['FEATURES'][i]['PRED_ID'] = sentence['PRED_TOKEN_ID']
    
    return sentence

if __name__ == "__main__":
    file_type = 'train'
    sentences = read_data(file_type)
    features = []
    for sent in tqdm(sentences):
        features.append(extract_pred_features(sent))
    #features_df = extract_features(file_path)
    print(features[5])
    #print(features_df.head(50))
