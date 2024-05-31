import pandas as pd
from tqdm import tqdm 
import pickle
import os

def read_data(file_type):
    """
    Read data from a CoNLL-U formatted file and parse it into sentences and tokens.
    Automatically checks if 'data/en_ewt-up-{file_type}.conllu' exists, if not asks for file_path

    Parameters:
    - file_type (str): The type of file to read (optional). 

    Returns:
    - tuple: A tuple containing a dictionary mapping document IDs and sentence IDs to sentence texts and a dataframe with the tokens information.
    """
    
    # Find filepath
    file_path = f'data/en_ewt-up-{file_type}.conllu'
    if not os.path.exists(file_path):
        file_path = input(f'Please provide the file path to the {file_type} dataset:\n')

    # Read data from the CoNLL-U formatted file
    with open(file_path, mode='r', encoding='utf-8') as f:
            sentences = []  # Dictionary to store sentences with their document IDs
            pred_sentences = []
            pred = 0
            sentence_id = ""
            sentence_text = ""
            doc_id = ""
            for line in tqdm(f.readlines()):
                line = line.strip('\n')
                # Extract document ID
                if line.startswith('# newdoc id'):
                    doc_id = line.split("= ")[1]
                # Extract sentence ID
                elif line.startswith('# sent_id'):
                    sentences.extend(pred_sentences)
                    #print(pred)
                    pred_sentences = []
                    pred = 0
                    sentence_id = line.split("= ")[1].replace(doc_id + '-', '')
                # Extract sentence text and store in the sentences dictionary
                elif line.startswith('# text'):
                    sentence_text = line.split("= ")[1]
                else:
                    # Parse token information
                    row = line.strip().split('\t')
                    if len(row) >= 11:  # Ensure all needed columns are present 
                        if row[10] != '_':
                            pred += 1    
                        
                        for i in range(11,len(row)):
                            if row[0] == '1':
                                pred_sentences.append({
                                    'DOC_ID': doc_id,
                                    'SENT_ID': sentence_id,
                                    'SENT_TEXT': sentence_text,
                                    'PRED_ID': str(i-11),
                                    'FEATURES': []
                                })
                            
                            if pred == i-10 and row[10] != '_':
                                pred_sentences[i-11]['PRED_FRAME'] = row[10]
                                pred_sentences[i-11]['PRED_TOKEN'] = row[1]
                                pred_sentences[i-11]['PRED_TOKEN_ID'] = row[0]
                            
                            pred_sentences[i-11]['FEATURES'].append({
                                'TOKEN_ID': row[0],
                                'TOKEN': row[1],
                                'LEMMA': row[2],
                                "UPOS": row[3],
                                "DEPHEAD":  row[6],
                                "DEPREL": row[7], 
                                "PRED": row[10] if pred == i-10 else '_',
                                "ROLE": row[i] if row[i] != 'V' and row[i] != 'C-V' else '_'
                                })
                            
            sentences.extend(pred_sentences)              
    
    return sentences

def convert_data(sentences, file_type):
    """
    Converts a list of sentence dictionaries to a .conllu format file.

    Parameters:
    - sentences (list of dict): A list of dictionaries, where each dictionary contains details of a sentence including its document 
      ID ('DOC_ID'), sentence ID ('SENT_ID'), text ('SENT_TEXT'), predicate ID ('PRED_ID'), and annotated features ('FEATURES').
    - file_type (str): A string indicating the type of the data being converted (e.g., 'train', 'test', 'dev'), which is used to name 
      the output file.

    Outputs:
    - A new file named 'converted-{file_type}.conllu' in the 'data' directory containing the converted sentences.
    
    """
    with open(f'data/converted-{file_type}.conllu', 'w', encoding='utf-8') as f:
        for sentence in tqdm(sentences):
            f.write('\n' + sentence['DOC_ID'] + '\n')
            f.write('sent_id = ' + sentence['DOC_ID'] + '-' + sentence['SENT_ID'] + '\n')
            f.write(sentence['SENT_TEXT'] + '\n')
            f.write(sentence['PRED_ID'] + '\n')
            for tokens in sentence['FEATURES']:
                f.write('\t'.join(tokens.values()) + '\n')
            

    print(f'\n The dataset is saved in data/converted-{file_type}.conllu')

if __name__ == "__main__":
    # Prompt the user to select the file type
    file_types = ['train', 'dev', 'test']
    print("Which file do you want to read?")
    for i, f in enumerate(file_types):
        print(f"({i+1})\t the {f} dataset")
    idx = int(input("Please provide the index: "))
    file_type = file_types[idx-1]

    print('\n Loading data...')
    sentences = read_data(file_type)
    convert_data(sentences, file_type)

