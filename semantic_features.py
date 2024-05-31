from get_data import read_data
from tqdm import tqdm
import spacy
from spacy.matcher import Matcher
import gensim

nlp = spacy.load("en_core_web_sm")

# Create pattern to match passive voice use
passive_rules = [
        [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}],
        [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBZ'}],
        [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'RB'}, {'TAG': 'VBN'}],
    ]
    # Create pattern to match active voice use
active_rules = [
        [{'DEP': 'nsubj'}, {'TAG': 'VBD', 'DEP': 'ROOT'}],
        [{'DEP': 'nsubj'}, {'TAG': 'VBP'}, {'TAG': 'VBG', 'OP': '!'}],
        [{'DEP': 'nsubj'}, {'DEP': 'aux', 'OP': '*'}, {'TAG': 'VB'}],
        [{'DEP': 'nsubj'}, {'DEP': 'aux', 'OP': '*'}, {'TAG': 'VBG'}],
        [{'DEP': 'nsubj'}, {'TAG': 'RB', 'OP': '*'}, {'TAG': 'VBG'}],
        [{'DEP': 'nsubj'}, {'TAG': 'RB', 'OP': '*'}, {'TAG': 'VBZ'}],
        [{'DEP': 'nsubj'}, {'TAG': 'RB', 'OP': '+'}, {'TAG': 'VBD'}],
    ]

matcher = Matcher(nlp.vocab)  # Init. the matcher with a vocab (note matcher vocab must share same vocab with docs)
matcher.add('Passive',  passive_rules)  # Add passive rules to matcher
matcher.add('Active', active_rules)  # Add active rules to matcher

# word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)   

def extract_semantic_features(sentence):
    """
    Extracts semantic features from a given sentence.

    Parameters:
    - sentence (dict): A dictionary representing a sentence containing linguistic annotations and metadata, including the text 
      of the sentence ('SENT_TEXT'), the tokenized features ('FEATURES'), and the predicate token ('PRED_TOKEN').

    Returns:
    - dict: The input sentence dictionary updated with semantic features.
    """
    
    sent = nlp(sentence['SENT_TEXT'])
    voice = {}
    for match_id, start, end in matcher(sent):
        string_id = nlp.vocab.strings[match_id]
        for i in range(start, end):
            voice[sent[i]] = string_id

    pred_voice = voice[sentence['PRED_TOKEN']] if sentence['PRED_TOKEN'] in voice.keys() else '-'
    
    # pred_emb = [0]*300 if sentence['PRED_TOKEN'] not in word_embedding_model else list(word_embedding_model[sentence['PRED_TOKEN']])
    
    # Calculate the token distance from the predicate for each token
    for i, token_dict in enumerate(sentence['FEATURES']): 
        # sentence['FEATURES'][i]['NEXT_LEMMA'] = [0]*300 if i+1 == len(sentence['FEATURES']) or sentence['FEATURES'][i+1]['LEMMA'] not in word_embedding_model else list(word_embedding_model[sentence['FEATURES'][i+1]['LEMMA']])
        # if i == 0:    
        #     sentence['FEATURES'][i]['LEMMA_EMB'] = [0]*300 if token_dict['LEMMA'] not in word_embedding_model else list(word_embedding_model[token_dict['LEMMA']])
        #     sentence['FEATURES'][i]['PREV_LEMMA'] = [0]*300
        # else:
        #     sentence['FEATURES'][i]['PREV_LEMMA'] = sentence['FEATURES'][i-1]['LEMMA_EMB'] if i != 0 else ''
        #     sentence['FEATURES'][i]['LEMMA_EMB'] = sentence['FEATURES'][i-1]['NEXT_LEMMA']
        
        token_dict['CURR_LEMMA'] = token_dict['LEMMA']  # Current lemma
        token_dict['PREV_LEMMA'] = sentence['FEATURES'][i-1]['LEMMA'] if i != 0 else ''  # Previous lemma
        token_dict['NEXT_LEMMA'] = sentence['FEATURES'][i+1]['LEMMA'] if i+1 < len(sentence['FEATURES']) else ''  # Next lemma

        sentence['FEATURES'][i]['PREV_UPOS'] = sentence['FEATURES'][i-1]['UPOS'] if i != 0 else ''
        sentence['FEATURES'][i]['NEXT_UPOS'] = sentence['FEATURES'][i+1]['UPOS'] if i+1 != len(sentence['FEATURES']) else ''
        sentence['FEATURES'][i]['VOICE'] = pred_voice
        # sentence['FEATURES'][i]['PRED_EMB'] = pred_emb
    
    return sentence

if __name__ == "__main__":
    file_type = 'train'
    sentences = read_data(file_type)
    features = []
    for sent in tqdm(sentences):
        features.append(extract_semantic_features(sent))
    #features_df = extract_features(file_path)
    print(features[5])
    #print(features_df.head(50))
