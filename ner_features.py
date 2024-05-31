from get_data import read_data
import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

def extract_ner_features(sent):
    """
    Extracts Named Entity Recognition (NER) features from a given sentence.

    Parameters:
    - sent (dict): A dictionary representing a sentence containing linguistic annotations and metadata, including the text of the 
      sentence ('SENT_TEXT') and features ('FEATURES') such as tokenization and part-of-speech tagging.

    Returns:
    - dict: The input sentence dictionary updated with NER features.
    """
       
    # Process the text with the Spacy NLP model
    doc = nlp(sent['SENT_TEXT'])

    # Initialize an empty list to hold the BIO tags
    bio_tags = ["O"] * len(doc)

    # Iterate over each entity in the doc
    for ent in doc.ents:
        # For each entity, set the tag of the first token to "B-" + entity label
        bio_tags[ent.start] = "B-" + ent.label_
        # For the rest of the tokens in the entity, set the tag to "I-" + entity label
        for i in range(ent.start + 1, ent.end):
            bio_tags[i] = "I-" + ent.label_

    # Save the tokens and their BIO tags to a list of tuples
    
    tokens = [t['TOKEN'] for t in sent['FEATURES']]
    i = 0
    for token, bio_tag in zip(doc, bio_tags):
        j = i
        while j < len(tokens) and token.text in tokens[j]:
            sent['FEATURES'][j]['NER'] = bio_tag
            j += 1
        i += 1

    return sent


if __name__ == "__main__":

    # Read the data from the 'train' file
    sentences = read_data('train')
    sent = sentences[0]
    print(sent['SENT_TEXT'], extract_ner_features(sent))
    
