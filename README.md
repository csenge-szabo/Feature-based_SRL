# Advanced-NLP-13

### Introduction
This project was carried out as part of the Advanced NLP Master's course at VU Amsterdam. The aim of this project is to implement a Logistic Regression model with lexical, semantic, syntactic and contextual features extracted from the data for Semantic Role Labeling.

### Authors: 
- Selin Acikel
- Murat Ertas
- Tessel Haagen
- Csenge Szabó

### Data: 
For this project we are utilizing the [Universal Proposition Banks version 1.0](https://universalpropositions.github.io) for English language, which was created with the aim to study semantic role labeling.

### Scripts and Usage:
Please use `python main.py` in the command line, which does the following:
- Preprocessing: sentences containing multiple predicates are duplicated depending on the number of predicates, sentences without predicates are removed, the labels marking the predicate ('V', 'C-V' are removed). If the default filepath is not found, it will ask for a filepath.
- Feature extraction: extracting lexical, dependency-based, semantic and contextual features from the preprocessed training and test data.
- Model training: training the Logistic Regression model using the training data.
- Model predictions: the model predicts the labels in the test set.
- Model evaluation: classification report and confusion matrix.

Statistical distribution: 
- run `statistics.py` to observe label distribution in the raw data
- run `converted_statistics.py` to observe label distribution in the preprocessed data

Extracted features:
- Lemma (current, previous, next)
- Universal POS (current, previous, next)
- Dependency relation
- Dependency head
- Dependency path
- Dependency distance from token to predicate
- Voice of the sentence: active/passive
- Named Entity Class (with BIO scheme)
- Predicate
- Distance from token to predicate
- Token's position (before or after the predicate)
- Predicate arguments (Propbank)
- Predicate roles (Propbank)

Auxiliary scripts, which are utilised in `main.py`:
- `context_features.py`
- `dependency_features.py`
- `ner_features.py`
- `propbank.py`
- `semantic_features.py`
- `get_data.py`









