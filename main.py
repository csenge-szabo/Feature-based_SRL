from context_features import extract_pred_features
from ner_features import extract_ner_features
from semantic_features import extract_semantic_features
from get_data import read_data
from dependency_features import extract_dependency_features
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import propbank
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse import hstack  # Changed from np.hstack to hstack to handle sparse matrices


def extract_features(dataset, vectorizer, pred_vectorizer):
    """
    Extracts features for the given dataset using the provided vectorizers. 
    
    Parameters:
    - dataset (str): The name of the dataset ('train' or 'test').
    - vectorizer (DictVectorizer): Vectorizer for converting feature dictionaries into feature vectors.
    - pred_vectorizer (CountVectorizer): Vectorizer for converting predicate arguments into feature vectors.
    
    Returns:
        tuple: Tuple containing feature matrix, gold labels, vectorizer, and predicate vectorizer.
    """
    raw_sentences = read_data(dataset)
    
    if not os.path.exists(f'predicates/{dataset}.pkl'):
        propbank.main(dataset)
    
    with open(f'predicates/{dataset}.pkl', "rb") as f:
        preds_dict = pickle.load(f)

    features = []
    golds = []
    args = []
    args2feat = []

    for sent in tqdm(raw_sentences):
        # Extract different features
        sent_features = extract_ner_features(sent)
        sent_features = extract_pred_features(sent_features)
        sent_features = extract_semantic_features(sent_features)
        sent_features = extract_dependency_features(sent_features)
        
        for token in sent_features['FEATURES']:
            # Get labels out, and delete from the data
            golds.append(token['ROLE'])
            for key in ['ROLE', 'LEMMA', 'TOKEN', 'PRED', 'DEPHEAD', 'TOKEN_ID', 'PRED_ID']:
                if key in token:
                    del token[key]
            
            # Get the arguments from propbank
            try:
                token_args = [a for a in preds_dict[sent['PRED_FRAME']] if 'arg' in a.lower()]
                args2feat.extend(token_args)
                args.append(' '.join(token_args))
            except KeyError:
                args.append("")
                
            features.append(token)
    
    if dataset == 'train':
        feature_matrix = vectorizer.fit_transform(features)
        pred_vectorizer = pred_vectorizer.fit(args2feat)
    else:
        feature_matrix = vectorizer.transform(features)

    args_features_matrix = pred_vectorizer.transform(args)

    # Use sparse hstack to combine feature matrices
    #feature_matrix = hstack([feature_matrix, args_features_matrix]).toarray()

    return feature_matrix, golds, vectorizer, pred_vectorizer


def train_model(train_data, train_labels):
    """
    Trains a logistic regression model on the training data.
    
    Parameters:
    train_data: Feature matrix for the training data.
    train_labels: Labels for the training data.
    """
    print("Training the logistic regression model...")
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
    model.fit(train_data, train_labels)

    # Save the model to a file
    with open('trained_logistic_regression_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)


def load_and_evaluate(test_data, test_labels):
    """
    Loads a pre-trained logistic regression model and evaluates it on the test data.
    
    Parameters:
    test_data: Feature matrix for the test data.
    test_labels: Labels for the test data.
    """    
    with open('trained_logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Predict on the test set
    print("Predicting on the test set...")
    predictions = model.predict(test_data)

    # Evaluate the model
    print("Evaluating the model...")
    report = classification_report(test_labels, predictions)
    print("Classification Report:\n", report)
    plot_confusion_matrix(test_labels, predictions, model.classes_)

    

def plot_confusion_matrix(test_labels, predictions, classes):
    """
    Plots a confusion matrix using the actual and predicted labels.
    
    Parameters:
    test_labels: Actual labels.
    predictions: Predicted labels by the model.
    classes: List of unique class labels.
    """
    cm = confusion_matrix(test_labels, predictions, labels=classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.title('Confusion Matrix')
    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    plt.colorbar()  # Add a colorbar to a plot
    plt.show()

if __name__ == "__main__":
    vectorizer = DictVectorizer(sparse=True)
    pred_vectorizer = CountVectorizer()
    model_path = 'trained_logistic_regression_model.pkl'

    # Check if the trained model file already exists
    if not os.path.exists(model_path):
        
        # Check if features are already extracted
        if os.path.exists('features/train.pkl') and os.path.exists('features/train_labels.pkl') and os.path.exists('features/test.pkl') and os.path.exists('features/test_labels.pkl'):
            
            print("Model file not found, starting and model training...")
            with open(f'features/train.pkl', 'rb') as f:
                train_features = pickle.load(f)
            with open(f'features/train_labels.pkl', 'rb') as f:
                train_labels = pickle.load(f)

            with open(f'features/test.pkl', 'rb') as f:
                test_features = pickle.load(f)
            with open(f'features/test_labels.pkl', 'rb') as f:
                test_labels = pickle.load(f)

        else:
            print("Model file and feature datasets not found, starting feature extracting and model training...")
        
            train_features, train_labels, vectorizer, pred_vectorizer = extract_features('train', vectorizer, pred_vectorizer)
            test_features, test_labels, vectorizer, pred_vectorizer = extract_features('test', vectorizer, pred_vectorizer)

            with open(f'features/train.pkl', 'rb') as f:
                pickle.dump(train_features, f)
            with open(f'features/train_labels.pkl', 'rb') as f:
                pickle.dump(train_labels, f)

            with open(f'features/test.pkl', 'rb') as f:
                pickle.dump(test_features, f)
            with open(f'features/test_labels.pkl', 'rb') as f:
                pickle.dump(test_labels, f)

        # Train and evaluate the logistic regression model
        train_model(train_features, train_labels)

    else:
        print("Model file found, loading model and evaluating...")
        
        # Load test datasets
        with open(f'features/test.pkl', 'rb') as f:
            test_features = pickle.load(f)
        with open(f'features/test_labels.pkl', 'rb') as f:
            test_labels = pickle.load(f)
        
        
        # Load and evaluate the logistic regression model
        load_and_evaluate(test_features, test_labels)


    

