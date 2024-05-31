import os
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    """
    Read data from a CoNLL-U formatted file, count sentences, tokens, and provide detailed predicate and argument statistics.

    Parameters:
    - file_path (str): Path to the CoNLL-U formatted file.

    Returns:
    - int: Number of sentences in the file.
    - int: Number of tokens in the file.
    - int: Number of sentences without predicates.
    - int: Number of sentences with predicates.
    - float: Percentage of sentences without predicates.
    - float: Percentage of sentences with predicates.
    - int: Number of unique predicates.
    - set: Set of unique predicates.
    - int: Number of unique arguments.
    - set: Set of unique arguments.
    """
    num_sentences = 0
    num_tokens = 0
    num_sentences_without_predicate = 0
    unique_predicates = set()
    unique_arguments = set()
    argument_counts = {}
    has_predicate = False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('# sent_id'):
                if not has_predicate:
                    num_sentences_without_predicate += 1
                num_sentences += 1
                has_predicate = False  # Reset for the next sentence
            
            elif not line.startswith('#') and line.strip() != '':
                num_tokens += 1
                columns = line.strip().split('\t')
                
                if len(columns) > 10 and columns[10] != '_':
                    has_predicate = True
                    unique_predicates.add(columns[10])
                
                # Collect arguments from columns 12 onwards
                for argument in columns[11:]:
                    if argument != '_' and argument not in ['C-V', 'R-V', 'V']:
                        unique_arguments.add(argument)
                        normalized_argument = argument.lstrip('C-').lstrip('R-') # We remove C- and R- from the argument categories to reduce categories for plots
                        argument_counts[normalized_argument] = argument_counts.get(normalized_argument, 0) + 1
                    
    num_sentences_with_predicate = num_sentences - num_sentences_without_predicate
    percent_without_predicate = (num_sentences_without_predicate / num_sentences) * 100
    percent_with_predicate = (num_sentences_with_predicate / num_sentences) * 100
    average_tokens_per_sentence = num_tokens / num_sentences 

    return (num_sentences, num_tokens, num_sentences_without_predicate, num_sentences_with_predicate, 
            percent_without_predicate, percent_with_predicate, unique_predicates, unique_arguments,
            argument_counts, average_tokens_per_sentence)

def analyze_files_in_directory(directory):
    """
    Analyze all files in the given directory that start with 'en_ewt-up'.

    Parameters:
    - directory (str): The directory containing the files to be analyzed.
    """
    for filename in os.listdir(directory):
        if filename.startswith('en_ewt-up') and filename.endswith('.conllu'):
            file_path = os.path.join(directory, filename)

            if 'train' in filename:
                dataset_title = 'Training Set'
            elif 'test' in filename:
                dataset_title = 'Test Set'
            elif 'dev' in filename:
                dataset_title = 'Development Set'
            
            print(f"\nAnalyzing {dataset_title}:")
            stats = read_data(file_path)
            print(f"Total sentences: {stats[0]}")
            print(f"Total tokens: {stats[1]}")
            print(f"Sentences without predicates: {stats[2]}")
            print(f"Sentences with predicates: {stats[3]}")
            print(f"Percentage of sentences without predicates: {stats[4]:.2f}%")
            print(f"Percentage of sentences with predicates: {stats[5]:.2f}%")
            print(f"Number of unique predicates: {len(stats[6])}")
            # print(f"Unique predicates: {stats[6]}")
            print(f"Number of unique arguments: {len(stats[7])}")
            print(f"Unique arguments: {stats[7]}")
            print(f"Average number of tokens per sentence: {stats[9]:.2f}")
            
            # Sorting arguments by counts in decreasing order
            sorted_arguments = sorted(stats[8].items(), key=lambda x: x[1], reverse=True)
            arguments, counts = zip(*sorted_arguments)
            
            # Plotting the argument distribution for the current file
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(arguments)))
            bars = plt.bar(arguments, counts, color=colors)

            # Annotating each bar with its count
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height, f'{count}', ha='center', va='bottom')
            
            plt.title(f'Argument Distribution in the {dataset_title}')
            plt.xticks(rotation=90)
            plt.ylabel('Frequency')
            plt.xlabel('Argument Types')
            plt.tight_layout()
            plt.show()
            
directory = '../data/'
analyze_files_in_directory(directory)

