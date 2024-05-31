import os

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
    has_predicate = False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == '': # New sentence
                if not has_predicate:
                    num_sentences_without_predicate += 1
                num_sentences += 1
                has_predicate = False

            else:
                columns = line.strip().split('\t')
                if len(columns) >= 10:
                    num_tokens += 1
                    if len(columns) > 10 and columns[10] != '_':  # Check for predicate
                        has_predicate = True
                        unique_predicates.add(columns[10])
                    if len(columns) > 11 and columns[11] != '_':  # Check for argument
                        unique_arguments.add(columns[11])
    
    num_sentences_without_predicate -= 1 # because files begin with an empty line which should not be counted as a sentence
    num_sentences_with_predicate = num_sentences - num_sentences_without_predicate
    percentage_without_predicate = (num_sentences_without_predicate / num_sentences) * 100
    percentage_with_predicate = (num_sentences_with_predicate / num_sentences) * 100
    num_unique_predicates = len(unique_predicates)
    num_unique_arguments = len(unique_arguments)
    average_tokens_per_sentence = num_tokens / num_sentences 


    return (num_sentences, num_tokens, num_sentences_without_predicate, num_sentences_with_predicate,
            percentage_without_predicate, percentage_with_predicate, num_unique_predicates,
            unique_predicates, num_unique_arguments, unique_arguments, average_tokens_per_sentence)
    

def analyze_files_in_directory(directory):
    """
    Analyze all files in the given directory that start with 'en_ewt-up'.

    Parameters:
    - directory (str): The directory containing the files to be analyzed.
    """

    for filename in os.listdir(directory):
        if filename.startswith('converted') and filename.endswith('.conllu'):
            file_path = os.path.join(directory, filename)
            stats = read_data(file_path)
            print(f"\nAnalyzing {filename}:")
            print(f"Total sentences: {stats[0]}")
            print(f"Total tokens: {stats[1]}")
            print(f"Sentences without predicates: {stats[2]}")
            print(f"Sentences with predicates: {stats[3]}")
            print(f"Percentage of sentences without predicates: {stats[4]:.2f}%")
            print(f"Percentage of sentences with predicates: {stats[5]:.2f}%")
            print(f"Number of unique predicates: {stats[6]}")
            # print(f"Unique predicates: {stats[7]}")
            print(f"Number of unique arguments: {stats[8]}")
            # print(f"Unique arguments: {stats[9]}")
            print(f"Average number of tokens per sentence: {stats[10]:.2f}")

            
directory = '../data/'
analyze_files_in_directory(directory)


