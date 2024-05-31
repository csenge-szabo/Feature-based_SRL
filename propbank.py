from nltk.corpus import propbank
import pickle

def extract_arguments(ins):
    """
    This function extracts the arguments from a given instance.

    Args:
        ins: An instance from the PropBank corpus.

    Returns:
        A list of arguments from the given instance.
    """
    result = []
    for arg in ins.arguments:
        result.append(arg[1])
    return result

def fun1(predicate):
    """
    This function finds all roles for a given predicate.

    Args:
        predicate: A predicate for which to find roles.

    Returns:
        A list of roles for the given predicate.
    """
    roleset = propbank.roleset(predicate)
    result = []
    for role in roleset.findall('roles/role'):
        result.append(role.attrib['descr'])
    return result

def fun2(pred):
    """
    This function finds the roles and arguments for a given predicate.

    Args:
        pred: A predicate for which to find roles and arguments.

    Returns:
        A dictionary where the key is the predicate and the value is a list of roles and arguments for the predicate.
    """
    result_dict = {}
    for instances in pb_instances:
        roleset = instances.roleset
        if roleset == pred:
            extract_arguments(instances)
            fun1(pred)
            result_dict[roleset] = fun1(pred) + extract_arguments(instances)

            break
    return result_dict

def get_predicates(file_path):
    """
    This function reads a tsv file and checks the column [10] for predicate.
    When there is a predicate, it applies the fun2 function.

    Args:
        file_path: The path to the tsv file.

    Returns:
        A list of predicates.
    """
    list_of_predicates = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        processed_sentences = []
        for line in file:
            if line.startswith('# sent_id'):
                sentence_id = line.split("= ")[1]
            elif not line.startswith('#') and line.strip() != '':
                columns = line.strip().split('\t')
                if len(columns) >= 11 and columns[10] != '_' and (columns[10] not in processed_sentences):
                    list_of_predicates.update(fun2(columns[10]))
                    processed_sentences.append(columns[10])
                else:
                    continue
            else:
                continue
    return list_of_predicates

def main(file_type):
    # Load all instances from the PropBank corpus
    pb_instances = propbank.instances()
    
    # Call the get_predicates function and store its result
    result = get_predicates(f'data/converted-{file_type}.conllu')

    # Open a file in write-binary mode
    with open(f'predicates/{file_type}.pkl', 'wb') as f:
        # Use pickle.dump to write the data to the file
        pickle.dump(result, f)

if __name__ == "__main__":
    # Prompt the user to select the file type
    file_types = ['train', 'dev', 'test']
    print("Which file do you want to read?")
    for i, f in enumerate(file_types):
        print(f"({i+1})\t the {f} dataset")
    idx = int(input("Please provide the index: "))
    file_type = file_types[idx-1]

    # Load all instances from the PropBank corpus
    pb_instances = propbank.instances()
    
    # Call the get_predicates function and store its result
    result = get_predicates(f'data/converted-{file_type}.conllu')

    # Open a file in write-binary mode
    with open(f'predicates/{file_type}.pkl', 'wb') as f:
        # Use pickle.dump to write the data to the file
        pickle.dump(result, f)

    # { 'come.03': ['thing (state) arising', 'source (from or in or of)', 'ARG1', 'ARG2-from'],
    #   'nominate.01': ['nominator', 'candidate', 'role of arg1', 'ARG0', 'ARG1', 'ARG2'],
    #   'replace.01': ['replacer'....,
    # }