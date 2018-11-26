import preprocess
import nltk

train_dir = './datasets/train/'
test_dir = './datasets/test/testing-PHI-Gold-fixed/'

train_gold_path = './train.gold'
test_gold_path = './test.gold'

def is_digit(token):
    return str(token.isdigit())

def contains_digit(token):
    return str(any(char.isdigit() for char in token))

def is_capitalized(token):
    return str(token[0].isupper())

def is_upper(token):
    return str(token.isupper())

def get_length(token):
    return str(len(token))

def get_contexts(index, tokens):
    prev, next = "" , ""

    if index == 0:
        prev = "<s>"
        next = tokens[index+1]

    elif index == len(tokens) - 1:
        next = "</s>"
        prev = tokens[index-1]

    else:
        prev = tokens[index-1]
        next = tokens[index+1]

    return prev, next


def get_BIO(index, token, tokens, entities, tags):
    prev, next = get_contexts(index, tokens)
    s = prev+" "+token
    if s in entities:
        return 'I'
    elif token in entities:
        return 'B'
    return 'O'

def write_to_file(path_in, path_out):

    data_dicts = preprocess.parse_xml(path_in)

    file = open(path_out,'w')
    n = 1
    for dict in data_dicts:
        print("processing file "+ str(n) + "/" + str(len(data_dicts)) + " in " +path_in)
        text = dict['text']
        tags = dict['tags']
        tokens = dict['tokens']
        entities = [tag['text'] for tag in tags]
        pos_tags = nltk.pos_tag(tokens)
        for i in range(len(tokens)):
            token = tokens[i]
            file.write(token + '\t')
            file.write(pos_tags[tokens.index(token)][1] + '\t')
            file.write(is_digit(token) + '\t')
            file.write(contains_digit(token) + '\t')
            file.write(is_capitalized(token) + '\t')
            file.write(is_upper(token) + '\t')
            file.write(get_length(token) + '\t')
            # contexts = get_contexts(i, tokens)
            # for c in contexts:
            #     file.write(c + '\t')
            file.write(get_BIO(i, token, tokens, entities, tags))
            file.write('\n')
        file.write('\n\n')
        n += 1

write_to_file(train_dir, train_gold_path)
write_to_file(test_dir, test_gold_path)
