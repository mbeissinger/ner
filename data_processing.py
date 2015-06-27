import os

def find_txt_files(data_dir='data/'):
    for root, dirs, files in os.walk(data_dir):
        for basename in files:
            name, ext = os.path.splitext(basename)
            if ext == ".txt":
                filename = os.path.join(root, basename)
                yield filename

def is_label(val):
    try:
        int(val)
        return False
    except ValueError:
        if str(val) == "O":
            return False
        else:
            return True

def parse_text_files(data_dir='data/'):
    data = []
    labels = []
    NOT_LABEL = "N/A"
    entity_vocab = {NOT_LABEL: 0}
    entity_idx = 1
    vocab = {}
    vocab_idx = 0
    # go through each txt file
    for filename in find_txt_files(data_dir):
        try:
            with open(filename, mode='r') as f:
                # get out the tokenized/labeled lines
                lines = f.readlines()
                # for each line, add its character embeddings to the data array.
                # if the character doesn't exist yet, add it to the vocab dict.
                for line in lines:
                    chars, label = line.split('\t', 1)
                    label = label.split('\n', 1)[0]
                    # first, deal with the label if it is not 0.
                    if is_label(label):
                        if label not in entity_vocab:
                            entity_vocab[label] = entity_idx
                            entity_idx += 1
                    else:
                        label = NOT_LABEL
                    # need to append a space after the last character
                    chars += ' '
                    for char in chars:
                        # add unseen char to vocab
                        if char not in vocab:
                            vocab[char] = vocab_idx
                            vocab_idx += 1
                        # add the character and label!
                        data.append(vocab[char])
                        labels.append(entity_vocab[label])
        except Exception:
            print("Error reading file %s!" % filename)
            raise
    return (data, vocab), (labels, entity_vocab)
