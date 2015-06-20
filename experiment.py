"""
Character-level recurrent model for named entity recognition.
"""
from __future__ import print_function, division
# normal imports
import os
import math
import numpy
import theano.tensor as T
# opendeep imports
import opendeep
from opendeep.data import Dataset, TRAIN, VALID, TEST
from opendeep.models import LSTM
from opendeep.monitor import Monitor, Plot
from opendeep.optimization import RMSProp
from opendeep.utils.misc import numpy_one_hot


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

class CharsNER(Dataset):
    """
    Need to process the .txt files with tagged tokens into character arrays as the dataset.
    """
    def __init__(self, sequence_length=100, train_split=0.85, valid_split=0.1):
        # for splitting the data into training, validation, and test sets.
        assert (0. < train_split <= 1.), "Train_split needs to be a fraction between (0, 1]."
        assert (0. <= valid_split < 1.), "Valid_split needs to be a fraction between [0, 1)."
        assert train_split + valid_split <= 1., "Train_split + valid_split can't be greater than 1."
        # make test_split the leftover percentage!
        test_split = 1 - (train_split + valid_split)

        # try to read the txt files and build the character array & vocabulary
        print("Processing files (this loads everything into memory)...")
        data = []
        labels = []
        NOT_LABEL = "N/A"
        self.entity_vocab = {NOT_LABEL: 0}
        entity_idx = 1
        self.vocab = {}
        vocab_idx = 0
        # go through each txt file
        for filename in find_txt_files():
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
                            if label not in self.entity_vocab:
                                self.entity_vocab[label] = entity_idx
                                entity_idx += 1
                        else:
                            label = NOT_LABEL
                        # need to append a space after the last character
                        chars += ' '
                        for char in chars:
                            # add unseen char to vocab
                            if char not in self.vocab:
                                self.vocab[char] = vocab_idx
                                vocab_idx += 1
                            # add the character and label!
                            data.append(self.vocab[char])
                            labels.append(self.entity_vocab[label])
            except Exception:
                print("Error reading file %s!" % filename)
                raise

        print("Converting characters and labels to one-hot vectors...")
        data = numpy_one_hot(numpy.asarray(data), n_classes=numpy.amax(self.vocab.values()) + 1)
        labels = numpy_one_hot(numpy.asarray(labels), n_classes=numpy.amax(self.entity_vocab.values()) + 1)
        print("Conversion done!")
        print("Vocab: %s" % str(sorted(self.vocab.keys())))
        print("Labels: %s" % str(sorted(self.entity_vocab.keys())))
        print("Dataset size:\t%s" % str(data.shape))
        print("Label size:\t%s" % str(labels.shape))

        print("Dividing into sequences of length %d..." % sequence_length)
        # first make sure to chop off the remainder of the data so sequence_length can divide evenly into data.
        length, self.vocab_size = data.shape
        _, self.label_size = labels.shape
        if length % sequence_length != 0:
            print("Chopping off %d characters!" % (length % sequence_length))
            data = data[:sequence_length * math.floor(length / sequence_length)]
            labels = labels[:sequence_length * math.floor(length / sequence_length)]
        # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, vocab_size)
        data = numpy.reshape(data, (length / sequence_length, sequence_length, self.vocab_size))
        labels = numpy.reshape(labels, (length / sequence_length, sequence_length, self.label_size))
        print("Dataset size:\t%s" % str(data.shape))
        print("Label size:\t%s" % str(labels.shape))

        # shuffle
        length = data.shape[0]
        shuffle_order = numpy.arange(length)
        numpy.random.shuffle(shuffle_order)
        data = data[shuffle_order]
        labels = labels[shuffle_order]

        self._train_len = int(math.floor(length * train_split))
        self._valid_len = int(math.floor(length * valid_split))
        self._test_len = int(max(length - self._valid_len - self._train_len, 0))
        self._seq_len = sequence_length

        # divide into train, valid, and test sets!
        self.train_X = opendeep.dataset_shared(data[:self._train_len], name='chars_train_x', borrow=True)
        self.train_Y = opendeep.dataset_shared(labels[:self._train_len], name='chars_train_y', borrow=True)

        if valid_split > 0:
            self.valid_X = opendeep.dataset_shared(data[self._train_len:self._train_len + self._valid_len],
                                                   name='chars_valid_x', borrow=True)
            self.valid_Y = opendeep.dataset_shared(labels[self._train_len:self._train_len + self._valid_len],
                                                   name='chars_valid_y', borrow=True)
        else:
            self.valid_X = None
            self.valid_Y = None

        if test_split > 0:
            self.test_X = opendeep.dataset_shared(data[self._train_len + self._valid_len:],
                                                  name='chars_test_x', borrow=True)
            self.test_Y = opendeep.dataset_shared(labels[self._train_len + self._valid_len:],
                                                  name='chars_test_y', borrow=True)
        else:
            self.test_X = None
            self.test_Y = None

        print("Dataset initialized!")

    def getSubset(self, subset):
        if subset is TRAIN:
            return self.train_X, self.train_Y
        elif subset is VALID:
            return self.valid_X, self.valid_Y
        elif subset is TEST:
            return self.test_X, self.test_Y
        else:
            return None, None

    def getDataShape(self, subset):
        if subset is TRAIN:
            return self._train_len, self._seq_len, self.vocab_size
        elif subset is VALID:
            return self._valid_len, self._seq_len, self.vocab_size
        elif subset is TEST:
            return self._test_len, self._seq_len, self.vocab_size
        else:
            return None


def main():
    dataset = CharsNER()

    # define our model! we are using lstm.
    hidden_size = 128

    lstm = LSTM(input_size=dataset.vocab_size,
                hidden_size=hidden_size,
                output_size=dataset.label_size,
                hidden_activation='tanh',
                inner_hidden_activation='sigmoid',
                activation='softmax',
                weights_init='uniform',
                weights_interval='montreal',
                r_weights_init='orthogonal',
                clip_recurrent_grads=5.,
                noise=False,
                noise_level=0.2,
                direction='bidirectional',
                cost_function='nll')

    # output is in shape (n_timesteps, n_sequences, data_dim)

    # calculate the mean prediction error over timesteps and batches
    predictions = T.argmax(lstm.get_outputs(), axis=2)
    actual = T.argmax(lstm.get_targets(), axis=2)
    error = T.mean(T.neq(predictions, actual))

    # optimizer
    optimizer = RMSProp(dataset=dataset,
                        model=lstm,
                        n_epoch=500,
                        batch_size=25,
                        save_frequency=10,
                        learning_rate=2e-3,
                        lr_decay="exponential",
                        lr_factor=0.95,
                        grad_clip=None,
                        hard_clip=False
                        )

    # monitors
    errors = Monitor(name='error', expression=error, train=True, valid=True, test=True)

    # plot the monitor
    plot = Plot('Chars', monitor_channels=[errors], open_browser=True)

    optimizer.train(plot=plot)


if __name__ == "__main__":
    # if you want debugging output from opendeep
    opendeep.config_root_logger()
    # run the experiment!
    main()
