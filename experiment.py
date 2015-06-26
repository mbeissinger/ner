"""
Character-level recurrent model for named entity recognition.
"""
from __future__ import print_function, division
# normal imports
import math
import numpy
import theano.tensor as T
# opendeep imports
import opendeep
from opendeep.data import MemoryDataset, TEST
from opendeep.models import LSTM
from opendeep.monitor import Monitor, Plot
from opendeep.optimization import RMSProp
from opendeep.utils.misc import numpy_one_hot
# internal imports
from data_processing import parse_text_files

def getCharsDataset(sequence_length=100, train_split=0.85, valid_split=0.1, data_dir='data/'):
    # try to read the txt files and build the character array & vocabulary
    print("Processing files (this loads everything into memory)...")
    (data, vocab), (labels, entity_vocab) = parse_text_files(data_dir)

    # l = numpy.asarray(labels)
    # percent_entity = numpy.sum(l>0) / l.shape[0]
    # print("Overall percent of characters that are entities: %s" % str(percent_entity))

    print("Converting characters and labels to one-hot vectors...")
    # data = data[:50000]
    # labels = labels[:50000]
    data = numpy_one_hot(numpy.asarray(data), n_classes=numpy.amax(vocab.values()) + 1)
    labels = numpy_one_hot(numpy.asarray(labels), n_classes=numpy.amax(entity_vocab.values()) + 1)
    print("Conversion done!")
    print("Vocab: %s" % str(sorted(vocab.keys())))
    print("Labels: %s" % str(sorted(entity_vocab.keys())))
    print("Dataset size:\t%s" % str(data.shape))
    print("Label size:\t%s" % str(labels.shape))

    print("Dividing into sequences of length %d..." % sequence_length)
    # first make sure to chop off the remainder of the data so sequence_length can divide evenly into data.
    length, vocab_size = data.shape
    _, label_size = labels.shape
    if length % sequence_length != 0:
        print("Chopping off %d characters!" % (length % sequence_length))
        data = data[:sequence_length * math.floor(length / sequence_length)]
        labels = labels[:sequence_length * math.floor(length / sequence_length)]
    # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, vocab_size)
    data = numpy.reshape(data, (length / sequence_length, sequence_length, vocab_size))
    labels = numpy.reshape(labels, (length / sequence_length, sequence_length, label_size))
    print("Dataset size:\t%s" % str(data.shape))
    print("Label size:\t%s" % str(labels.shape))

    # shuffle
    length = data.shape[0]
    shuffle_order = numpy.arange(length)
    numpy.random.shuffle(shuffle_order)
    data = data[shuffle_order]
    labels = labels[shuffle_order]

    # split!
    assert (0. < train_split <= 1.), "Train_split needs to be a fraction between (0, 1]."
    assert (0. <= valid_split < 1.), "Valid_split needs to be a fraction between [0, 1)."
    assert train_split + valid_split <= 1., "Train_split + valid_split can't be greater than 1."
    # make test_split the leftover percentage!
    test_split = 1 - (train_split + valid_split)
    length = data.shape[0]
    _train_len = int(math.floor(length * train_split))
    _valid_len = int(math.floor(length * valid_split))
    _test_len = int(math.floor(length * test_split))

    # do the splits!
    if _valid_len > 0:
        valid_X = data[_train_len:_train_len + _valid_len]
        valid_Y = labels[_train_len:_train_len + _valid_len]
    else:
        valid_X, valid_Y = None, None

    if _test_len > 0:
        test_X = data[_train_len + _valid_len:]
        test_Y = labels[_train_len + _valid_len:]
    else:
        test_X, test_Y = None, None

    train_X = data[:_train_len]
    train_Y = labels[:_train_len]

    train_num = numpy.sum(numpy.argmax(train_Y, axis=2) > 0)
    train_denom = numpy.prod(train_Y.shape[:2])
    train_percent_entity = train_num / train_denom
    print("Percent of training characters that are entities: %s" % str(train_percent_entity))

    valid_num = numpy.sum(numpy.argmax(valid_Y, axis=2) > 0)
    valid_denom = numpy.prod(valid_Y.shape[:2])
    valid_percent_entity = valid_num / valid_denom
    print("Percent of validation characters that are entities: %s" % str(valid_percent_entity))

    test_num = numpy.sum(numpy.argmax(test_Y, axis=2) > 0)
    test_denom = numpy.prod(test_Y.shape[:2])
    test_percent_entity = test_num / test_denom
    print("Percent of testing characters that are entities: %s" % str(test_percent_entity))

    test_percent_entity = (train_num+valid_num+test_num) / (train_denom+valid_denom+test_denom)
    print("Overall percent of characters that are entities: %s" % str(test_percent_entity))

    dataset = MemoryDataset(train_X=train_X, train_Y=train_Y,
                            valid_X=valid_X, valid_Y=valid_Y,
                            test_X=test_X, test_Y=test_Y)

    return dataset, vocab, entity_vocab

def train(lstm, dataset):
    # let's monitor the error %
    # output is in shape (n_timesteps, n_sequences, data_dim)
    # calculate the mean prediction error over timesteps and batches
    predictions = T.argmax(lstm.get_outputs(), axis=2)
    actual = T.argmax(lstm.get_targets()[0].dimshuffle(1, 0, 2), axis=2)
    error = T.mean(T.neq(predictions, actual))

    # optimizer - RMSProp generally good for recurrent nets, lr taken from Karpathy's char-rnn project.
    optimizer = RMSProp(
        dataset=dataset,
        n_epoch=500,
        batch_size=300,
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

    lstm.train(optimizer=optimizer, plot=plot)

def get_entities(predictions, vocab, entity_vocab):
    # find continuous entity characters across timesteps
    entities = []
    previous_labels = [0 for _ in range(predictions.shape[1])]
    continuous_strings = ["" for _ in range(len(previous_labels))]
    for i, batch in enumerate(predictions):
        for j, label in enumerate(batch):
            # if not continuous, reset
            if label != previous_labels[j]:
                entity = continuous_strings[j]
                # add only if the label is an entity
                if previous_labels[j] != 0:
                    label_string = entity_vocab.get(previous_labels[j])
                    entities.append((entity, label_string))
                continuous_strings[j] = ""
            data_char = vocab.get(numpy.argmax(data[i, j]))
            continuous_strings[j] += data_char
            previous_labels[j] = label

    return entities




def main():
    dataset, vocab, entity_vocab = getCharsDataset(sequence_length=100)

    # define our model! we are using lstm.
    hidden_size = 64

    lstm = LSTM(input_size=len(vocab),
                hidden_size=hidden_size,
                output_size=len(entity_vocab),
                hidden_activation='tanh',
                inner_hidden_activation='sigmoid',
                activation='softmax',
                weights_init='uniform',
                weights_interval='montreal',
                r_weights_init='orthogonal',
                clip_recurrent_grads=5.,
                noise='dropout',
                noise_level=0.2,
                direction='bidirectional',
                cost_function='nll',
                cost_args={'one_hot': True})

    # train the lstm on our dataset!
    train(lstm, dataset)

    # test on some data (a few articles in size)!
    test_dataset, _, _ = getCharsDataset(sequence_length=50000)
    test_data, test_labels = dataset.get_subset(TEST, batch_size=1)

    data = test_data.next()
    # data has to be vectorized characters (or you could use the vocab dictionary to create matrices from strings)
    character_probs = lstm.run(data)
    # this has the shape (timesteps, batches, data), so swap axes to (batches, timesteps, data)
    character_probs = numpy.swapaxes(character_probs, 0, 1)
    # now extract the guessed entities
    predictions = numpy.argmax(character_probs, axis=2)



if __name__ == "__main__":
    # if you want debugging output from opendeep
    opendeep.config_root_logger()
    # run the experiment!
    main()
