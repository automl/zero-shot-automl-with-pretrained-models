"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging

import numpy as np
import tensorflow as tf
import yaml

# fmt: off
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath("."))  # This is needed for the run_local_test ingestion

here = os.path.dirname(os.path.abspath(__file__))
model_dirs = [
    '',  # current directory
    'winner_cv',  # AutoCV/AutoCV2 winner model
    'winner_nlp',  # AutoNLP 2nd place winner
    'winner_speech',  # AutoSpeech winner
    'winner_tabular'  # simple NN model
]
for model_dir in model_dirs:
    sys.path.append(os.path.join(here, model_dir))

from winner_cv.model import Model as AutoCVModel  # isort:skip
from winner_nlp.model import Model as AutoNLPModel  # isort:skip
from winner_speech.model import Model as AutoSpeechModel  # isort:skip
from winner_tabular.model import Model as TabularModel  # isort:skip

# fmt: on

DOMAIN_TO_MODEL = {
    'image': AutoCVModel,
    'video': AutoCVModel,
    'text': AutoNLPModel,
    'speech': AutoSpeechModel,
    'tabular': TabularModel
}


class Model():
    """A model that combine all winner solutions. Using domain inferring and
  apply winner solution in the corresponding domain."""

    def __init__(self, metadata, model_config=None, model_config_name=None):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        if model_config is None:
            model_config_name = model_config_name or "default.yaml"
            with open(os.path.join(here, "configs", model_config_name)) as stream:
                model_config = yaml.safe_load(stream)

        self.done_training = False
        self.metadata = metadata
        self.domain = infer_domain(metadata)
        logger.info("The inferred domain of current dataset is: {}."\
                    .format(self.domain))
        self.domain_metadata = get_domain_metadata(metadata, self.domain)
        print(self.domain)
        DomainModel = DOMAIN_TO_MODEL[self.domain]
        self.domain_model = DomainModel(self.domain_metadata, model_config)

    def train(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train
        self.set_domain_dataset(dataset, is_training=True)

        # Train the model
        self.domain_model.train(
            self.domain_dataset_train, remaining_time_budget=remaining_time_budget
        )

        # Update self.done_training
        self.done_training = self.domain_model.done_training

    def test(self, dataset, remaining_time_budget=None):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and
        # store as self.domain_dataset_test
        self.set_domain_dataset(dataset, is_training=False)

        # As the original metadata doesn't contain number of test examples, we
        # need to add this information
        if self.domain in ['text', 'speech'] and\
           (not self.domain_metadata['test_num'] >= 0):
            self.domain_metadata['test_num'] = len(self.X_test)

        # Make predictions
        Y_pred = self.domain_model.test(
            self.domain_dataset_test, remaining_time_budget=remaining_time_budget
        )

        # Update self.done_training
        self.done_training = self.domain_model.done_training

        return Y_pred

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

    def to_numpy(self, dataset, is_training):
        """Given the TF dataset received by `train` or `test` method, compute two
        lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
        `Y_test` for `test`. Although `Y_test` will always be an
        all-zero matrix, since the test labels are not revealed in `dataset`.
        The computed two lists will by memorized as object attribute:
          self.X_train
          self.Y_train
        or
          self.X_test
          self.Y_test
        according to `is_training`.
        WARNING: since this method will load all data in memory, it's possible to
          cause Out Of Memory (OOM) error, especially for large datasets (e.g.
          video/image datasets).
        Args:
          dataset: a `tf.data.Dataset` object, received by the method `self.train`
            or `self.test`.
          is_training: boolean, indicates whether it concerns the training set.
        Returns:
          two lists of NumPy arrays, for features and labels respectively. If the
            examples all have the same shape, they can be further converted to
            NumPy arrays by:
              X = np.array(X)
              Y = np.array(Y)
            And in this case, `X` will be of shape
              [num_examples, sequence_size, row_count, col_count, num_channels]
            and `Y` will be of shape
              [num_examples, num_classes]
        """
        if is_training:
            subset = 'train'
        else:
            subset = 'test'
        attr_X = 'X_{}'.format(subset)
        attr_Y = 'Y_{}'.format(subset)

        # Only iterate the TF dataset when it's not done yet
        if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            X = []
            Y = []
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        example, labels = sess.run(next_element)
                        X.append(example)
                        Y.append(labels)
                    except tf.errors.OutOfRangeError:
                        break
            setattr(self, attr_X, X)
            setattr(self, attr_Y, Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        return X, Y

    def set_domain_dataset(self, dataset, is_training=True):
        """Recover the dataset in corresponding competition format (esp. AutoNLP
        and AutoSpeech) and set corresponding attributes:
          self.domain_dataset_train
          self.domain_dataset_test
        according to `is_training`.
        """
        if is_training:
            subset = 'train'
        else:
            subset = 'test'
        attr_dataset = 'domain_dataset_{}'.format(subset)

        if not hasattr(self, attr_dataset):
            logger.info(
                "Begin recovering dataset format in the original " +
                "competition for the subset: {}...".format(subset)
            )
            if self.domain == 'text':
                # Get X, Y as lists of NumPy array
                X, Y = self.to_numpy(dataset, is_training=is_training)

                # Retrieve vocabulary (token to index map) from metadata and construct
                # the inverse map
                vocabulary = self.metadata.get_channel_to_index_map()
                index_to_token = [None] * len(vocabulary)
                for token in vocabulary:
                    index = vocabulary[token]
                    index_to_token[index] = token

                # Get separator depending on whether the dataset is in Chinese
                if is_chinese(self.metadata):
                    sep = ''
                else:
                    sep = ' '

                # Construct the corpus
                corpus = []
                for x in X:  # each x in X is a list of indices (but as float)
                    tokens = [index_to_token[int(i)] for i in x]
                    document = sep.join(tokens)
                    corpus.append(document)

                # Construct the dataset for training or test
                if is_training:
                    labels = np.array(Y)
                    domain_dataset = corpus, labels
                else:
                    domain_dataset = corpus

                # Set the attribute
                setattr(self, attr_dataset, domain_dataset)

            elif self.domain == 'speech':
                # Get X, Y as lists of NumPy array
                X, Y = self.to_numpy(dataset, is_training=is_training)

                # Convert each array to 1-D array
                X = [np.squeeze(x) for x in X]

                # Construct the dataset for training or test
                if is_training:
                    labels = np.array(Y)
                    domain_dataset = X, labels
                else:
                    domain_dataset = X

                # Set the attribute
                setattr(self, attr_dataset, domain_dataset)

            elif self.domain in ['image', 'video', 'tabular']:
                setattr(self, attr_dataset, dataset)
            else:
                raise ValueError("The domain {} doesn't exist.".format(self.domain))


def infer_domain(metadata):
    """Infer the domain from the shape of the 4-D tensor.

      Args:
        metadata: an AutoDLMetadata object.
      """
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    channel_to_index_map = metadata.get_channel_to_index_map()
    domain = None
    if sequence_size == 1:
        if row_count == 1 or col_count == 1:
            domain = "tabular"
        else:
            domain = "image"
    else:
        if row_count == 1 and col_count == 1:
            if len(channel_to_index_map) > 0:
                domain = "text"
            else:
                domain = "speech"
        else:
            domain = "video"
    return domain


def is_chinese(metadata):
    """Judge if the dataset is a Chinese NLP dataset. The current criterion is if
    each word in the vocabulary contains one single character, because when the
    documents are in Chinese, we tokenize each character when formatting the
    dataset.

    Args:
    metadata: an AutoDLMetadata object.
    """
    domain = infer_domain(metadata)
    if domain != 'text':
        return False
    for i, token in enumerate(metadata.get_channel_to_index_map()):
        if len(token) != 1:
            return False
        if i >= 100:
            break
    return True


def get_domain_metadata(metadata, domain, is_training=True):
    """Recover the metadata in corresponding competitions, esp. AutoNLP
    and AutoSpeech.

    Args:
    metadata: an AutoDLMetadata object.
    domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
    """
    if domain == 'text':
        # Fetch metadata info from `metadata`
        class_num = metadata.get_output_size()
        num_examples = metadata.size()
        language = 'ZH' if is_chinese(metadata) else 'EN'
        time_budget = 1200  # WARNING: Hard-coded

        # Create domain metadata
        domain_metadata = {}
        domain_metadata['class_num'] = class_num
        if is_training:
            domain_metadata['train_num'] = num_examples
            domain_metadata['test_num'] = -1
        else:
            domain_metadata['train_num'] = -1
            domain_metadata['test_num'] = num_examples
        domain_metadata['language'] = language
        domain_metadata['time_budget'] = time_budget

        return domain_metadata
    elif domain == 'speech':
        # Fetch metadata info from `metadata`
        class_num = metadata.get_output_size()
        num_examples = metadata.size()

        # WARNING: hard-coded properties
        file_format = 'wav'
        sample_rate = 16000

        # Create domain metadata
        domain_metadata = {}
        domain_metadata['class_num'] = class_num
        if is_training:
            domain_metadata['train_num'] = num_examples
            domain_metadata['test_num'] = -1
        else:
            domain_metadata['train_num'] = -1
            domain_metadata['test_num'] = num_examples
        domain_metadata['file_format'] = file_format
        domain_metadata['sample_rate'] = sample_rate

        return domain_metadata
    else:
        return metadata


def get_logger(verbosity_level):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger('INFO')
