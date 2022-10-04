"""An example of code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py and metadata together with other necessary files
such as tasks_to_run.yaml, Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

import datetime
import logging
import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from wrn1d import WideResNet1d
from wrn2d import WideResNet2d
from wrn3d import WideResNet3d

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)

# PyTorch Model class
class TorchModel(nn.Module):
    """
    Defines a module that will be created in '__init__' of the 'Model' class below, and will be used for training and predictions.
    """

    def __init__(self, input_shape, output_dim):
        """a simple linear model"""
        super(TorchModel, self).__init__()

        fc_size = np.prod(input_shape)
        print("input_shape, fc_size", input_shape, fc_size)
        self.fc = nn.Linear(fc_size, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model:
    def __init__(self, metadata):
        """
        The initalization procedure for your method given the metadata of the task
        """
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata

        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = math.prod(self.metadata_.get_output_shape())

        self.num_examples_train = self.metadata_.size()

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()
        self.num_test = self.metadata_.get_output_shape()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )

        # TODO
        self.input_shape = (sequence_size, channel, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)

        # getting an object for the PyTorch Model class for Model Class
        # use CUDA if available
        # TODO
        depth = 16  # TODO increase to 40
        spacetime_dims = np.count_nonzero(np.array(self.input_shape)[[0, 2, 3]] != 1)
        logger.info(f"Using WRN of dimension {spacetime_dims}")
        if spacetime_dims == 1:
            self.model = WideResNet1d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=0.0,
                in_channels=channel,
            )
        elif spacetime_dims == 2:
            self.model = WideResNet2d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=0.0,
                in_channels=channel,
            )
        elif spacetime_dims == 3:
            self.model = WideResNet3d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=0.0,
                in_channels=channel,
            )
        elif spacetime_dims == 0:  # Special case where we have channels only
            self.model = WideResNet1d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=0.0,
                in_channels=1,
            )
        else:
            raise NotImplementedError

        print("\nPyModel Defined\n")
        print(self.model)
        self.model.to(self.device)

        # PyTorch Optimizer and Criterion
        if self.metadata_.get_task_type() == "continuous":
            self.criterion = nn.MSELoss()
        elif self.metadata_.get_task_type() == "single-label":
            self.criterion = nn.CrossEntropyLoss()
        elif self.metadata_.get_task_type() == "multi-label":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.cumulated_num_steps = 0
        self.estimated_time_per_step = None
        self.total_test_time = 0
        self.cumulated_num_tests = 0
        self.estimated_time_test = None
        self.trained = False

        # PYTORCH
        # Critical number for early stopping
        self.num_epochs_we_want_to_train = 100

        # no of examples at each step/batch
        self.train_batch_size = 128
        self.test_batch_size = 128

    def get_dataloader(self, dataset, batch_size, split):
        """Get the PyTorch dataloader. Do not modify this method.
        Args:
          dataset:
          batch_size : batch_size for training set

        Return:
          dataloader: PyTorch Dataloader
        """
        if split == "train":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=dataset.collate_fn,
            )
        elif split == "test":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        return dataloader

    def train(
        self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None
    ):
        """
        CHANGE ME
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        """

        """Train this algorithm on the Pytorch dataset.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

        ****************************************************************************
        ****************************************************************************
        IMPORTANT: the loop of calling `train` and `test` will only run if
            self.done_training = False
          (the corresponding code can be found in ingestion.py, search
          'M.done_training')
          Otherwise, the loop will go on until the time budget is used up. Please
          pay attention to set self.done_training = True when you think the model is
          converged or when there is not enough time for next round of training.
        ****************************************************************************
        ****************************************************************************

        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor

          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a pre-split validation set is provided, in which case you should use it for any validation purposes. Otherwise, you are free to create your own validation split(s) as desired.
          
          val_metadata: a 'DecathlonMetadata' object, corresponding to 'val_dataset'.

          remaining_time_budget: time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
        """

        steps_to_train = self.get_steps_to_train(remaining_time_budget)
        if steps_to_train <= 0:
            logger.info(
                "Not enough time remaining for training. "
                + "Estimated time for training per step: {:.2f}, ".format(
                    self.estimated_time_per_step
                )
                + "but remaining time budget is: {:.2f}. ".format(remaining_time_budget)
                + "Skipping..."
            )
            self.done_training = True
        else:
            msg_est = ""
            if self.estimated_time_per_step:
                msg_est = "estimated time for this: " + "{:.2f} sec.".format(
                    steps_to_train * self.estimated_time_per_step
                )
            logger.info(
                "Begin training for another {} steps...{}".format(
                    steps_to_train, msg_est
                )
            )

            # If PyTorch dataloader for training set doen't already exists, get the train dataloader
            if not hasattr(self, "trainloader"):
                self.trainloader = self.get_dataloader(
                    dataset,
                    self.train_batch_size,
                    "train",
                )

            train_start = time.time()

            # Training loop
            # TODO remove
            steps_to_train = len(self.trainloader)
            logger.info(f"steps_to_train {steps_to_train}")
            # TODO remove
            self.trainloop(self.criterion, self.optimizer, steps=steps_to_train)
            train_end = time.time()

            # Update for time budget managing
            # TODO
            train_duration = train_end - train_start
            self.total_train_time += train_duration
            self.cumulated_num_steps += steps_to_train
            self.estimated_time_per_step = (
                self.total_train_time / self.cumulated_num_steps
            )
            logger.info(
                "{} steps trained. {:.2f} sec used. ".format(
                    steps_to_train, train_duration
                )
                + "Now total steps trained: {}. ".format(self.cumulated_num_steps)
                + "Total time used for training: {:.2f} sec. ".format(
                    self.total_train_time
                )
                + "Current estimated time per step: {:.2e} sec.".format(
                    self.estimated_time_per_step
                )
            )

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.

        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """
        if self.done_training:
            return None

        if self.choose_to_stop_early():
            logger.info("Oops! Choose to stop early for next call!")
            self.done_training = True
        test_begin = time.time()
        if (
            remaining_time_budget
            and self.estimated_time_test
            and self.estimated_time_test > remaining_time_budget
        ):
            logger.info(
                "Not enough time for test. "
                + "Estimated time for test: {:.2e}, ".format(self.estimated_time_test)
                + "But remaining time budget is: {:.2f}. ".format(remaining_time_budget)
                + "Stop train/predict process by returning None."
            )
            return None

        msg_est = ""
        if self.estimated_time_test:
            msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
        logger.info("Begin testing..." + msg_est)

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )

        # get predictions from the test loop
        predictions = self.testloop(self.testloader)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration
        self.cumulated_num_tests += 1
        self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
            + "Current estimated time for test: {:.2e} sec.".format(
                self.estimated_time_test
            )
        )
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

    def trainloop(self, criterion, optimizer, steps):
        """Training loop with no of given steps
        Args:
          criterion: PyTorch Loss function
          Optimizer: PyTorch optimizer for training
          steps: No of steps to train the model

        Return:
          None, updates the model parameters
        """
        self.model.train()
        data_iterator = iter(self.trainloader)
        for _ in tqdm(range(steps)):
            try:
                images, labels = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.trainloader)
                images, labels = next(data_iterator)

            images = images.float().to(self.device)
            # labels = nn.functional.one_hot(labels, num_classes=18)
            labels = labels.float().to(self.device)
            optimizer.zero_grad()

            logits = self.model(images)
            # print('logits.shape', logits.shape)
            # print('labels', labels.shape)
            # FIXME make sure that things are correctly reshaped...

            loss = criterion(logits, labels.reshape(labels.shape[0], -1))
            if hasattr(self, "scheduler"):
                self.scheduler.step(loss)
            loss.backward()
            optimizer.step()

    def get_steps_to_train(self, remaining_time_budget):
        """Get number of steps for training according to `remaining_time_budget`.

        The strategy is:
          1. If no training is done before, train for 10 steps (ten batches);
          2. Otherwise, estimate training time per step and time needed for test,
             then compare to remaining time budget to compute a potential maximum
             number of steps (max_steps) that can be trained within time budget;
          3. Choose a number (steps_to_train) between 0 and max_steps and train for
             this many steps. Double it each time.
        """
        if not remaining_time_budget:  # This is never true in the competition anyway
            remaining_time_budget = 1200  # if no time limit is given, set to 20min

        if not self.estimated_time_per_step:
            steps_to_train = 100
        else:
            if self.estimated_time_test:
                tentative_estimated_time_test = self.estimated_time_test
            else:
                tentative_estimated_time_test = 50  # conservative estimation for test
            max_steps = int(
                (remaining_time_budget - tentative_estimated_time_test)
                / self.estimated_time_per_step
            )
            max_steps = max(max_steps, 1)
            if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
                steps_to_train = int(
                    2**self.cumulated_num_tests
                )  # Double steps_to_train after each test
            else:
                steps_to_train = 0
        return steps_to_train

    def testloop(self, dataloader):
        """
        Args:
          dataloader: PyTorch test dataloader

        Return:
          preds: Predictions of the model as Numpy Array.
        """
        preds = []
        with torch.no_grad():
            self.model.eval()
            for images, _ in iter(dataloader):
                if torch.cuda.is_available():
                    images = images.float().cuda()
                else:
                    images = images.float()
                logits = self.model(images)

                # Choose correct prediction type
                if self.metadata_.get_task_type() == "continuous":
                    pred = logits
                elif self.metadata_.get_task_type() == "single-label":
                    pred = torch.softmax(logits, dim=1).data
                elif self.metadata_.get_task_type() == "multi-label":
                    pred = torch.sigmoid(logits).data
                else:
                    raise NotImplementedError

                preds.append(pred.cpu().numpy())

        preds = np.vstack(preds)
        return preds

    def choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
        process).
        """
        # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
        # return np.random.rand() < self.early_stop_proba
        batch_size = self.train_batch_size
        num_examples = self.metadata_.size()
        num_epochs = self.cumulated_num_steps * batch_size / num_examples
        logger.info("Model already trained for {} epochs.".format(num_epochs))
        return (
            num_epochs > self.num_epochs_we_want_to_train
        )  # Train for at least certain number of epochs then stop


def get_logger(verbosity_level):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
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


logger = get_logger("INFO")
