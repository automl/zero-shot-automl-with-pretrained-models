"""An skeleton of the code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

Your changes should be , at minimum, to the '__init__', 'train', and 'test' in the 'Model' class, which will determine:
- How your method/model are initialized given the task metadata
- How your method/model will utilize the provided training data, validation data, and remaining time budget
Feel free to add new variables/functions to augment your method, and refer to the provided baselines as simple examples that implement rudimentary early-stopping or model customization to task type.

To create a valid submission, zip model.py, metadata, and the optional 'tasks_to_run.yaml' together with any other necessary files such as Python modules/packages, pre-trained weights, etc. The final zip file should not exceed 300MB.
- The 'metadata' file is necessary for CodaLab ingestion and should not be changed.
- 'task_to_run.yaml' will specify a subset of the tasks to run on. If not included in the submission, all tasks will be run. See the included example.
"""


# Feel free to install/import other packages/modules
import datetime
import logging
import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# seeding randomness for reproducibility if needed
# np.random.seed(42)
# torch.manual_seed(1)

class Model:
    def __init__(self, metadata):
        '''
        CHANGE ME
        The initalization procedure for your method given the metadata of the task
        '''
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        
        '''
        Attribute necessary for ingestion program, as Model.train() is called in a loop.        
        Can be set to True if your method is ready to stop training early.
        '''        
        self.done_training = False
        
        # Store metadata
        self.metadata_ = metadata

        '''
        Getting the task's input/output dimensions
        '''       
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

        self.input_shape = (channel, sequence_size, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)


        '''
        Storing the task type and the final metric used in evaluation
        You can use these to adjust how the model is created and how the training/objective are determined, for example
        '''
        self.task_type = self.metadata_.get_task_type()
        self.final_metric = self.metadata_.get_final_metric()

        # Initialize a model!
        
        
        #####
        
        
        '''
        Attributes for managing time budget ; optional, may be helpful
        '''
        self.birthday = time.time()
        self.total_train_time = 0
        self.cumulated_num_steps = 0
        self.estimated_time_per_step = None
        self.total_test_time = 0
        self.cumulated_num_tests = 0
        self.estimated_time_test = None
        self.trained = False



    def get_dataloader(self, dataset, batch_size, split):
        """Get the PyTorch dataloader. Do not modify this method, or your submission may break on certain tasks.
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

    def train(self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None):
        '''
        CHANGE ME
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        '''
        
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

         # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )

        train_start = time.time()

        # However your method is trained goes here!
        
        #####
        
        train_end = time.time()

        # Update for time budget managing
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
          predictions: A `numpy.ndarray` matrix containing your method's predictions
        """

        test_begin = time.time()
    
        # Create test dataloader
        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )

        # Use your method to performence inference on the test data here!
        predictions = None

        #####
        
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

'''
Use this logger as necessary
'''
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
