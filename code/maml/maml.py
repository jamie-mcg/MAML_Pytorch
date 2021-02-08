import numpy as np
import torch
import torch.nn as nn

from itertools import repeat

CRITERION = {
    "mse": nn.MSELoss
}

OPTIMIZER = {
    "adam": torch.optim.Adam
}

class MAML(nn.Module):
    """
    This is a MAML (Model Agnostic Meta-Learning) object. The MAML algorithm reproduced here is
    taken from [1]. The algorithm uses an inner loop, designed to adapt to individual tasks from 
    an optimum initial parameter set. These initial parameters are learnt using an outer-loop 
    which optimizes depending on the inner-loop results.

    The algorithm is "model agnostic" meaning that any model suited to any Machine Learning (ML) 
    problem can be loading into this Meta-Learning "wrapper" and the MAML algorithm can be applied.

    Methods:
    - inner_loop():
        This method defines the inner loop of the MAML algorithm. Inside this method, we use a set
        of training tasks and measure the models adaptability to these.

    - train():
        This method defines the outer loop of the MAML algorithm. In this loop we make the 
        adaptations to the base learner model.

    [1] - C. Finn, et. al, "Model Agnostic Meta-Learning", (2017).
    """
    def __init__(self, model, alpha, beta, inner_steps, metatrain_dataloader, metatest_dataloader, 
                    inner_criterion="mse", optimizer="adam", batch_size=2, print_every=100):
        super(MAML, self).__init__()

        # Store the base learner model.
        self._model = model

        # Check to see if GPU is available
        self._train_on_gpu = True if torch.cuda.is_available() else False

        # Define the inner (alpha) and outer (beta) learning rates.
        self._alpha = alpha
        self._beta = beta

        # Define the number of inner steps to take.
        self._inner_steps = inner_steps

        # Define the batch size of tasks.
        self._batch_size = batch_size

        # Store the training and testing dataloaders.
        self._metatrain_dataloader = metatrain_dataloader
        self._metatest_dataloader = metatest_dataloader

        # Define the loss criterion to be used in the inner loop.
        self._inner_criterion = CRITERION[inner_criterion.lower()]()

        # Define the PyTorch optimizer to be used in the outer loop.
        self._optimizer = OPTIMIZER[optimizer.lower()](model.parameters(), beta)

        # Define necessary results parameters.
        self._print_every = print_every
        self.training_losses = []
        self.validation_losses = []
        self.iterations = []

    def inner_loop(self, X_train, y_train, X_test, y_test, valid=False):
        """
        This method defines the inner loop of the MAML algorithm. Given some meta-training 
        and meta-testing data for a single task, we take a number of inner steps towards the
        optimum parameters for this task.

        Inputs:
        - X_train, y_train: The training data for a single task in the form of a tensor.
        - X_test, y_test: The test data for a single task in the form of a tensor.
        - valid: (Optional) Whether or not we are inside the validation loop, boolean value.

        Outputs:
        - mt_loss: A float containing the measured meta-training loss of the task.
        """
        temp_weights = [param for param in self._model.parameters()]

        for inner_step in range(self._inner_steps):
            output = self._model(X_train, temp_weights)
            loss = self._inner_criterion(output, y_train)

            grads = torch.autograd.grad(loss, temp_weights)
            
            temp_weights = [w - self._alpha * g for w, g in zip(temp_weights, grads)]

        mt_output = self._model(X_test, temp_weights)
        mt_loss = self._inner_criterion(mt_output, y_test)

        if not valid:
            mt_loss.backward()

        return float(mt_loss)

    def train(self, epochs):
        """
        This method is the main method called to initiate the training of the MAML object.
        Here we iterate over all task data provided for a number of epochs to train the model 
        and find a set of parameters that are able to generalise well to all tasks in the 
        datasets.

        Inputs:
        - epochs: Integer defining the number of epochs to train for.
        """
        meta_iterations = 0

        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0

            for n, data in enumerate(self._metatrain_dataloader):
                X_train, y_train = data["train"]
                X_test, y_test = data["test"]
                self._optimizer.zero_grad()

                losses = list(map(self.inner_loop, X_train, y_train, X_test, y_test))
                running_loss += np.mean(losses)

                for param in self._model.parameters():
                    param.grad = param.grad / self._batch_size

                self._optimizer.step()
                meta_iterations += 1

                if meta_iterations % self._print_every == 0:
                    mt_running_loss = 0
                    for m, mt_data in enumerate(self._metatest_dataloader):
                        mt_X_train, mt_y_train = mt_data["train"]
                        mt_X_test, mt_y_test = mt_data["test"]

                        mt_losses = list(map(self.inner_loop, mt_X_train, mt_y_train, mt_X_test, mt_y_test, repeat(True)))

                        mt_running_loss += np.mean(mt_losses)

                    mt_valid_loss = mt_running_loss / (m + 1)

                    print(f"Meta iteration: {meta_iterations} .. Training loss: {running_loss / meta_iterations}")
                    print(f"Validation loss: {mt_valid_loss}")
                    self.iterations.append(meta_iterations)
                    self.training_losses.append(running_loss / meta_iterations) 
                    self.validation_losses.append(mt_valid_loss) 

    def __call__(self, epochs):
        self.train(epochs)
                    





    