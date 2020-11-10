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
    def __init__(self, model, alpha, beta, inner_steps, metatrain_dataloader, metatest_dataloader, 
                    inner_criterion="mse", optimizer="adam", batch_size=2):
        super(MAML, self).__init__()

        self._model = model

        self._train_on_gpu = True if torch.cuda.is_available() else False

        self._alpha = alpha
        self._beta = beta

        self._inner_steps = inner_steps
        self._batch_size = batch_size

        self._metatrain_dataloader = metatrain_dataloader
        self._metatest_dataloader = metatest_dataloader

        self._inner_criterion = CRITERION[inner_criterion.lower()]()
        self._optimizer = OPTIMIZER[optimizer.lower()](model.parameters(), beta)

    def inner_loop(self, X_train, y_train, X_test, y_test, valid=False):
        
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

                if meta_iterations % 50 == 0:
                    mt_running_loss = 0
                    for m, mt_data in enumerate(self._metatest_dataloader):
                        mt_X_train, mt_y_train = mt_data["train"]
                        mt_X_test, mt_y_test = mt_data["test"]

                        mt_losses = list(map(self.inner_loop, mt_X_train, mt_y_train, mt_X_test, mt_y_test, repeat(True)))

                        mt_running_loss += np.mean(losses)

                    mt_valid_loss = mt_running_loss / m

                    print(f"Meta iteration: {meta_iterations} .. Training loss: {running_loss / meta_iterations}")
                    print(f"Validation loss: {mt_valid_loss}")

    def __call__(self, epochs):
        self.train(epochs)
                    





    