import torch
import torch.nn as nn

CRITERION = {
    "mse": nn.MSELoss
}

OPTIMIZER = {
    "adam": torch.optim.Adam
}

class MAML(nn.Module):
    def __init__(self, model, alpha, beta, inner_steps, metatrain_dataloader, metatest_dataloader, 
                    inner_criterion="mse", optimizer="adam"):
        super(MAML, self).__init__()

        self._model = model

        self._train_on_gpu = True if torch.cuda.is_available() else False

        self._alpha = alpha
        self._beta = beta

        self._inner_steps = inner_steps

        self._metatrain_dataloader = metatrain_dataloader
        self._metatest_dataloader = metatest_dataloader

        self._inner_criterion = CRITERION[inner_criterion.lower()]()
        self._optimizer = OPTIMIZER[optimizer.lower()](model.parameters(), beta)

    # def weight_update(self, weights, grads):
    #     for w, g in zip(weights, grads):
    #         w -= self._alpha * g

    #     return weights

    def inner_loop(self, data):
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        temp_weights = [param for param in self._model.parameters()]

        for inner_step in range(self._inner_steps):
            output = self._model(X_train, temp_weights)
            loss = self._inner_criterion(output, y_train)
            print(f"inner_step: {inner_step} - Loss: {loss}")

            grads = torch.autograd.grad(loss, temp_weights)
            
            # temp_weights = self.weight_update(temp_weights, grads)
            temp_weights = [w - self._alpha * g for w, g in zip(temp_weights, grads)]

        mt_output = self._model(X_test, temp_weights)
        mt_loss = self._inner_criterion(mt_output, y_test)
        mt_loss.backward()
        print(f"Test loss: {mt_loss} \n")

        return float(mt_loss)

    def train(self, epochs):
        meta_iterations = 0

        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0

            for data in self._metatrain_dataloader:
                running_loss += self.inner_loop(data)

                self._optimizer.step()
                meta_iterations += 1

                # if meta_iterations % 10 == 0:

    def __call__(self, epochs):
        self.train(epochs)
                    





    