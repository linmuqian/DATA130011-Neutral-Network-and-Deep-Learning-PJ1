import numpy as np
import os
from tqdm import tqdm
import pandas as pd


class RunnerM():
    """
    This is an example to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.log_data = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]

                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.model.clear_grad()
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")
                    self.log_data.append([epoch, iteration, trn_loss, trn_score, dev_loss, dev_score])

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score

        self.best_score = best_score

        # Save the training log to an Excel file
        df = pd.DataFrame(self.log_data, columns=['Epoch', 'Iteration', 'Train Loss', 'Train Score', 'Dev Loss', 'Dev Score'])
        excel_path = os.path.join(save_dir, 'training_log.xlsx')
        df.to_excel(excel_path, index=False)

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)
    


class RunnerM_early():
    """
    This is a model with early stopping.
    """

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model 
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.log_data = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        patience = kwargs.get("patience", 10)  # Early stopping patience

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        best_epoch = 0
        no_improvement_count = 0  # Counter for early stopping: if no improvement, increse this counter

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]

                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.model.clear_grad()
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")
                    self.log_data.append([epoch, iteration, trn_loss, trn_score, dev_loss, dev_score])

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation score for {patience} epochs.")
                break

        self.best_score = best_score

        # Save the training log to an Excel file
        df = pd.DataFrame(self.log_data, columns=['Epoch', 'Iteration', 'Train Loss', 'Train Score', 'Dev Loss', 'Dev Score'])
        excel_path = os.path.join(save_dir, 'training_log.xlsx')
        df.to_excel(excel_path, index=False)

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)


class RunnerM_c():
    """
    This is a model to test CNN with early stopping.
    Since CNN model is too slow to train, we do not score the dev set every iteration.
    """

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.log_data = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        patience = kwargs.get("patience", 10)  # Early stopping patience

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        best_epoch = 0
        no_improvement_count = 0  # Counter for early stopping: if no improvement, increase this counter
        last_dev_score = 0
        last_dev_loss = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]

                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.model.clear_grad()
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if (iteration + 1) % 50 == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    last_dev_score = dev_score
                    last_dev_loss = dev_loss
                else:
                    dev_score = last_dev_score
                    dev_loss = last_dev_loss

                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")
                    self.log_data.append([epoch, iteration, trn_loss, trn_score, dev_loss, dev_score])

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation score for {patience} epochs.")
                break

        self.best_score = best_score

        # Save the training log to an Excel file
        df = pd.DataFrame(self.log_data, columns=['Epoch', 'Iteration', 'Train Loss', 'Train Score', 'Dev Loss', 'Dev Score'])
        excel_path = os.path.join(save_dir, 'training_log.xlsx')
        df.to_excel(excel_path, index=False)
    
    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)