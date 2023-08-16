from .base_trainer import CustomTrainer
import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ...customexception import ModelError
from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..trainer_utils import set_seed
from ..training_callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from .base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class RNNTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model.model_config
        )

        # run sanity check on the model
        self._run_model_sanity_check(self.model, self.train_loader)

        logger.info("Model passed sanity check !\n")

        self._training_signature = (
            str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.training_config.output_dir,
            f"{self.model.model_name}_training_{self._training_signature}",
        )

        self.training_dir = training_dir

        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            logger.info(
                f"Created {training_dir}. \n"
                "Training config, checkpoints and final model will be saved here.\n"
            )

        log_verbose = False

        # set up log file
        if log_output_dir is not None:
            log_dir = log_output_dir
            log_verbose = True

            # if dir does not exist create it
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                logger.info(f"Created {log_dir} folder since did not exists.")
                logger.info("Training logs will be recodered here.\n")
                logger.info(" -> Training can be monitored here.\n")

            # create and set logger
            log_name = f"training_logs_{self._training_signature}"

            file_logger = logging.getLogger(log_name)
            file_logger.setLevel(logging.INFO)
            f_handler = logging.FileHandler(
                os.path.join(log_dir, f"training_logs_{self._training_signature}.log")
            )
            f_handler.setLevel(logging.INFO)
            file_logger.addHandler(f_handler)

            # Do not output logs in the console
            file_logger.propagate = False

            file_logger.info("Training started !\n")
            file_logger.info(
                f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
                f" - batch_size: {self.training_config.batch_size}\n"
                f" - checkpoint saving every {self.training_config.steps_saving}\n"
            )

            file_logger.info(f"Model Architecture: {self.model}\n")
            file_logger.info(f"Optimizer: {self.optimizer}\n")

        logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10
        # is this already stored anywhere else ?
        self.losses_train = torch.empty(self.training_config.num_epochs)

        self.losses_valid = torch.empty(self.training_config.num_epochs)
        self.losses_x_train = torch.empty(self.training_config.num_epochs)
        self.losses_y_train = torch.empty(self.training_config.num_epochs)
        self.losses_x_valid = torch.empty(self.training_config.num_epochs)
        self.losses_y_valid = torch.empty(self.training_config.num_epochs)

        for epoch in range(1, self.training_config.num_epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = {}

            (
                epoch_train_loss,
                epoch_train_loss_x,
                epoch_train_loss_y,
                epoch_train_loss_ce,
                epoch_train_loss_nll,
            ) = self.train_step(epoch)
            self.losses_train[epoch - 1] = epoch_train_loss
            self.losses_x_train[epoch - 1] = epoch_train_loss_x
            self.losses_y_train[epoch - 1] = epoch_train_loss_y

            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_epoch_loss_x"] = epoch_train_loss_x
            metrics["train_epoch_loss_y"] = epoch_train_loss_y
            metrics["train_epoch_loss_ce"] = epoch_train_loss_ce.item()
            metrics["train_epoch_loss_nll"] = epoch_train_loss_nll.item()

            if self.eval_dataset is not None:
                (
                    epoch_eval_loss,
                    epoch_eval_loss_x,
                    epoch_eval_loss_y,
                    epoch_eval_loss_ce,
                    epoch_eval_loss_nll,
                ) = self.eval_step(epoch)
                self.losses_valid[epoch - 1] = epoch_eval_loss
                self.losses_x_valid[epoch - 1] = epoch_eval_loss_x
                self.losses_y_valid[epoch - 1] = epoch_eval_loss_y

                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_epoch_loss_x"] = epoch_eval_loss_x
                metrics["eval_epoch_loss_y"] = epoch_eval_loss_y
                metrics["eval_epoch_loss_ce"] = epoch_eval_loss_ce.item()
                metrics["eval_epoch_loss_nll"] = epoch_eval_loss_nll.item()

                self._schedulers_step(epoch_eval_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(epoch_train_loss)

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            if (
                self.training_config.steps_predict is not None
                and epoch % self.training_config.steps_predict == 0
            ):
                true_data, reconstructions, generations = self.predict(best_model)

                self.callback_handler.on_prediction_step(
                    self.training_config,
                    true_data=true_data,
                    reconstructions=reconstructions,
                    generations=generations,
                    global_step=epoch,
                )
            self.callback_handler.on_epoch_end(training_config=self.training_config)

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                self.save_checkpoint(
                    model=best_model, dir_path=training_dir, epoch=epoch
                )
                logger.info(f"Saved checkpoint at epoch {epoch}\n")

                if log_verbose:
                    file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log_rnn(
                self.training_config, metrics, logger=logger, global_step=epoch
            )

        final_dir = os.path.join(training_dir, "final_model")

        self.save_model(best_model, dir_path=final_dir)
        logger.info("Training ended!")
        logger.info(f"Saved final model in {final_dir}")

        self.callback_handler.on_train_end(self.training_config)

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
        )

        self.model.eval()

        epoch_loss = 0
        epoch_loss_x = 0
        epoch_loss_y = 0
        epoch_loss_nll = 0
        epoch_loss_ce = 0

        for inputs in self.eval_loader:
            inputs = self._set_inputs_to_device(inputs)

            try:
                with torch.no_grad():
                    model_output = self.model(
                        inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
                    )

            except RuntimeError:
                model_output = self.model(
                    inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
                )

            loss = model_output.loss
            loss_x = model_output.loss_x
            loss_y = model_output.loss_y

            epoch_loss += loss.item()
            epoch_loss_x += loss_x.item()
            epoch_loss_y += loss_y.item()
            epoch_loss_ce += model_output.loss_recon_ce.detach()
            epoch_loss_nll += model_output.loss_recon_nll.detach()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_loss /= len(self.eval_loader)
        epoch_loss_x /= len(self.eval_loader)
        epoch_loss_y /= len(self.eval_loader)
        epoch_loss_nll /= len(self.eval_loader)
        epoch_loss_ce /= len(self.eval_loader)

        return epoch_loss, epoch_loss_x, epoch_loss_y, epoch_loss_ce, epoch_loss_nll

    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """

        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        # set model in train model
        self.model.train()

        epoch_loss = 0
        epoch_loss_x = 0
        epoch_loss_y = 0
        epoch_loss_ce = 0
        epoch_loss_nll = 0
        for inputs in self.train_loader:
            inputs = self._set_inputs_to_device(inputs)
            model_output = self.model(
                inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
            )

            self._optimizers_step(model_output)

            loss = model_output.loss
            loss_x = model_output.loss_x
            loss_y = model_output.loss_y

            epoch_loss += loss.item()
            epoch_loss_x += loss_x.item()
            epoch_loss_y += loss_y.item()
            epoch_loss_nll += model_output.loss_recon_nll.detach()
            epoch_loss_ce += model_output.loss_recon_ce.detach()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        self.model.update()

        epoch_loss /= len(self.train_loader)
        epoch_loss_x /= len(self.train_loader)
        epoch_loss_y /= len(self.train_loader)
        epoch_loss_ce /= len(self.train_loader)
        epoch_loss_nll /= len(self.train_loader)

        return epoch_loss, epoch_loss_x, epoch_loss_y, epoch_loss_ce, epoch_loss_nll
