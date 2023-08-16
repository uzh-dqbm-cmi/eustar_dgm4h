import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional
import pickle

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
import json

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


from ...data.datasets import custom_collate


class BaseTrainer:
    """Base class to perform model training.

    Args:
        model (BaseAE): A instance of :class:`~pythae.models.BaseAE` to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        eval_dataset (BaseDataset): The evaluation dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_config (BaseTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`BaseTrainerConfig` is used. Default: None.

        optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training. If None, a :class:`~torch.optim.Adam` optimizer is used. Default: None.

        scheduler (~torch.optim.lr_scheduler): An instance of `torch.optim.Optimizer` used for
            training. If None, a :class:`~torch.optim.Adam` optimizer is used. Default: None.

        callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
            A list of callbacks to use during training.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[BaseTrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        if training_config is None:
            training_config = BaseTrainerConfig()

        if training_config.output_dir is None:
            output_dir = "dummy_output_dir"
            training_config.output_dir = output_dir

        if not os.path.exists(training_config.output_dir):
            os.makedirs(training_config.output_dir)
            logger.info(
                f"Created {training_config.output_dir} folder since did not exist.\n"
            )

        self.training_config = training_config

        set_seed(self.training_config.seed)

        device = (
            "cuda"
            if torch.cuda.is_available() and not training_config.no_cuda
            else "cpu"
        )

        # place model on device
        model = model.to(device)
        model.device = device

        # set optimizer
        if optimizer is None:
            optimizer = self.set_default_optimizer(model)

        else:
            optimizer = self._set_optimizer_on_device(optimizer, device)

        # set scheduler
        if scheduler is None:
            scheduler = self.set_default_scheduler(model, optimizer)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.eval_results = {}

        self.device = device

        # Define the loaders
        train_loader = self.get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            eval_loader = self.get_eval_dataloader(eval_dataset)

        else:
            logger.info(
                "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        if callbacks is None:
            callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=callbacks, model=model, optimizer=optimizer, scheduler=scheduler
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

    def set_default_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:
        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )

        return optimizer

    def set_default_scheduler(
        self, model: BaseAE, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, verbose=True
        )

        return scheduler

    def _run_model_sanity_check(self, model, loader):
        try:
            inputs = next(iter(loader))
            train_dataset = self._set_inputs_to_device(inputs)
            model(train_dataset)

        except Exception as e:
            raise ModelError(
                "Error when calling forward method from model. Potential issues: \n"
                " - Wrong model architecture -> check encoder, decoder and metric architecture if "
                "you provide yours \n"
                " - The data input dimension provided is wrong -> when no encoder, decoder or metric "
                "provided, a network is built automatically but requires the shape of the flatten "
                "input data.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

    def _set_optimizer_on_device(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

        return optim

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):
        inputs_on_device = inputs

        if self.device == "cuda":
            cuda_inputs = dict.fromkeys(inputs)

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    cuda_inputs[key] = inputs[key].cuda()

                else:
                    cuda_inputs[key] = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device

    def _optimizers_step(self, model_output=None):
        loss = model_output.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _schedulers_step(self, metrics=None):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics)

        else:
            self.scheduler.step()

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
        self.all_losses_train = torch.empty(
            self.training_config.num_epochs, 3 + len(self.model.classifiers)
        )
        self.loss_train_cv = torch.empty(self.training_config.num_epochs, 1)

        self.all_losses_valid = torch.empty(
            self.training_config.num_epochs, 3 + len(self.model.classifiers)
        )
        self.loss_valid_cv = torch.empty(self.training_config.num_epochs, 1)

        self.all_losses_train_unw = torch.empty(
            self.training_config.num_epochs, 3 + len(self.model.classifiers)
        )
        self.all_losses_valid_unw = torch.empty(
            self.training_config.num_epochs, 3 + len(self.model.classifiers)
        )
        self.all_losses_train_pred = torch.empty(
            self.training_config.num_epochs, 2 + len(self.model.classifiers)
        )
        self.all_losses_valid_pred = torch.empty(
            self.training_config.num_epochs, 2 + len(self.model.classifiers)
        )
        self.all_losses_train_unw_pred = torch.empty(
            self.training_config.num_epochs, 2 + len(self.model.classifiers)
        )
        self.all_losses_valid_unw_pred = torch.empty(
            self.training_config.num_epochs, 2 + len(self.model.classifiers)
        )
        self.losses_recon_stack_train = torch.empty(
            self.training_config.num_epochs, len(self.model.splits_x0)
        )
        self.losses_recon_stack_eval = torch.empty(
            self.training_config.num_epochs, len(self.model.splits_x0)
        )
        self.losses_recon_stack_train_pred = torch.empty(
            self.training_config.num_epochs, len(self.model.splits_x0)
        )
        self.losses_recon_stack_eval_pred = torch.empty(
            self.training_config.num_epochs, len(self.model.splits_x0)
        )
        self.loss_ce_recon_train = torch.empty(self.training_config.num_epochs)
        self.loss_ce_recon_train_pred = torch.empty(self.training_config.num_epochs)
        self.loss_nll_recon_train = torch.empty(self.training_config.num_epochs)
        self.loss_nll_recon_train_pred = torch.empty(self.training_config.num_epochs)
        self.loss_ce_recon_eval = torch.empty(self.training_config.num_epochs)
        self.loss_ce_recon_eval_pred = torch.empty(self.training_config.num_epochs)
        self.loss_nll_recon_eval = torch.empty(self.training_config.num_epochs)
        self.loss_nll_recon_eval_pred = torch.empty(self.training_config.num_epochs)
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
                epoch_train_loss_cv,
                epoch_train_losses,
                epoch_train_losses_unw,
                epoch_train_recon_stack,
                epoch_train_losses_pred,
                epoch_train_losses_unw_pred,
                epoch_train_recon_stack_pred,
                epoch_train_loss_ce,
                epoch_train_loss_ce_pred,
                epoch_train_loss_nll,
                epoch_train_loss_nll_pred,
            ) = self.train_step(epoch)
            self.all_losses_train[epoch - 1, :] = epoch_train_losses
            self.loss_train_cv[epoch - 1, :] = epoch_train_loss_cv
            self.all_losses_train_unw[epoch - 1, :] = epoch_train_losses_unw
            self.losses_recon_stack_train[epoch - 1, :] = epoch_train_recon_stack
            self.all_losses_train_pred[epoch - 1, :] = epoch_train_losses_pred
            self.all_losses_train_unw_pred[epoch - 1, :] = epoch_train_losses_unw_pred
            self.losses_recon_stack_train_pred[
                epoch - 1, :
            ] = epoch_train_recon_stack_pred
            self.loss_ce_recon_train[epoch - 1] = epoch_train_loss_ce
            self.loss_ce_recon_train_pred[epoch - 1] = epoch_train_loss_ce_pred
            self.loss_nll_recon_train[epoch - 1] = epoch_train_loss_nll
            self.loss_nll_recon_train_pred[epoch - 1] = epoch_train_loss_nll_pred

            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_epoch_loss_cv"] = epoch_train_loss_cv
            metrics["train_epoch_losses"] = epoch_train_losses.cpu()
            metrics["train_epoch_losses_unw"] = epoch_train_losses_unw.cpu()
            metrics["train_epoch_losses_pred"] = epoch_train_losses_pred.cpu()
            metrics["train_epoch_losses_unw_pred"] = epoch_train_losses_unw_pred.cpu()
            metrics["train_epoch_loss_ce"] = epoch_train_loss_ce.item()
            metrics["train_epoch_loss_ce_pred"] = epoch_train_loss_ce_pred.item()
            metrics["train_epoch_loss_nll"] = epoch_train_loss_nll.item()
            metrics["train_epoch_loss_nll_pred"] = epoch_train_loss_nll_pred.item()

            if self.eval_dataset is not None:
                (
                    epoch_eval_loss,
                    epoch_eval_loss_cv,
                    epoch_eval_losses,
                    epoch_eval_losses_unw,
                    epoch_eval_recon_stack,
                    epoch_eval_losses_pred,
                    epoch_eval_losses_unw_pred,
                    epoch_eval_recon_stack_pred,
                    epoch_eval_loss_ce,
                    epoch_eval_loss_ce_pred,
                    epoch_eval_loss_nll,
                    epoch_eval_loss_nll_pred,
                ) = self.eval_step(epoch)
                self.all_losses_valid[epoch - 1, :] = epoch_eval_losses
                self.loss_valid_cv[epoch - 1, :] = epoch_eval_loss_cv
                self.all_losses_valid_unw[epoch - 1, :] = epoch_eval_losses_unw
                self.losses_recon_stack_eval[epoch - 1, :] = epoch_eval_recon_stack
                self.all_losses_valid_pred[epoch - 1, :] = epoch_eval_losses_pred
                self.all_losses_valid_unw_pred[
                    epoch - 1, :
                ] = epoch_eval_losses_unw_pred
                self.losses_recon_stack_eval_pred[
                    epoch - 1, :
                ] = epoch_eval_recon_stack_pred
                self.loss_ce_recon_eval[epoch - 1] = epoch_eval_loss_ce
                self.loss_ce_recon_eval_pred[epoch - 1] = epoch_eval_loss_ce_pred
                self.loss_nll_recon_eval[epoch - 1] = epoch_eval_loss_nll
                self.loss_nll_recon_eval_pred[epoch - 1] = epoch_eval_loss_nll_pred
                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_epoch_loss_cv"] = epoch_eval_loss_cv
                metrics["eval_epoch_losses"] = epoch_eval_losses.cpu()
                metrics["eval_epoch_losses_unw"] = epoch_eval_losses_unw.cpu()
                metrics["eval_epoch_losses_pred"] = epoch_eval_losses_pred.cpu()
                metrics["eval_epoch_losses_unw_pred"] = epoch_eval_losses_unw_pred.cpu()
                metrics["eval_epoch_loss_ce"] = epoch_eval_loss_ce.item()
                metrics["eval_epoch_loss_ce_pred"] = epoch_eval_loss_ce_pred.item()
                metrics["eval_epoch_loss_nll"] = epoch_eval_loss_nll.item()
                metrics["eval_epoch_loss_nll_pred"] = epoch_eval_loss_nll_pred.item()

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
                self.eval_results["best_eval_loss"] = best_eval_loss
                self.eval_results["epochs"] = epoch

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model
                self.eval_results["best_train_loss"] = best_train_loss
                self.eval_results["epochs"] = epoch

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

            self.callback_handler.on_log(
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
        epoch_loss_cv = 0
        epoch_loss_ce = 0
        epoch_loss_nll = 0
        epoch_loss_ce_pred = 0
        epoch_loss_nll_pred = 0
        epoch_losses = torch.zeros(
            3 + len(self.model.classifiers), device=self.model.device
        )  # [overall_loss, rec_loss, beta*kl, class_loss_1, class_loss_2,...]

        epoch_recon_losses_stacked = torch.zeros(
            len(self.model.splits_x0), device=self.model.device, requires_grad=False
        )

        epoch_losses_unweighted = torch.zeros(
            3 + len(self.model.classifiers), device=self.model.device
        )

        epoch_losses_pred = torch.zeros(
            2 + len(self.model.classifiers), device=self.model.device
        )  # [overall_loss, rec_loss, beta*kl, class_loss_1, class_loss_2,...]

        epoch_recon_losses_stacked_pred = torch.zeros(
            len(self.model.splits_x0), device=self.model.device, requires_grad=False
        )

        epoch_losses_unweighted_pred = torch.zeros(
            2 + len(self.model.classifiers), device=self.model.device
        )

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

            epoch_loss += loss.item()
            epoch_loss_cv += model_output.loss_cv.item()

            # epoch_losses += torch.tensor(
            #     [
            #         model_output.loss.item(),
            #         model_output.loss_recon.item(),
            #         model_output.loss_kld.item(),
            #         model_output.loss_class.item(),
            #     ]
            # )
            epoch_loss_nll += model_output.loss_recon_nll.detach()
            epoch_loss_ce += model_output.loss_recon_ce.detach()
            epoch_loss_nll_pred += model_output.loss_recon_nll_pred.detach()
            epoch_loss_ce_pred += model_output.loss_recon_ce_pred.detach()
            epoch_losses += (
                torch.cat(
                    [
                        torch.sum(model_output.losses_weighted).reshape(-1, 1),
                        model_output.losses_weighted,
                    ]
                )
                .detach()
                .flatten()
            )

            epoch_losses_unweighted += torch.cat(
                [
                    torch.sum(model_output.losses_unweighted).reshape(-1, 1),
                    model_output.losses_unweighted,
                ]
            ).flatten()

            epoch_losses_pred += (
                torch.cat(
                    [
                        torch.sum(model_output.losses_weighted_pred).reshape(-1, 1),
                        model_output.losses_weighted_pred,
                    ]
                )
                .detach()
                .flatten()
            )

            epoch_losses_unweighted_pred += torch.cat(
                [
                    torch.sum(model_output.losses_unweighted_pred).reshape(-1, 1),
                    model_output.losses_unweighted_pred,
                ]
            ).flatten()

            epoch_recon_losses_stacked += model_output.loss_recon_stacked.detach()
            epoch_recon_losses_stacked_pred += (
                model_output.loss_recon_stacked_pred.detach()
            )

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_loss /= len(self.eval_loader)
        epoch_loss_cv /= len(self.eval_loader)
        epoch_losses /= len(self.eval_loader)
        epoch_losses_unweighted /= len(self.eval_loader)
        epoch_recon_losses_stacked /= len(self.eval_loader)

        epoch_losses_pred /= len(self.eval_loader)
        epoch_losses_unweighted_pred /= len(self.eval_loader)
        epoch_recon_losses_stacked_pred /= len(self.eval_loader)
        epoch_loss_ce /= len(self.eval_loader)
        epoch_loss_ce_pred /= len(self.eval_loader)
        epoch_loss_nll /= len(self.eval_loader)
        epoch_loss_nll_pred /= len(self.eval_loader)

        return (
            epoch_loss,
            epoch_loss_cv,
            epoch_losses,
            epoch_losses_unweighted,
            epoch_recon_losses_stacked,
            epoch_losses_pred,
            epoch_losses_unweighted_pred,
            epoch_recon_losses_stacked_pred,
            epoch_loss_ce,
            epoch_loss_ce_pred,
            epoch_loss_nll,
            epoch_loss_nll_pred,
        )

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
        epoch_loss_cv = 0
        epoch_loss_ce = 0
        epoch_loss_nll = 0
        epoch_loss_ce_pred = 0
        epoch_loss_nll_pred = 0
        epoch_losses = torch.zeros(
            3 + len(self.model.classifiers), device=self.model.device
        )  # [overall_loss, rec_loss, beta*kl, class_loss_1, class_loss_2,...]

        epoch_losses_unweighted = torch.zeros(
            3 + len(self.model.classifiers), device=self.model.device
        )

        epoch_recon_losses_stacked = torch.zeros(
            len(self.model.splits_x0), device=self.model.device, requires_grad=False
        )
        epoch_losses_pred = torch.zeros(
            2 + len(self.model.classifiers), device=self.model.device
        )  # [overall_loss, rec_loss, beta*kl, class_loss_1, class_loss_2,...]

        epoch_losses_unweighted_pred = torch.zeros(
            2 + len(self.model.classifiers), device=self.model.device
        )

        epoch_recon_losses_stacked_pred = torch.zeros(
            len(self.model.splits_x0), device=self.model.device, requires_grad=False
        )
        for inputs in self.train_loader:
            inputs = self._set_inputs_to_device(inputs)
            model_output = self.model(
                inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
            )

            self._optimizers_step(model_output)

            loss = model_output.loss
            epoch_loss_cv += model_output.loss_cv.item()
            epoch_loss += loss.item()
            epoch_loss_nll += model_output.loss_recon_nll.detach()
            epoch_loss_ce += model_output.loss_recon_ce.detach()
            epoch_loss_nll_pred += model_output.loss_recon_nll_pred.detach()
            epoch_loss_ce_pred += model_output.loss_recon_ce_pred.detach()
            # epoch_losses += torch.tensor(
            #     [
            #         model_output.loss.item(),
            #         model_output.loss_recon.item(),
            #         model_output.loss_kld.item(),
            #         model_output.loss_class.item(),
            #     ]
            # )

            epoch_losses += (
                torch.cat(
                    [
                        torch.sum(model_output.losses_weighted).reshape(-1, 1),
                        model_output.losses_weighted,
                    ]
                )
                .detach()
                .flatten()
            )

            epoch_losses_unweighted += (
                torch.cat(
                    [
                        torch.sum(model_output.losses_unweighted).reshape(-1, 1),
                        model_output.losses_unweighted,
                    ]
                )
                .detach()
                .flatten()
            )
            epoch_losses_pred += (
                torch.cat(
                    [
                        torch.sum(model_output.losses_weighted_pred).reshape(-1, 1),
                        model_output.losses_weighted_pred,
                    ]
                )
                .detach()
                .flatten()
            )

            epoch_losses_unweighted_pred += (
                torch.cat(
                    [
                        torch.sum(model_output.losses_unweighted_pred).reshape(-1, 1),
                        model_output.losses_unweighted_pred,
                    ]
                )
                .detach()
                .flatten()
            )

            epoch_recon_losses_stacked += model_output.loss_recon_stacked.detach()
            epoch_recon_losses_stacked_pred += (
                model_output.loss_recon_stacked_pred.detach()
            )

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        self.model.update()

        epoch_loss /= len(self.train_loader)
        epoch_loss_cv /= len(self.train_loader)
        epoch_losses /= len(self.train_loader)
        epoch_losses_unweighted /= len(self.train_loader)
        epoch_recon_losses_stacked /= len(self.train_loader)

        epoch_losses_pred /= len(self.train_loader)
        epoch_losses_unweighted_pred /= len(self.train_loader)
        epoch_recon_losses_stacked_pred /= len(self.train_loader)
        epoch_loss_ce /= len(self.train_loader)
        epoch_loss_ce_pred /= len(self.train_loader)
        epoch_loss_nll /= len(self.train_loader)
        epoch_loss_nll_pred /= len(self.train_loader)

        return (
            epoch_loss,
            epoch_loss_cv,
            epoch_losses,
            epoch_losses_unweighted,
            epoch_recon_losses_stacked,
            epoch_losses_pred,
            epoch_losses_unweighted_pred,
            epoch_recon_losses_stacked_pred,
            epoch_loss_ce,
            epoch_loss_ce_pred,
            epoch_loss_nll,
            epoch_loss_nll_pred,
        )

    def save_model(self, model: BaseAE, dir_path: str):
        """This method saves the final model along with the config files

        Args:
            model (BaseAE): The model to be saved
            dir_path (str): The folder where the model and config files should be saved
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save model
        model.save(dir_path)

        # save training config
        self.training_config.save_json(dir_path, "training_config")
        # save evaluation results
        with open(dir_path + "/eval_results.json", "w") as f:
            json.dump(self.eval_results, f)

        self.callback_handler.on_save(self.training_config)

    def save_checkpoint(self, model: BaseAE, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizer
        torch.save(
            deepcopy(self.optimizer.state_dict()),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )

        # save scheduler
        torch.save(
            deepcopy(self.scheduler.state_dict()),
            os.path.join(checkpoint_dir, "scheduler.pt"),
        )

        # save model
        model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")

    def predict(self, model: BaseAE):
        model.eval()

        # with torch.no_grad():

        inputs = self.eval_loader.dataset[
            : min(self.eval_loader.dataset.data.shape[0], 10)
        ]
        inputs = self._set_inputs_to_device(inputs)

        model_out = model(inputs)
        reconstructions = model_out.recon_x.cpu().detach()
        z_enc = model_out.z
        z = torch.randn_like(z_enc)
        normal_generation = model.decoder(z).reconstruction.detach().cpu()

        return inputs["data"], reconstructions, normal_generation


class CustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=custom_collate,
        )

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )
