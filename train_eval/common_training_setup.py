import multiprocessing
import os
from os import path

import torch
from torch.utils.tensorboard import SummaryWriter

from train_eval.training import train


class TrainingConfig:
    def __init__(self, datasets, batch_size, nb_epochs, learning_rate, experiment_name,
                 scheduler, save_logic, ignore_validation=False, customized_collate_fn=None):
        self.datasets = datasets
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.scheduler = scheduler
        self.save_logic = save_logic

        self.ignore_validation = ignore_validation
        self.customized_collate_fn = customized_collate_fn


def run_specific_experiment(training_config, training_logic, training_args,eval_logic=None, eval_args=None, summary=None):
    if summary is None:
        summary = SummaryWriter()

    train_set, test_set = training_config.datasets
    model_name = training_config.experiment_name

    batch_size = training_config.batch_size
    nb_epochs = training_config.nb_epochs
    ignore_validation = training_config.ignore_validation
    summary.add_hparams({"model_name": model_name,
                         "learning rate": training_config.learning_rate,
                         "batch size": batch_size,
                         "max epochs": nb_epochs}, {})

    out_folder = "saved_models/{}".format(model_name)
    if not path.exists(out_folder):
        os.makedirs(out_folder)

    if training_args is not None:
        train(training_config,
              train_set,
              training_logic,
              training_args,
              batch_size,
              nb_epochs,
              summary,
              out_folder,
              ignore_validation=ignore_validation,
              eval_logic=eval_logic)
        print("Finished Training")
        del train_set

    if eval_logic is not None and eval_args is not None:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=multiprocessing.cpu_count(),
                                                  shuffle=False, persistent_workers=True, prefetch_factor=6,
                                                  collate_fn=training_config.customized_collate_fn)
        eval_results, eval_scores = eval_logic(test_loader, **eval_args)
        if "src" in eval_results and "pred" in eval_results:
            print(eval_scores)

    summary.close()
