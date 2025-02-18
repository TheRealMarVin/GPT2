import time
from datetime import timedelta

import numpy as np
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm


def train(training_config, train_dataset, training_logic, training_args, batch_size,
          n_epochs, summary, save_file,
          early_stop=None, train_ratio=0.85, true_index=1,
          ignore_validation=False,
          reshuffle_train=10,
          eval_logic=None):

    if ignore_validation:
        nb_elem_train = len(train_dataset)
    else:
        nb_elem_train = int(len(train_dataset) * train_ratio)

    best_valid_loss = float("inf")
    reshuffle_counter = 0

    for epoch in range(n_epochs):
        if "epoch" in training_args:
            training_args["epoch"] = epoch

        if reshuffle_counter % reshuffle_train == 0:
            if ignore_validation:
                train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8,
                                                             shuffle=True, drop_last=True, persistent_workers=True,
                                                             prefetch_factor=6, collate_fn=training_config.customized_collate_fn)
            else:
                x, y = torch.utils.data.random_split(train_dataset, [nb_elem_train, len(train_dataset) - nb_elem_train])
                train_iterator = torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=8, shuffle=True,
                                                             drop_last=True, persistent_workers=True, prefetch_factor=6,
                                                             collate_fn=training_config.customized_collate_fn)
                eval_iterator = torch.utils.data.DataLoader(y, batch_size=batch_size, num_workers=8, shuffle=False,
                                                            drop_last=True, persistent_workers=True, prefetch_factor=6,
                                                            collate_fn=training_config.customized_collate_fn)
        reshuffle_counter += 1

        start_time = time.time()

        train_metrics = training_logic(iterator=train_iterator, **training_args)
        scheduler_metrics = train_metrics
        if not ignore_validation and "metrics_dict" in training_args and eval_logic != None:
            evaluation_metrics = training_args["metrics_dict"]
            global_metrics_dict = {}
            if "global_metrics_dict" in training_args:
                global_metrics_dict = training_args["global_metrics_dict"]

            _, validation_metrics = eval_logic(eval_iterator, training_args["model"],
                                               metrics_dict=evaluation_metrics, global_metrics_dict=global_metrics_dict,
                                               true_index=true_index)
            scheduler_metrics = validation_metrics

        if training_config.scheduler is not None:
            if type(training_config.scheduler) == lr_scheduler.ReduceLROnPlateau:
                training_config.scheduler.step(scheduler_metrics["loss"])
            else:
                training_config.scheduler.step()

        end_time = time.time()

        delta_time = timedelta(seconds=(end_time - start_time))

        if scheduler_metrics["loss"] < best_valid_loss:
            best_valid_loss = scheduler_metrics["loss"]
            training_config.save_logic(save_file)

        header_str = "Current Epoch: {} -> train_eval time: {}".format(epoch + 1, delta_time)
        train_str = metrics_to_string(train_metrics, "train")
        validation_str = ""
        if not ignore_validation:
            validation_str = metrics_to_string(validation_metrics, "val")
        print("{}\n\t{} - {}".format(header_str, train_str, validation_str))
        log_metrics_in_tensorboard(summary, train_metrics, epoch, "train")
        if not ignore_validation:
            log_metrics_in_tensorboard(summary, validation_metrics, epoch, "val")
        summary.flush()

        if early_stop is not None:
            if early_stop.should_stop(validation_metrics):
                break

    return best_valid_loss


def default_train_epoch(model, iterator, optimizer, criterion, metrics_dict, global_metrics_dict={}, true_index=1):
    if torch.cuda.is_available():
        model = model.cuda()

    metric_scores = {}
    all_pred = []
    all_true = []
    for k, _ in metrics_dict.items():
        metric_scores[k] = 0

    for k, _ in global_metrics_dict.items():
        metric_scores[k] = 0

    model.train()

    for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="train"):
        src = batch[0]
        y_true = batch[true_index]

        if len(y_true.shape) == 1:
            y_true = y_true.type("torch.LongTensor")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            src = src.cuda()
            y_true = y_true.cuda()

        optimizer.zero_grad()

        y_pred = model(src)
        loss = criterion(y_pred, y_true)

        loss.backward()
        optimizer.step()

        for k, metric in metrics_dict.items():
            metric_scores[k] += metric(y_pred, y_true).item()

        all_pred.extend(y_pred.detach().cpu().numpy())
        all_true.extend(y_true.detach().cpu().numpy())

        del src
        del loss
        del y_pred
        del y_true

    for k, v in metric_scores.items():
        metric_scores[k] = v / len(iterator)

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    for k, metric in global_metrics_dict.items():
        curr_epoch_metric = metric(all_pred, all_true)
        if type(curr_epoch_metric) is torch.Tensor:
            metric_scores[k] += curr_epoch_metric.item()
        else:
            metric_scores[k] += curr_epoch_metric
        del curr_epoch_metric

    return metric_scores


def log_metrics_in_tensorboard(summary, metrics, epoch, prefix):
    for k, val in metrics.items():
        summary.add_scalar("{}/{}".format(prefix, k), val, epoch + 1)


def metrics_to_string(metrics, prefix):
    res = []
    for k, val in metrics.items():
        res.append("{} {}:{:.4f}".format(prefix, k, val))

    return " ".join(res)
