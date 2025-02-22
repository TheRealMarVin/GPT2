import torch
from tqdm import tqdm


def next_token_train_epoch(model, iterator, optimizer, criterion, metrics_dict):
    if torch.cuda.is_available():
        model = model.cuda()

    metric_scores = {}
    for k, _ in metrics_dict.items():
        metric_scores[k] = 0

    model.train()

    for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="train"):
        src = batch["input_ids"]
        y_true = batch["labels"]

        if len(y_true.shape) == 1:
            y_true = y_true.type("torch.LongTensor")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            src = src.cuda()
            y_true = y_true.cuda()

        optimizer.zero_grad()

        y_pred = model(src)
        loss = criterion(y_pred.flatten(0,1), y_true.flatten())

        loss.backward()
        optimizer.step()

        for k, metric in metrics_dict.items():
            metric_scores[k] += metric(y_pred.flatten(0,1), y_true.flatten()).item()

        del src
        del loss
        del y_pred
        del y_true

    for k, v in metric_scores.items():
        metric_scores[k] = v / len(iterator)

    return metric_scores


def next_token_evaluate(iterator, model, metrics_dict, global_metrics_dict={}, true_index=1):
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    metric_scores = {}
    for k, _ in metrics_dict.items():
        metric_scores[k] = 0

    with torch.no_grad():
        all_pred = []
        all_true = []
        all_src = []

        for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="eval"):
            src = batch["input_ids"]
            y_true = batch["labels"]

            if len(y_true.shape) == 1:
                y_true = y_true.type("torch.LongTensor")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                src = src.cuda()
                y_true = y_true.cuda()

            y_pred = model(src)
            for k, metric in metrics_dict.items():
                curr_batch_metric = metric(y_pred.flatten(0,1), y_true.flatten())
                metric_scores[k] += curr_batch_metric.item()
                del curr_batch_metric

            if type(y_pred) is tuple:
                y_pred = y_pred[0].detach().cpu().numpy()
            else:
                y_pred = y_pred.detach().cpu().numpy()

            all_pred.extend(y_pred)
            all_src.extend(src.detach().cpu().numpy())
            all_true.extend(y_true.detach().cpu().numpy())

            del y_pred
            del src
            del y_true

    for k, v in metric_scores.items():
        metric_scores[k] = v / len(iterator)

    results = {"src": all_src, "pred": all_pred, "true": all_true}

    return results, metric_scores
