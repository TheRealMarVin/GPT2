import numpy as np
import torch
from tqdm import tqdm


def reshape_prediction_for_compatibility(raw_output):
    reshaped_res = []
    for x in range(raw_output[0].shape[0]):
        curr_out = []
        for y in raw_output:
            curr_out.append(y[x])

        reshaped_res.append(curr_out)

    return np.array(reshaped_res)


def default_evaluate(iterator, model, metrics_dict, global_metrics_dict={}, true_index=1):
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    metric_scores = {}
    for k, _ in metrics_dict.items():
        metric_scores[k] = 0

    for k, _ in global_metrics_dict.items():
        metric_scores[k] = 0

    with torch.no_grad():
        all_pred = []
        all_true = []
        all_src = []

        for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="eval"):
            src = batch[0]
            y_true = batch[true_index]

            if len(y_true.shape) == 1:
                y_true = y_true.type("torch.LongTensor")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                src = src.cuda()
                y_true = y_true.cuda()

            y_pred = model(src)
            for k, metric in metrics_dict.items():
                curr_epoch_metric = metric(y_pred, y_true)
                if type(curr_epoch_metric) is torch.Tensor:
                    metric_scores[k] += curr_epoch_metric.item()
                else:
                    metric_scores[k] += curr_epoch_metric
                del curr_epoch_metric

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

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    for k, metric in global_metrics_dict.items():
        curr_epoch_metric = metric(all_pred, all_true)
        if type(curr_epoch_metric) is torch.Tensor:
            metric_scores[k] += curr_epoch_metric.item()
        else:
            metric_scores[k] += curr_epoch_metric
        del curr_epoch_metric

    results = {"src": all_src, "pred": all_pred, "true": all_true}

    return results, metric_scores


def convert_string(tokenizer, device, field, text):
    tokenized_text = tokenizer(text)
    res = torch.LongTensor([field.vocab.stoi[x] for x in tokenized_text]).to(device)
    res = res.unsqueeze(-1)
    return res
