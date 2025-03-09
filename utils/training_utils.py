import torch


def custom_collate_fn(batch, input_padding, output_ignore_index):
    inputs = [item["input_ids"] for item in batch]
    outputs = [item["labels"] for item in batch]

    max_length = max(seq.size(0) for seq in inputs)
    max_length_out = max(seq.size(0) for seq in outputs)

    if max_length != max_length_out:
        raise Exception("sequence length not matching for in put and output")

    padded_inputs = torch.full((len(inputs), max_length), input_padding, dtype=torch.long)
    padded_outputs = torch.full((len(outputs), max_length), output_ignore_index, dtype=torch.long)

    for i, (seq1, seq2) in enumerate(zip(inputs, outputs)):
        padded_inputs[i, -seq1.size(0):] = seq1
        padded_outputs[i, -seq2.size(0):] = seq2

    return {
        "input_ids": padded_inputs,
        "labels": padded_outputs
    }
