def format_input_for_alpaca(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def format_output_for_alpaca(entry):
    desired_response = f"\n\n### Response:\n{entry['output']}"
    return desired_response


def display_sample(model, instruction, instruction_input):
    entry = {"instruction": instruction, "input": instruction_input, "output": ""}
    start_context = format_input_for_alpaca(entry) + format_output_for_alpaca(entry)

    out = model.generate_text(contexts=start_context, max_length=30, temperature=1.2,
                              eos_id=model.tokenizer.eos_token_id, remove_context=True)

    print("********************************")
    print("input: ", entry)
    print("output: ", out)
