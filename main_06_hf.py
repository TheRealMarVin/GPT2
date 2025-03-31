import argparse

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, \
    pipeline

from utils.download_sherlock_datasets import download_sherlock_dataset


# pip install transformers datasets accelerate

def _train_hugging_face(model_name):
    _fine_tune_sherlock(model_name)
    _fine_tune_instruct()
    _test_on_questions()

def _fine_tune_sherlock(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    data_file = download_sherlock_dataset("data", "sherlock.txt")
    dataset = load_dataset("text", data_files={"train": data_file})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="gpt2-sherlock-lm",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",
        learning_rate=5e-5,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    trainer.save_model("gpt2-sherlock-lm")
    tokenizer.save_pretrained("gpt2-sherlock-lm")

def _fine_tune_instruct():
    model_path = "gpt2-sherlock-lm"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": "data/instruction-data.json"})

    def format_example(ex):
        if ex["input"] and len(ex["input"].strip()) > 0:
            prompt = (
                f"Below is an instruction that describes a task, paired with an input.\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task.\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n"
            )

        full_text = prompt + ex["output"] + "### End"
        return {"text": full_text}

    formatted_dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="gpt2-sherlock-instruct",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",
        learning_rate=5e-5,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model("gpt2-sherlock-instruct")
    tokenizer.save_pretrained("gpt2-sherlock-instruct")

def _test_on_questions():
    model_path = "gpt2-sherlock-instruct"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    with open("test_data/sherlock_questions.txt", "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    def build_prompt(question):
        return (
            "Below is an instruction that describes a task.\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n"
            "### Response:\n"
        )

    results = []
    for i, question in enumerate(questions):
        prompt = build_prompt(question)
        output = generator(
            prompt,
            max_length=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1
        )[0]["generated_text"]

        answer = output.split("### Response:\n")[-1].strip()
        answer = answer.split("### End")[0].strip()
        results.append((question, answer))

    for q, a in results:
        print(f"Q: {q}\nA: {a}\n{'-' * 60}")

    with open("results/hugging_face_test_answers.txt", "w", encoding="utf-8") as out:
        for q, a in results:
            out.write(f"Q: {q}\nA: {a}\n{'-' * 60}\n")

if __name__ == "__main__":
    model_name = "gpt2-medium"

    parser = argparse.ArgumentParser(description="Configuration to launch training the complete model.")
    parser.add_argument("--model", type=str, default=model_name,
                        help="Model name to use for training.")
    args = parser.parse_args()

    _train_hugging_face(args.model)