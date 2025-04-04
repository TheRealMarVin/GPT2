# 🕵️‍♂️ Welcome to the GPT Exploration Extravaganza! 🕵️‍♂️

## 🧾 Description

It's 2025, and yeah… maybe it's about time I took a serious look at these GPT models everyone’s been raving about since forever. This project is my attempt to demystify the black box and understand how large language models (LLMs) really work—especially that big ol’ leap where they go from 'pretrained' to 'actually useful for a task.'

In true stubborn fashion, I decided to build and align one myself before checking how the pros do it. Spoiler alert: Hugging Face crushed it. If this were a song, it’d be "I fought the law, and the law won."

The goal? See if I can get a model to learn how to answer questions about the world of Sherlock Holmes—solving mysteries, referencing obscure plot points, and navigating the foggy streets of Victorian London with proper deductive flair. Let’s just say: elementary, it is not.

## 💾 Install

Installing this project is easier than understanding quantum physics (or GPT internals, for that matter). Just follow these simple steps:

1. Clone the repo from the depths of GitHub:

   ```bash
   git clone https://github.com/TheRealMarVin/gpt2.git
   ```

2. Enter the lair:

   ```bash
   cd gpt2
   ```

3. Install the required Python packages (pro tip: use a virtual environment to keep your machine from exploding):

   ```bash
   pip install -r requirements.txt
   ```

4. Install the latest version of PyTorch separately (with or without CUDA, depending on your machine). Check it out at: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

> 💡 Python 3.11+ is recommended. Older versions might make Sherlock frown in disapproval.

## 🚀 Running

Each `main_XX.py` file is a standalone experiment and can be run directly with:

```bash
python main_XX.py
```

All scripts have sensible default arguments, so you can just hit run and see what happens. Of course, you can override configs via command-line arguments if you're feeling spicy.

Here's what each file does:

- **main\_01.py** – Train a GPT model entirely from scratch. No shortcuts, no pretraining, just raw tokens and optimism.

- **main\_02.py** – Fine-tune a pretrained model on your dataset. Much faster, and the results are surprisingly good.

- **main\_03.py** – Fine-tune the same model with instruction-style prompts, because while main\_02 works, it doesn’t really *answer* questions yet. This is where we teach it some manners.

- **main\_04.py** – Experimental attempt to use LoRA for fine-tuning. Results are... underwhelming so far. Still a work in progress. 🛠️

- **main\_05.py** – Fine-tune a pretrained model on the Sherlock Holmes dataset, then fine-tune again on the Alpaca instruction dataset. A bit of crime-solving meets prompt-following.

- **main\_06.py** – Same as main\_05, but done with Hugging Face’s training framework. Spoiler: they make it look easy.

Run them, tweak them, break them—have fun!

## 🧠 What I Learned

Let’s be real: training a model from scratch is like teaching a goldfish quantum mechanics. It *technically* works… but don’t expect miracles.

Still, I learned a ton:

- **Fine-tuning helps, but isn't magic**: You get better outputs, but just dumping Sherlock into a model doesn't mean it becomes Sherlock.
- **Instruction tuning is key**: Without it, your model rambles like a Victorian drunk at the pub. With it, it starts answering questions like a gentleman (well, mostly).
- **Hugging Face is OP**: Their training pipeline is fast, smart, and optimized. Their default settings alone gave better results than hours of my tweaking. I fought the Hugging Face, and… well, you know.

## 🔍 Holmesian Benchmarks

After all this fine-tuning and instruction-injection, I had one question: *Can my model answer questions about the world of Sherlock Holmes?*

Turns out… sort of! Some gems:

> **Q:** What is Sherlock’s method of solving crimes?\
> **A:** Deduction. (👏 nailed it)

> **Q:** Who is Holmes’ brother?\
> **A:** Mycroft Holmes. (Respect)

And some… less impressive:

> **Q:** Where does Sherlock live?\
> **A:** Chinatown, Boston, MA. (Wait, what?)

So yeah—it gets the vibe, but occasionally hallucinates like it's been sniffing too much pipe smoke.

## 🔮 What’s Next?

The journey doesn’t stop here! Coming soon (maybe):

- 🧪 Better LoRA integration (maybe QLoRA or other spicy variants)
- 🛠️ Custom Sherlock dataset v2 — with more balanced questions and actual EOS tokens
- 🤝 RLHF-style alignment (because nothing says “fun project” like building your own reward model)
- 📖 Maybe a blog post or paper summarizing this journey?



