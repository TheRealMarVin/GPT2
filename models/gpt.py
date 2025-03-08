import torch
import torch.nn as nn

from models.gpt_backbone import GPTBackBone


class GPT(GPTBackBone):
    def __init__(self, config):
        super().__init__(config)

    def _init_extra_modules(self, config):
        embedding_dim = config["model"]["embedding_dim"]
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, self.tokenizer.vocab_size, bias=False)

    def forward(self, x):
        x = super().forward(x)
        x = self.final_norm(x)

        return self.out_layer(x)

    def generate_text(self, contexts, eos_id=None, remove_context=False):
        self.eval()

        encoded_context = self.tokenizer.encode(contexts)
        context_tensor = torch.tensor(encoded_context).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        context_tensor = context_tensor.to(device)

        top_k = self.config["model"]["sampler"].get("top_k", None)
        top_p = self.config["model"]["sampler"].get("top_p", None)

        assert not (top_k and top_p), "Cannot use both top-k and top-p sampling at the same time."

        max_length = self.config["model"]["sampler"].get("max_length", 30) # I have to provide a limit if nothing is there otherwise you may feel some pain
        temperature = self.config["model"]["sampler"].get("temperature", 0.0)

        for _ in range(max_length):
            learnable_context = context_tensor[:, -self.tokenizer.vocab_size:]

            with torch.no_grad():
                logits = self(learnable_context)
            logits = logits[:, -1, :]

            # Apply top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            # Apply top-p (nucleus) filtering
            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above top_p
                mask = cumulative_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False  # Always keep at least one token

                sorted_logits[mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

            # Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if next_token == eos_id:
                break

            context_tensor = torch.cat((context_tensor, next_token), dim=1)

        decoded_text = self.tokenizer.decode(context_tensor.squeeze(0).tolist())
        if remove_context:
            decoded_text = decoded_text[len(contexts):]
        return decoded_text
