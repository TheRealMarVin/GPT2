import torch
import torch.nn as nn

from models.gpt_backbone import GPTBackBone


class GPT(GPTBackBone):
    def __init__(self, config):
        super().__init__(config)

        embedding_dim = config["model"]["embedding_dim"]
        self.top_k = None

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, self.tokenizer.vocab_size, bias=False)

    def forward(self, x):
        x = super().forward(x)
        x = self.final_norm(x)

        return self.out_layer(x)

    def generate_text(self, contexts, max_length, temperature=0.0, eos_id=None, remove_context=False):
        self.eval()

        encoded_context = self.tokenizer.encode(contexts)
        context_tensor = torch.tensor(encoded_context).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        context_tensor = context_tensor.to(device)

        for _ in range(max_length):
            learnable_context = context_tensor[:, -self.tokenizer.vocab_size:]

            with torch.no_grad():
                logits = self(learnable_context)
            logits = logits[:, -1, :]

            if self.top_k is not None:
                top_logits, _ = torch.topk(logits, self.top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

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