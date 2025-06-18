import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_adapter.adapter import HypervectorAdapter
from llm_adapter.hooks import patch_model


def main():
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits_before = model(**inputs).logits

    adapter = HypervectorAdapter(hidden_size=model.config.n_embd, r=4)
    patch_model(model, adapter)

    with torch.no_grad():
        logits_after = model(**inputs).logits

    attached = all(hasattr(block.mlp, "hv_adapter") for block in model.transformer.h)
    diff = (logits_after - logits_before).abs().mean().item()
    print("Adapters attached to all blocks:", attached)
    print(f"Mean absolute diff after patch: {diff:.4f}")


if __name__ == "__main__":
    main()
