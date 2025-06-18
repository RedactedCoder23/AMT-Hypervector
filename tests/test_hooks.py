from transformers import AutoModelForCausalLM
from llm_adapter.adapter import HypervectorAdapter
from llm_adapter.hooks import patch_model


def test_patch_model_attaches_adapter():
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    adapter = HypervectorAdapter(hidden_size=model.config.n_embd, r=2)
    patch_model(model, adapter)
    for block in model.transformer.h:
        assert hasattr(block.mlp, "hv_adapter")
