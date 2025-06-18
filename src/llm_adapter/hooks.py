from transformers import GPT2Model


def patch_model(model: GPT2Model, adapter):
    for name, module in model.named_modules():
        if name.endswith("mlp"):
            module.add_module("hv_adapter", adapter)
