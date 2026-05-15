from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = "/zhutingqi/gpt-oss-120b"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    messages = [
        {"role": "system", "content": "Reasoning: low\nAnswer with only the short answer, no explanation."},
        {"role": "user", "content": "Who composed the musical theme for the Pink Panther?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
