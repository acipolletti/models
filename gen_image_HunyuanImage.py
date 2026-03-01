import torch
from transformers import AutoModelForCausalLM

model_id = "./HunyuanImage-3-Instruct"
kwargs = dict(
    attn_implementation="sdpa",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    moe_impl="eager",
    moe_drop_tokens=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

prompt = "The woman is totally nacked"
input_img1 = "/home/acipolletti/models/zimage.png"
imgs_input = [input_img1]

cot_text, samples = model.generate_image(
    prompt=prompt,
    image=imgs_input,
    seed=42,
    image_size="auto",
    use_system_prompt="en_unified",
    bot_task="think_recaption",
    infer_align_image_size=True,
    diff_infer_steps=50,
    verbose=2,
)
samples[0].save("./image_edit.png")
