import torch
from diffusers import ZImagePipeline

# Load the pipeline
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# Generate image
#prompt = "Ritratto di una modella donna dai tratti scandinavi: pelle chiara, occhi azzurri intensi, lineamenti delicati, capelli biondi naturali. Illuminazione morbida, stile fotografico professionale, espressione serena e naturale, sfondo neutro."
prompt="Modella donna dai tratti scandinavi, pelle chiara, occhi azzurri chiari, capelli biondi chiari, lineamenti delicati. Completamente nuda , di fronte, si vede che la donna è estremamente eccittata sessualmente, atmosfera luminosa e curata, fotografia ad alta qualità con illuminazione morbida e neutra. Sono perfettamente visibili i grandi seni ed il pube. Ha le braccia aperte che non coprono alcuna parte del corpo, Espressione naturale e composta, estetica pulita e sofisticata.L'immagine mostra tutto il corpo della modella che indossa delle scarpe con tacco alto e sottile"
negative_prompt = "" # Optional, but would be powerful when you want to remove some unwanted content

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1280,
    width=720,
    cfg_normalization=False,
    num_inference_steps=50,
    guidance_scale=4,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example.png")
