import os
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    model_dir = "textual_inversion_octocat"
    prompt = "GitHub Octocat in the style of <octocat> with a cat head and octopus tentacles, practicing yoga in a seated pose on a flying carpet, psychedelic rainbow background, (((concept art, colorful, realistic proportions, symmetrical, 8K resolution)))"
    pipe = StableDiffusionPipeline.from_pretrained(model_dir,torch_dtype=torch.float16).to("cuda")

    # Create output dir
    output_dir = f"output/{datetime.today().isoformat()}"
    os.makedirs(output_dir, exist_ok=True)

    # Write prompt to text file in output dir
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt) 
    
    for i in range(30):
        output_path = f"{output_dir}/{str(i).zfill(3)}.png"
        print(output_path)

        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        image.save(output_path)
