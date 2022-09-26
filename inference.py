import os
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    model_dir = "textual_inversion_octocat"
    

    prompts = [
        # "GitHub Octocat in the style of <octocat> with a cat head and octopus tentacles, practicing yoga in a seated pose on a flying carpet, psychedelic rainbow background, (((concept art, colorful, realistic proportions, symmetrical, 8K resolution)))",
        # "Woodblock print of a GitHub Octocat in the style of <octocat> with a cat head and octopus tentacles, riding a surfboard on the the Great Wave off Kanagawa, by Katsushika Hokusai",
        # "A painting of a GitHub Octocat in the style of <octocat> by Olivia Bürki",
        # "<octocat>",
        # "A GitHub <octocat> with cat head and octopus body, wearing elegant designer fashion with mayan pattern and native style, aztec street fashion, gapmoe yandere grimdark, trending on pixiv fanbox, painted by greg rutkowski makoto shinkai takashi takeuchi studio ghibli, akihiko yoshida",
        # "A GitHub <octocat> with cat head and octopus body, 2d character design, vector art, digital art, portrait, 4 k, 8 k, sharp focus, smooth, illustration, concept art",
        # "a colorful comic noir illustration of an <octocat> with cat head and octopus body in new orleans by sachin teng,  pastel lighting, cinematic, depth of field, 8 k, high contrast, trending on artstation"
        "highly detailed portrait of an astronaut <octocat> with cat head and octopus body, by Andy Warhol, 4k resolution, nier:automata inspired, bravely default inspired, vibrant but dreary but upflifting red, black and white color scheme!!! ((Space nebula background))",
        "art deco octocat with cat head and octopus body in the style of <octocat>, art deco, vintage, retro label, 1920s"
    ]

    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(model_dir,torch_dtype=torch.float16).to("cuda")

    for prompt in prompts:
        print("\n\nprompt: ", prompt)
        

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
