# Stable Diffusion + Textual Inversion with Codespaces and Replicate

This demo repo illustrates how to run GitHub Codespaces to fine-tune [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion) with your own concept using [textual inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion), and deploy it to Replicate as a cloud API.

🐲 For a live running example, check out [@chenxwh](https://github.com/chenxwh)'s [Spyro the Dragon model](https://replicate.com/cjwbw/sd-textual-inversion-spyro-dragon).

* [Set up your environment](#set-up-your-environment)
* [Fine-tune your concept](#fine-tune-your-concept)
* [Test your trained concept locally](#test-your-trained-concept-locally)
* [Deploy your trained concept on Replicate](#deploy-your-trained-concept-on-replicate)

## Set up your environment

1. Clone this repo and install [Cog](https://github.com/replicate/cog#install) if you haven't already. If you're using GitHub Codespaces, you can install Cog and all the required dependencies by running this command:
    ```
    curl https://replicate.github.io/codespaces/scripts/install-cog.sh | bash
    ```

2.  Run
    ```
    cog run bash
    ```

    to install the dependencies and set up the environment for running `textual_inversion`

3. Run 

    ```
    huggingface-cli login
    ```

    and paste your [HuggingFace token](https://huggingface.co/settings/tokens). You will also need to agree to the terms for accessing [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) on the HuggingFace website.


## Fine-tune your concept

4. Prepare your 3-5 images for your concept (e.g. images in `./dragon`), and modify training parameters in `train` accordingly. You probably need to modify the following:
    ```
    --train_data_dir="dragon" 
    --learnable_property="style" 
    --placeholder_token="<spyro-dragon>"
    --initializer_token="game" 
    --output_dir="textual_inversion_spyro-dragon"
    ```
    and then run
    ```
    bash train
    ```
    The fine-tuning of your concept should be started now, it may take more than one hour to finish.

## Test your trained concept locally

5. Once the concept is trained, you can test it (still in the `cog run bash` env) using this command:
    ```
    python inference.py --model_dir <path_to_model_with_your_trained_concept> --prompt <prompt_with_your_trained_concept> 
    ```
    The output should be saved at `<your_output_path>`

## Deploy your trained concept on Replicate

6. If all works fine, it's time to push your model to Replicate so other people can try your cool concept!

    First, change the `model_id` in [`predict.py`](https://github.com/chenxwh/replicate-sd-textual-inversion/blob/main/predict.py#L15) with your trained concept (same as `output_dir` from `train`). 
    
    Create a new model at [replicate.com/create](https://replicate.com/create). Follow the guide and then you will see the command to log in and push your model to your created page.
    For instance, in a new terminal:

    ```
    cog login
    cog push r8.im/cjwbw/sd-textual-inversion-spyro-dragon
    ```

    You will be prompted to provide your Replicate token after `cog login` to get permission for pushing to the page you created.

    Once your model is pushed, you can run predictions on the website like [this here](https://replicate.com/cjwbw/sd-textual-inversion-spyro-dragon) or use the API:

    ```py
    import replicate
    model = replicate.models.get("cjwbw/sd-textual-inversion-spyro-dragon")
    output = model.predict(prompt="Golden Gate Bridge in style of <spyro-dragon>")
    ```