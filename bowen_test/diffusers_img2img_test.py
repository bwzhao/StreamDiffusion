import dearpygui.dearpygui as dpg
import numpy as np
import PIL.Image
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler, StableDiffusionPipeline
from diffusers.utils import load_image, make_image_grid
from src.streamdiffusion import StreamDiffusion
from src.streamdiffusion.image_utils import postprocess_image


def generate_image(pipe, prompt, init_image, texture_tag):
    # stream.update_init_noise()
    # x_output = stream(input_image)
    generator = torch.manual_seed(-1)
    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=4,
        guidance_scale=1,
        strength=0.6,
        generator=generator,
    ).images[0]

    image.putalpha(255)
    dpg.set_value(texture_tag, np.asarray(image).flatten() / 255.0)


def main():
    height = 512
    width = 512
    texture_tag = "display_image"
    input_tag = "input_image"

    empty_data = []
    for i in range(0, height * width):
        empty_data.append(255 / 255)
        empty_data.append(255 / 255)
        empty_data.append(255 / 255)
        empty_data.append(255 / 255)

    # You can load any models using diffuser's StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1",
        safety_checker=None,
    ).to(device=torch.device("cuda"), dtype=torch.float16)

    pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

    pipe.enable_model_cpu_offload()

    prompt = "1 girl with dog hair, thick frame glasses"

    # Prepare image
    path_input_image = "./images/inputs/input.png"
    init_image = load_image(path_input_image).resize((512, 512))

    dpg.create_context()
    dpg.create_viewport(title="Diffusers img2img Test", width=2 * width, height=700)
    dpg.setup_dearpygui()

    with dpg.value_registry():
        dpg.add_string_value(
            default_value=f"Model: KBlueLeaf/kohaku-v2.1 (SD1.5) + LCM-LoRA\n"
            + "GPU: RTX3090\n"
            + f"Prompt: {prompt} \n"
            + f"Resolution: {width}*{height}",
            tag="prompt",
        )
        dpg.add_string_value(default_value=f"", tag="fps")

    dpg.set_global_font_scale(2)

    with dpg.texture_registry():
        dpg.add_dynamic_texture(width, height, empty_data, tag=texture_tag)
        _input_width, _input_height, channels, data = dpg.load_image(path_input_image)
        dpg.add_static_texture(_input_width, _input_height, data, tag=input_tag)

    with dpg.window(
        label=f"Output Image",
        width=width,
        height=height,
        pos=[width, 0],
    ):
        with dpg.drawlist(width=width, height=height):
            dpg.draw_image(
                f"display_image",
                (0, 0),
                (width, height),
                uv_min=(0, 0),
                uv_max=(1, 1),
            )
    with dpg.window(
        label=f"Input Image",
        width=width,
        height=height,
        pos=[0, 0],
    ):
        with dpg.drawlist(width=width, height=height):
            dpg.draw_image(
                input_tag,
                (0, 0),
                (width, height),
                uv_min=(0, 0),
                uv_max=(1, 1),
            )
    dpg.show_viewport()

    with dpg.window(
        label=f"Controller",
        width=2 * width,
        height=700 - 512,
        pos=[0, height],
    ):
        dpg.add_text(label="Prompt", source="prompt")
        dpg.add_text(label="FPS", source="fps", color=(255, 0, 0, 255))

    import time

    input_image = init_image
    while dpg.is_dearpygui_running():
        last_time = time.time()
        generate_image(pipe, prompt, input_image, texture_tag)
        generate_time = time.time() - last_time
        dpg.set_value("fps", f"Generation Time: {generate_time:.2f}s")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
