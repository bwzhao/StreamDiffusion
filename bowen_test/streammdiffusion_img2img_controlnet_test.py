import dearpygui.dearpygui as dpg
import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderTiny, ControlNetModel, StableDiffusionPipeline
from diffusers.utils import load_image
from src.streamdiffusion import StreamDiffusionControlNetPipeline
from src.streamdiffusion.image_utils import postprocess_image


def generate_image(stream, input_image, control_image, texture_tag):
    stream.update_init_noise()
    x_output = stream(input_image, control_image)
    # x_output = stream.txt2img()

    image = postprocess_image(x_output, output_type="pil")[0]
    image.putalpha(255)
    dpg.set_value(texture_tag, np.asarray(image).flatten() / 255.0)


def main():
    height = 512
    width = 512
    texture_tag = "display_image"
    input_tag = "input_image"
    control_tag = "control_image"

    text_prompt = "1 girl with dog hair, thick frame glasses"

    empty_data = []
    for i in range(0, height * width):
        empty_data.append(255 / 255)
        empty_data.append(255 / 255)
        empty_data.append(255 / 255)
        empty_data.append(255 / 255)

    # You can load any models using diffuser's StableDiffusionPipeline
    base_model = "Lykon/dreamshaper-7"
    pipe = StableDiffusionPipeline.from_pretrained(base_model).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device=pipe.device, dtype=pipe.dtype)
    # Wrap the pipeline in StreamDiffusion
    stream = StreamDiffusionControlNetPipeline(
        pipe,
        controlnet=controlnet,
        t_index_list=[10, 20, 30, 40],
        # t_index_list=[],
        torch_dtype=torch.float16,
        use_denoising_batch=False,
    )

    # If the loaded model is not LCM, merge LCM
    stream.load_lcm_lora()
    stream.fuse_lora()
    # Use Tiny VAE for further acceleration
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )
    # Enable acceleration
    pipe.enable_xformers_memory_efficient_attention()

    prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
    # Prepare the stream
    stream.prepare(prompt, num_inference_steps=50, seed=-1)

    # Prepare image
    path_input_image = "./images/inputs/img2img-init.png"
    path_control_image = "./images/inputs/images_control.png"

    init_image = load_image(path_input_image).resize((512, 512))
    control_image = load_image(path_control_image).resize((512, 512))

    # Warmup >= len(t_index_list) x frame_buffer_size
    for _ in range(4):
        stream(init_image, control_image)

    dpg.create_context()
    dpg.create_viewport(
        title="StreamDiffusion img2img Test", width=3 * width, height=700
    )
    dpg.setup_dearpygui()

    with dpg.value_registry():
        dpg.add_string_value(
            default_value=f"Model: {base_model} (SD1.5) + LCM-LoRA\n"
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
        _input_width, _input_height, channels, data = dpg.load_image(path_control_image)
        dpg.add_static_texture(_input_width, _input_height, data, tag=control_tag)

    with dpg.window(
        label=f"Output Image",
        width=width,
        height=height,
        pos=[2 * width, 0],
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
        label=f"Control Image",
        width=width,
        height=height,
        pos=[width, 0],
    ):
        with dpg.drawlist(width=width, height=height):
            dpg.draw_image(
                control_tag,
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
        width=3 * width,
        height=700 - 512,
        pos=[0, height],
    ):
        dpg.add_text(label="Prompt", source="prompt")
        dpg.add_text(label="FPS", source="fps", color=(255, 0, 0, 255))

    import time

    input_image = init_image
    while dpg.is_dearpygui_running():
        last_time = time.time()
        generate_image(stream, input_image, control_image, texture_tag)
        generate_time = time.time() - last_time
        dpg.set_value("fps", f"Generation Time: {generate_time:.2f}s")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
