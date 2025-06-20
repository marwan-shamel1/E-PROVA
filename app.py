import sys
sys.path.append('./')
from PIL import Image
import torch
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import gradio as gr
import base64
from flask_cors import CORS
from pyngrok import ngrok, conf
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline

# Configure ngrok
ngrok.set_auth_token("2wf3BO5M2QXfadRiq2PMQhxWtHY_4D3t4XZRpqsSmxPsMTNpn")

app = Flask(__name__)
CORS(app)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def base64_to_pil(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

# Initialize models
print("Initializing models...")
base_path = 'yisol/IDM-VTON'

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)

noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def start_tryon(human_img, garm_img, garment_des, is_checked=True, is_checked_crop=False, denoise_steps=30, seed=42):
    try:
        if human_img is None or garm_img is None:
            print("Error: Both human image and garment image must be provided")
            return None, None

        openpose_model.preprocessor.body_estimation.model.to(device)
        pipe.to(device)
        pipe.unet_encoder.to(device)

        # Ensure images are in RGB format and properly sized
        try:
            garm_img = garm_img.convert("RGB").resize((768,1024))
            human_img_orig = human_img.convert("RGB")    
        except Exception as e:
            print(f"Error converting images: {str(e)}")
            return None, None
        
        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768,1024))
        else:
            human_img = human_img_orig.resize((768,1024))

        if is_checked:
            try:
                keypoints = openpose_model(human_img.resize((384,512)))
                model_parse, _ = parsing_model(human_img.resize((384,512)))
                mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
                mask = mask.resize((768,1024))
            except Exception as e:
                print(f"Error in mask generation: {str(e)}")
                mask = Image.new('L', (768, 1024), 0)
                mask_gray = Image.new('RGB', (768, 1024), (128, 128, 128))
        else:
            mask = Image.new('L', (768, 1024), 0)
            mask_gray = Image.new('RGB', (768, 1024), (128, 128, 128))

        try:
            mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
            mask_gray = to_pil_image((mask_gray+1.0)/2.0)
        except Exception as e:
            print(f"Error processing mask: {str(e)}")
            mask_gray = Image.new('RGB', (768, 1024), (128, 128, 128))

        try:
            human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

            args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
            pose_img = args.func(args,human_img_arg)    
            pose_img = pose_img[:,:,::-1]    
            pose_img = Image.fromarray(pose_img).resize((768,1024))
        except Exception as e:
            print(f"Error in pose estimation: {str(e)}")
            pose_img = Image.new('RGB', (768, 1024), (128, 128, 128))
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    with torch.inference_mode():
                        # Get embeddings for the main prompt
                        prompt_embeds_result = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                        
                        # Handle both 2-value and 4-value returns
                        if len(prompt_embeds_result) == 4:
                            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_embeds_result
                        else:
                            prompt_embeds, negative_prompt_embeds = prompt_embeds_result
                            pooled_prompt_embeds = None
                            negative_pooled_prompt_embeds = None
                                    
                        # Process garment prompt
                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                        with torch.inference_mode():
                            prompt_embeds_c_result = pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )
                            
                            # Handle both 2-value and 4-value returns for garment prompt
                            if len(prompt_embeds_c_result) == 4:
                                prompt_embeds_c, _, _, _ = prompt_embeds_c_result
                            else:
                                prompt_embeds_c, _ = prompt_embeds_c_result

                        pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                        garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                        seed_int = int(seed) if seed is not None else None
                        generator = torch.Generator(device).manual_seed(seed_int) if seed_int is not None else None

                        denoise_steps = int(denoise_steps)

                        # Print shapes for debugging
                        print(f"prompt_embeds shape: {prompt_embeds.shape}")
                        print(f"negative_prompt_embeds shape: {negative_prompt_embeds.shape}")

                        # Prepare pipeline inputs
                        pipeline_inputs = {
                            'prompt_embeds': prompt_embeds.to(device,torch.float16),
                            'negative_prompt_embeds': negative_prompt_embeds.to(device,torch.float16),
                            'num_inference_steps': denoise_steps,
                            'generator': generator,
                            'strength': 1.0,
                            'pose_img': pose_img.to(device,torch.float16),
                            'text_embeds_cloth': prompt_embeds_c.to(device,torch.float16),
                            'cloth': garm_tensor.to(device,torch.float16),
                            'mask_image': mask,
                            'image': human_img, 
                            'height': 1024,
                            'width': 768,
                            'ip_adapter_image': garm_img.resize((768,1024)),
                            'guidance_scale': 2.0,
                        }

                        # Add pooled embeddings if they exist
                        if pooled_prompt_embeds is not None:
                            pipeline_inputs['pooled_prompt_embeds'] = pooled_prompt_embeds.to(device,torch.float16)
                        if negative_pooled_prompt_embeds is not None:
                            pipeline_inputs['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds.to(device,torch.float16)

                        images = pipe(**pipeline_inputs)[0]

                        if is_checked_crop:
                            out_img = images[0].resize(crop_size)        
                            human_img_orig.paste(out_img, (int(left), int(top)))    
                            return human_img_orig, mask_gray
                        else:
                            return images[0], mask_gray
    except Exception as e:
        print(f"Error in start_tryon: {str(e)}")
        return human_img_orig, mask_gray

# Flask API endpoint
@app.route('/virtual-tryon', methods=['POST'])
def virtual_tryon():
    try:
        data = request.get_json()
        
        # Get base64 images from request
        person_image_base64 = data.get('person_image')
        garment_image_base64 = data.get('garment_image')
        garment_description = data.get('garment_description', 'a t-shirt')
        
        if not person_image_base64 or not garment_image_base64:
            return jsonify({
                "status": "error",
                "message": "Both person_image and garment_image are required"
            }), 400
        
        # Convert base64 to PIL images
        person_img = base64_to_pil(person_image_base64)
        garment_img = base64_to_pil(garment_image_base64)
        
        # Process images
        result_image, mask_image = start_tryon(
            human_img=person_img,
            garm_img=garment_img,
            garment_des=garment_description,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42
        )
        
        # Convert result to base64
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "status": "success",
            "result_image": result_base64
        })
        
    except Exception as e:
        print(f"Error in virtual_tryon: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload Your Photo")
            person_image = gr.Image(type="pil", label='Human. Upload image and use auto-masking', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=False)
        
        with gr.Column():
            gr.Markdown("### Garment")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    garment_des = gr.Textbox(
                        placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts",
                        show_label=False,
                        elem_id="prompt",
                        value="a t-shirt"  # Default value
                    )
            garment_image = gr.Image(label="Fetched Garment Image", type="pil", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Result")
            output_image = gr.Image(label="Output", elem_id="output-img", show_share_button=False)
            masked_image = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
    
    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
    
    def process_and_display(person_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
        try:
            if person_img is None:
                print("Error: No person image provided")
                return None, None, None
            
            # Convert person image to base64
            buffered = BytesIO()
            person_img.save(buffered, format="PNG")
            person_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Get garment image from backend
            backend_url = "https://e-prova.vercel.app/AI/get-garment"
            garment_response = requests.get(backend_url)
            if garment_response.status_code != 200:
                print("Error: Failed to get garment image from backend")
                return None, None, None
                
            garment_data = garment_response.json()
            if 'garment_image' not in garment_data or not garment_data['garment_image']:
                print("Error: No garment image in response")
                return None, None, None
                
            garment_base64 = garment_data['garment_image']
            try:
                garment_img = Image.open(BytesIO(base64.b64decode(garment_base64)))
            except Exception as e:
                print(f"Error decoding garment image: {str(e)}")
                return None, None, None
            
            # Call the Flask API
            response = requests.post(
                'http://localhost:5000/virtual-tryon',
                json={
                    'person_image': person_base64,
                    'garment_image': garment_base64,
                    'garment_description': garment_des if garment_des else 'a t-shirt'
                }
            )
            
            if response.status_code == 200:
                result_data = response.json()
                if result_data['status'] == 'success':
                    result_base64 = result_data['result_image']
                    try:
                        result_img = Image.open(BytesIO(base64.b64decode(result_base64)))
                        return result_img, None, garment_img
                    except Exception as e:
                        print(f"Error decoding result image: {str(e)}")
                        return None, None, garment_img
                else:
                    print("Error:", result_data.get("message"))
                    return None, None, garment_img
            else:
                print("Error:", response.json().get("message"))
                return None, None, garment_img
        except Exception as e:
            print("Error:", str(e))
            return None, None, None
    
    try_button.click(
        fn=process_and_display,
        inputs=[person_image, garment_des, is_checked, is_checked_crop, denoise_steps, seed],
        outputs=[output_image, masked_image, garment_image]
    )

if __name__ == '__main__':
    # Start ngrok
    public_url = ngrok.connect(5000)
    print(f' * Public URL: {public_url}')
    print(f' * Use this URL in your Node.js backend')
    
    # Run Flask app in a separate thread
    import threading
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    flask_thread.start()
    
    # Launch Gradio interface
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
