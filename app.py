import torch
from io import BytesIO
import base64
from diffusers import DiffusionPipeline
import PIL.Image
import base64
import io
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import requests
import boto3


class InferlessPythonModel:
    def initialize(self):
        brightness_controlnet = ControlNetModel.from_pretrained(
            "ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16
        )

        tile_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16,
            use_safetensors=False,
        )

        controller = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/DreamShaper",
            controlnet=[brightness_controlnet, tile_controlnet],
            torch_dtype=torch.float16,
            use_safetensors=False,
            safety_checker=None,
        ).to("cuda")

        controller.enable_xformers_memory_efficient_attention()
        self.pipe = controller

    def resize_for_condition_image(self, input_image, resolution: int = 512):
        import PIL.Image

        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=PIL.Image.LANCZOS)
        return img



    def download_image(self, url):
        if "base64," in url:
            url = url.split("base64,")[1]
            image = PIL.Image.open(io.BytesIO(base64.b64decode(url)))
            return image.convert("RGB")
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")
    
    
    def upload_image_to_s3(self, image, bucket_name, file_name, region_name='us-west-2'):
        # Save the image to a bytes buffer in PNG format
        buff = BytesIO()
        image.save(buff, format="PNG")

        # Get the byte data from the buffer
        image_bytes = buff.getvalue()

        # Initialize the S3 client
        s3 = boto3.client('s3', region_name=region_name)

        # Upload the image bytes to S3 with public-read ACL and PNG content type
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=image_bytes, ContentType='image/png', ACL='public-read')

        # Construct the public URL for the uploaded image
        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{file_name}"

        print(f"Image uploaded to {url}")
        return url

    def generate_qr(self, url):
        # TODO: Generate QR code
        pass
    
    def infer(self, inputs):
        input_image_url = inputs["input_image_url"]
        input_image = self.download_image(input_image_url).resize((512, 512))

        tile_input_image = self.resize_for_condition_image(input_image)
        
        prompt = inputs["prompt"]
        image = self.pipe(
            prompt,
            image=[input_image, tile_input_image],
            height=512,
            width=512,
            num_inference_steps=100,
            controlnet_conditioning_scale=[0.45, 0.25],
            guidance_scale=9,
        )["images"][0]
        
        image = PIL.Image.blend(input_image, image, 0.9)
        
        url = self.upload_image_to_s3(image, "qart-public", prompt.replace(" ", "_").strip() + ".png")

        return {"url": url}

    def finalize(self):
        self.pipe = None
    