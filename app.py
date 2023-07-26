import torch
from io import BytesIO
import base64
from diffusers import DiffusionPipeline


class InferlessPythonModel:
    def initialize(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "Lykon/DreamShaper", 
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda:0")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue())
        return {"generated_image_base64": img_str.decode("utf-8")}

    def finalize(self):
        self.pipe = None