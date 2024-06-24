from starlette.requests import Request
from typing import Dict

from ray import serve
from transformers import pipeline
from ray.serve import Application


# sample model "translation_en_to_fr"
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class GenericModelServer:
    def __init__(self, modelName: str):
        # Load model
        print(f"Initializing model with message: {modelName}")
        self.model = pipeline(modelName, model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


def app_builder(args: Dict[str, str]) -> Application:
    return GenericModelServer.bind(args["modelName"])
