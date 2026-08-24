import json
import logging
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict

from .base import ImageGenerationProvider

logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW = {
    "3": {
        "inputs": {
            "seed": 12345,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
        "class_type": "KSampler",
    },
    "4": {
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"},
        "class_type": "CheckpointLoaderSimple",
    },
    "5": {
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "class_type": "EmptyLatentImage",
    },
    "6": {
        "inputs": {"text": "prompt", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
    },
    "7": {
        "inputs": {"text": "watermark, text, ugly, bad quality", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
    },
    "8": {
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        "class_type": "VAEDecode",
    },
    "9": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
        "class_type": "SaveImage",
    },
}


class ComfyUIImageGenerationProvider(ImageGenerationProvider):
    """Image generation using a local ComfyUI instance."""

    def __init__(self, config: Dict[str, Any]):
        self.url = config.get("url", "http://127.0.0.1:8188")
        # Ensure url does not end with a slash
        self.url = self.url.rstrip("/")
        self.workflow_file = config.get("workflow_file")

    def generate(self, prompt: str) -> str:
        logger.info("Generating image with ComfyUI at %s", self.url)

        if self.workflow_file:
            try:
                with open(self.workflow_file, "r", encoding="utf-8") as f:
                    workflow_str = f.read()

                # Replace the placeholder with the actual prompt
                # Note: json.dumps(prompt)[1:-1] escapes the prompt for JSON
                # and removes the surrounding quotes
                escaped_prompt = json.dumps(prompt)[1:-1]
                workflow_str = workflow_str.replace("__PROMPT__", escaped_prompt)

                workflow = json.loads(workflow_str)
            except Exception as e:
                logger.error("Failed to load workflow file: %s", e)
                return f"Error loading ComfyUI workflow from {self.workflow_file}: {e}"
        else:
            workflow = json.loads(json.dumps(DEFAULT_WORKFLOW))
            # Find the positive prompt node for the default workflow
            found_prompt_node = False
            for _, node in workflow.items():
                if node.get("class_type") == "CLIPTextEncode":
                    inputs = node.get("inputs", {})
                    if "text" in inputs:
                        if not found_prompt_node:
                            inputs["text"] = prompt
                            found_prompt_node = True

            if not found_prompt_node:
                logger.warning(
                    "Could not definitively find the positive prompt "
                    "node in the workflow."
                )

        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        data = json.dumps(payload).encode("utf-8")

        try:
            req = urllib.request.Request(f"{self.url}/prompt", data=data)
            with urllib.request.urlopen(req) as response:  # nosec B310
                resp_data = json.loads(response.read())
                prompt_id = resp_data.get("prompt_id")
        except Exception as e:
            logger.error("Failed to submit prompt to ComfyUI: %s", e)
            return f"Error connecting to ComfyUI at {self.url}: {e}"

        if not prompt_id:
            return "Failed to get prompt_id from ComfyUI."

        logger.info("ComfyUI prompt submitted. ID: %s", prompt_id)

        # Poll history for completion
        # We will poll for up to 2 minutes
        max_retries = 120
        history_data = {}
        for _ in range(max_retries):
            try:
                req = urllib.request.Request(f"{self.url}/history/{prompt_id}")
                with urllib.request.urlopen(req) as response:  # nosec B310
                    history = json.loads(response.read())
                    if prompt_id in history:
                        history_data = history[prompt_id]
                        break
            except Exception as e:
                logger.error("Error polling history: %s", e)
            time.sleep(1)
        else:
            return f"ComfyUI generation timed out after {max_retries} seconds."

        # Extract image URL
        outputs = history_data.get("outputs", {})
        image_filename = None
        folder_type = "temp"
        subfolder = ""

        for _, output in outputs.items():
            if "images" in output and len(output["images"]) > 0:
                image_info = output["images"][0]
                image_filename = image_info.get("filename")
                subfolder = image_info.get("subfolder", "")
                folder_type = image_info.get("type", "temp")
                break

        if image_filename:
            img_url = (
                f"{self.url}/view?filename={urllib.parse.quote(image_filename)}"
                f"&type={folder_type}"
                f"&subfolder={urllib.parse.quote(subfolder)}"
            )
            return f"![Generated Image]({img_url})"

        return "ComfyUI generation completed, but no image was returned."
