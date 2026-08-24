import json
import logging
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict

from .base import MusicGenerationProvider

logger = logging.getLogger(__name__)

# We won't provide a complex default for audio, it's highly specific
DEFAULT_WORKFLOW: Dict[str, Any] = {}


class ComfyUIMusicGenerationProvider(MusicGenerationProvider):
    """Music generation using a local ComfyUI instance."""

    def __init__(self, config: Dict[str, Any]):
        self.url = config.get("url", "http://127.0.0.1:8188")
        self.url = self.url.rstrip("/")
        self.workflow_file = config.get("workflow_file")

    def generate(self, prompt: str, lyrics: str = "") -> str:
        logger.info("Generating music with ComfyUI at %s", self.url)

        if not self.workflow_file:
            return "Error: No workflow_file configured for ComfyUI music generation."

        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                workflow_str = f.read()

            escaped_prompt = json.dumps(prompt)[1:-1]
            workflow_str = workflow_str.replace("__PROMPT__", escaped_prompt)

            if lyrics:
                escaped_lyrics = json.dumps(lyrics)[1:-1]
                workflow_str = workflow_str.replace("__LYRICS__", escaped_lyrics)
            else:
                workflow_str = workflow_str.replace("__LYRICS__", "")

            workflow = json.loads(workflow_str)
        except Exception as e:
            logger.error("Failed to load audio workflow file: %s", e)
            return (
                f"Error loading ComfyUI audio workflow from {self.workflow_file}: {e}"
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
            logger.error("Failed to submit audio prompt to ComfyUI: %s", e)
            return f"Error connecting to ComfyUI at {self.url}: {e}"

        if not prompt_id:
            return "Failed to get prompt_id from ComfyUI."

        logger.info("ComfyUI audio prompt submitted. ID: %s", prompt_id)

        # Poll history for completion
        # Audio generation can take a bit, wait up to 3 minutes
        max_retries = 180
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
            return f"ComfyUI audio generation timed out after {max_retries} seconds."

        # Extract audio URL
        outputs = history_data.get("outputs", {})
        audio_filename = None
        folder_type = "temp"
        subfolder = ""

        for _, output in outputs.items():
            # Check for audio
            if "audio" in output and len(output["audio"]) > 0:
                audio_info = output["audio"][0]
                audio_filename = audio_info.get("filename")
                subfolder = audio_info.get("subfolder", "")
                folder_type = audio_info.get("type", "temp")
                break
            # Some nodes might save audio under 'images' mistakenly or as a generic file
            elif "images" in output and len(output["images"]) > 0:
                # Check extension just in case
                filename = output["images"][0].get("filename", "")
                if (
                    filename.endswith(".mp3")
                    or filename.endswith(".wav")
                    or filename.endswith(".flac")
                ):
                    audio_info = output["images"][0]
                    audio_filename = audio_info.get("filename")
                    subfolder = audio_info.get("subfolder", "")
                    folder_type = audio_info.get("type", "temp")
                    break

        if audio_filename:
            audio_url = (
                f"{self.url}/view?filename={urllib.parse.quote(audio_filename)}"
                f"&type={folder_type}"
                f"&subfolder={urllib.parse.quote(subfolder)}"
            )
            result_str = f"[Generated Audio]({audio_url})"
            if lyrics:
                result_str += f"\n\nLyrics:\n{lyrics}"
            return result_str

        return (
            "ComfyUI audio generation completed, "
            "but no audio file was returned in history."
        )
