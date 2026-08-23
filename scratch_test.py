import sys, os
from google import genai
from google.genai import types

def generate():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY")
        return
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents="Hello from Gemini!",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"
                    )
                )
            )
        )
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            print("Found inline data!")
            print(type(part.inline_data.data))
            break
generate()
