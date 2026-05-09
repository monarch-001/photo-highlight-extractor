import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Optional, Any, Dict
import json

load_dotenv()

class VisionAgent:
    def __init__(self, name: str, system_instruction: str, model_id: str = "gemini-2.5-flash"):
        self.name = name
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = model_id
        self.system_instruction = system_instruction

    async def run(self, prompt: str, image_path: Optional[str] = None, history: Optional[List[dict]] = None) -> str:
        contents = []
        
        # Add system instruction as the first part of the message or via config
        # Google AI SDK usually takes system_instruction in GenerateContentConfig
        
        if history:
            for msg in history:
                contents.append(types.Content(role=msg["role"], parts=[types.Part(text=msg["content"])]))
        
        parts = [types.Part(text=prompt)]
        
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = f.read()
            parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data)))
            
        contents.append(types.Content(role="user", parts=parts))
        
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type="application/json" if "JSON" in self.system_instruction else "text/plain"
        )
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config
        )
        
        return response.text

# --- Agent System Prompts ---

ROUTER_PROMPT = """
You are the Router Agent for the AI Photo Highlight Extractor.
Your job is to analyze the user's request and the available photo collection to determine the best workflow.
You can route to Face Analysis, Scene Analysis, Duplicate Detection, Quality Scoring, or Highlight Extraction.
Explain your routing decision.
"""

QUALITY_PROMPT = """
You are the Quality Analysis Agent. 
Analyze the image for technical defects: blur, noise, low lighting, bad framing, or overexposure.
Assign a quality score from 0 to 100.
If the score is below 40, provide a rejection reason.
Return results in JSON format: {"score": int, "rejected": bool, "reason": string}
"""

FACE_PROMPT = """
You are the Face & Emotion Analysis Agent.
Detect faces, evaluate expressions (smiles, joy, focus), and identify if eyes are closed.
Identify emotionally strong moments or best expressions.
Return results in JSON format: {"faces_detected": int, "emotion_score": int, "summary": string, "is_emotionally_strong": bool}
"""

SCENE_PROMPT = """
You are the Scene Classification Agent.
Identify the scene type: landscape, food, group photo, selfie, indoor, outdoor, event, sports, travel.
Assign a scene label and confidence.
Return results in JSON format: {"label": string, "confidence": float, "description": string}
"""

HIGHLIGHT_PROMPT = """
You are the Highlight Selection Agent.
Rank photos based on quality, facial clarity, emotional strength, and scene uniqueness.
Balance the collection for diversity (avoid selecting too many of the same scene).
Explain WHY a photo was selected as a highlight.
Return results in JSON format: {"rank": int, "is_highlight": bool, "explanation": string}
"""

CAPTION_PROMPT = """
You are the Caption Generation Agent.
Create descriptive and engaging captions for highlight photos.
Summarize the key moment or aesthetic value.
"""

# Instantiate Agents
router_agent = VisionAgent("Router", ROUTER_PROMPT)
quality_agent = VisionAgent("Quality", QUALITY_PROMPT)
face_agent = VisionAgent("Face", FACE_PROMPT)
scene_agent = VisionAgent("Scene", SCENE_PROMPT)
highlight_agent = VisionAgent("Highlight", HIGHLIGHT_PROMPT)
caption_agent = VisionAgent("Caption", CAPTION_PROMPT)
