import asyncio
import os
import json
from typing import List, Dict, Any
from .agents import quality_agent, face_agent, scene_agent, highlight_agent, caption_agent
from .vision_tools import get_blur_score, get_exposure_score, analyze_faces, get_image_hash
from .memory import MemoryManager

memory = MemoryManager()

async def analyze_image_parallel(image_path: str) -> Dict[str, Any]:
    """
    Runs Face, Scene, and Quality analysis in parallel for a single image.
    This demonstrates the Parallel Workflow concept.
    """
    # Deterministic tools (running locally)
    blur = get_blur_score(image_path)
    exposure = get_exposure_score(image_path)
    faces_info = analyze_faces(image_path)
    
    # Semantic agents (running via Gemini API)
    # We pass the image path to Gemini for visual reasoning
    tasks = [
        quality_agent.run(f"Blur score: {blur}, Exposure: {exposure}. Analyze technical quality.", image_path),
        face_agent.run(f"InsightFace detected {faces_info['count']} faces. Analyze expressions.", image_path),
        scene_agent.run("Classify this scene.", image_path)
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "path": image_path,
        "filename": os.path.basename(image_path),
        "blur_score": blur,
        "exposure_score": exposure,
        "quality_analysis": json.loads(results[0]),
        "face_analysis": json.loads(results[1]),
        "scene_analysis": json.loads(results[2]),
        "face_count": faces_info['count']
    }

async def sequential_curation_workflow(folder_path: str) -> List[Dict[str, Any]]:
    """
    Executes the full curation pipeline sequentially.
    Demonstrates Sequential Workflow.
    """
    await memory.initialize()
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    all_analysis = []
    
    # 1. Parallel Analysis for each image
    for img_path in image_files:
        print(f"Analyzing {img_path}...")
        analysis = await analyze_image_parallel(img_path)
        all_analysis.append(analysis)
        
        # Save to memory
        await memory.save_image_metadata({
            "path": analysis["path"],
            "filename": analysis["filename"],
            "blur_score": analysis["blur_score"],
            "exposure_score": analysis["exposure_score"],
            "face_count": analysis["face_count"],
            "scene_label": analysis["scene_analysis"]["label"],
            "explanation": analysis["quality_analysis"].get("reason", "")
        })

    # 2. Duplicate Detection (Sequential logic)
    hashes = {}
    for analysis in all_analysis:
        h = get_image_hash(analysis["path"])
        if h in hashes:
            # Duplicate found, mark in memory
            # We keep the one with higher blur score (sharper)
            existing = hashes[h]
            if analysis["blur_score"] > existing["blur_score"]:
                # Mark existing as duplicate
                await memory.save_image_metadata({"path": existing["path"], "is_duplicate": 1})
                hashes[h] = analysis
            else:
                await memory.save_image_metadata({"path": analysis["path"], "is_duplicate": 1})
        else:
            hashes[h] = analysis

    # 3. Highlight Selection (Loop Refinement)
    highlights = await loop_refinement_workflow(all_analysis)
    
    return highlights

async def loop_refinement_workflow(analyses: List[Dict[str, Any]], target_count: int = 5) -> List[Dict[str, Any]]:
    """
    Iteratively refines the highlight selection to ensure diversity and quality.
    Demonstrates Loop Workflow.
    """
    print("Refining highlights...")
    selected_paths = []
    iterations = 0
    max_iterations = 3
    
    current_highlights = []
    
    while len(current_highlights) < target_count and iterations < max_iterations:
        # Ask Highlight Agent to pick the best from the remaining
        remaining = [a for a in analyses if a["path"] not in selected_paths]
        
        # In a real loop, we might pass the current selection to the agent to avoid repetition
        # For simplicity, we'll pick top scoring ones that aren't duplicates
        # But we use the Highlight Agent to "Reason"
        
        for a in remaining:
            if len(current_highlights) >= target_count: break
            
            # Agent-as-Tool: Highlight Agent "consults" Quality Agent results (already in analysis)
            prompt = f"Technical Quality: {a['quality_analysis']}. Scene: {a['scene_analysis']}. Should this be a highlight?"
            res = await highlight_agent.run(prompt, a["path"])
            selection = json.loads(res)
            
            if selection["is_highlight"]:
                a["highlight_rank"] = selection["rank"]
                a["highlight_explanation"] = selection["explanation"]
                current_highlights.append(a)
                selected_paths.append(a["path"])
                
                # Update memory
                await memory.save_image_metadata({
                    "path": a["path"],
                    "is_highlight": 1,
                    "highlight_score": float(selection["rank"]),
                    "explanation": selection["explanation"]
                })
        
        iterations += 1
        
    return current_highlights
