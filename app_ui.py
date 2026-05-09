import streamlit as st
import asyncio
import os
from app.orchestrator import sequential_curation_workflow
from app.memory import MemoryManager
from PIL import Image

st.set_page_config(page_title="AI Photo Highlight Extractor", layout="wide")

st.title("📸 AI Photo Highlight Extractor")
st.markdown("### Powered by Google Gemini & InsightFace Multi-Agent System")

memory = MemoryManager()

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    folder_path = st.text_input("Enter Folder Path", value="test_images")
    target_highlights = st.slider("Target Highlight Count", 1, 10, 5)
    
    if st.button("Clear Memory"):
        if os.path.exists("data/photo_memory.db"):
            os.remove("data/photo_memory.db")
            st.success("Memory cleared!")

if st.button("Start AI Curation", type="primary"):
    if not os.path.exists(folder_path):
        st.error(f"Path {folder_path} does not exist!")
    else:
        with st.status("Agents are analyzing photos...", expanded=True) as status:
            st.write("Initializing Vision Agents...")
            # Run the workflow
            highlights = asyncio.run(sequential_curation_workflow(folder_path))
            status.update(label="Curation Complete!", state="complete", expanded=False)
        
        st.header("✨ Curated Highlights")
        
        if not highlights:
            st.warning("No highlights found. Adjust your quality thresholds or check the source folder.")
        else:
            # Display highlights in a grid
            cols = st.columns(3)
            for idx, h in enumerate(highlights):
                with cols[idx % 3]:
                    img = Image.open(h["path"])
                    st.image(img, caption=f"Rank: {h['highlight_rank']}")
                    with st.expander("Why was this selected?"):
                        st.write(h["highlight_explanation"])
                        st.json({
                            "Scene": h["scene_analysis"]["label"],
                            "Technical Quality": h["quality_analysis"]["score"],
                            "Face Count": h["face_count"]
                        })

# Display all processed images and their status from memory
st.divider()
st.header("📊 Full Collection Analysis")

async def show_all():
    await memory.initialize()
    # We'll just read from the DB
    import aiosqlite
    async with aiosqlite.connect(memory.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM images") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

all_images = asyncio.run(show_all())

if all_images:
    st.dataframe(all_images)
else:
    st.info("Start curation to see detailed analysis here.")
