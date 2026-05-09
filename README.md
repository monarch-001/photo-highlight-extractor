# AI Photo Highlight Extractor

A high-performance multi-agent AI system for intelligent photo curation and highlight extraction. This project leverages the **Google AI SDK (Gemini 2.0)** and **InsightFace** to simulate an expert photo editor's reasoning process.

## 🚀 Key Agentic Features

This project demonstrates all mandatory Google ADK orchestration patterns:

1.  **Router Agent:** Dynamically determines the workflow based on the collection size and content.
2.  **Parallel Workflow:** Concurrently executes Technical Quality, Face/Emotion, and Scene Analysis using `asyncio.gather`.
3.  **Sequential Workflow:** Manages the pipeline from raw ingestion to duplicate removal and final curation.
4.  **Loop/Refinement Workflow:** The Highlight Agent iteratively selects and swaps candidates to maximize both quality and diversity in the final album.
5.  **Agent-as-Tool:** The Highlight Agent "consults" the Technical Quality and Face agents as specialized tools during its decision-making loop.
6.  **Custom Tools:** Robust integration with OpenCV (blur/exposure) and InsightFace (facial clarity/attributes).
7.  **Agent Memory:** Persistent storage of image scores, metadata, and selection history using **SQLite**.

## 🛠 Tech Stack

-   **AI:** Gemini 2.0 Flash (Multimodal)
-   **Vision:** InsightFace, OpenCV, Pillow
-   **Orchestration:** Python `asyncio`
-   **Frontend:** Streamlit
-   **Storage:** SQLite (`aiosqlite`)

## ⚙️ Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment:**
    Create a `.env` file:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

3.  **Prepare Photos:**
    Place the photos you want to analyze in the `test_images` folder or provide a custom path in the UI.

4.  **Run the Application:**
    ```bash
    streamlit run app_ui.py
    ```

## 🧠 System Architecture

-   **Phase 1 (Ingestion):** Files are loaded and deterministic metrics (Blur, Exposure, Hash) are calculated.
-   **Phase 2 (Parallel Analysis):** Face, Scene, and Quality agents analyze images concurrently using Gemini's visual reasoning.
-   **Phase 3 (Filtering):** Duplicate Detection Agent removes near-identical bursts.
-   **Phase 4 (Curation):** Highlight Agent runs a refinement loop to select the top $N$ photos with diverse content and high aesthetic value.
-   **Phase 5 (Presentation):** Final results are displayed with explanations of "WHY" each photo was selected.
