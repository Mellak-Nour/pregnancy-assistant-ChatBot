"""
=============================================================================
PREGNANCY HEALTH RAG ASSISTANT
=============================================================================
A Retrieval-Augmented Generation (RAG) system built with:
  - Gemini API (google-generativeai) for language generation
  - FAISS for vector similarity search
  - SentenceTransformers for text embeddings
  - CSV-based medical knowledge datasets

SYSTEM ARCHITECTURE:
  User Question
       ↓
  Load User Context (pregnancy week, illnesses, allergies)
       ↓
  Danger Detection (check for emergency symptoms)
       ↓
  Retrieve relevant docs from FAISS vector database
       ↓
  Build structured prompt (persona + context + question)
       ↓
  Send to Gemini API
       ↓
  Return structured guidance response

DOCUMENTATION:
  Vector Database Choice (FAISS):
    FAISS (Facebook AI Similarity Search) was chosen because:
    - It is lightweight and runs entirely in-memory or locally
    - No server setup required — ideal for single-file deployment
    - Extremely fast cosine/L2 similarity search even on large datasets
    - Well-supported Python bindings via faiss-cpu
    - Perfect for small-to-medium medical knowledge bases

  Retrieval Logic (Top-K Search):
    1. The user question is embedded into a 384-dim vector using
       SentenceTransformers (all-MiniLM-L6-v2)
    2. FAISS performs an exact L2 nearest-neighbor search
    3. Top-5 most semantically similar document chunks are retrieved
    4. These chunks form the context injected into the Gemini prompt

  Prompting Strategy:
    - Role/persona prompt establishes the assistant as a safe medical guide
    - User profile (week, trimester, allergies, illnesses) personalizes answers
    - Retrieved documents provide grounded, factual medical context
    - Danger detection pre-checks trigger emergency guidance before LLM call
    - Output format is structured: Answer / Safety Advice / Recommendation

  Dataset:
    - foods.csv: 40+ food items with safety ratings, trimester applicability, notes
    - exercise.csv: 30 exercises with trimester safety and modification notes
    - symptoms.csv: 45 symptoms with risk levels (low/medium/high) and actions
    - weekly_guidance.csv: Week-by-week baby development, nutrition, exercise tips
    All rows are converted to natural language text chunks for embedding.
=============================================================================

INSTALLATION:
  pip install google-generativeai sentence-transformers faiss-cpu numpy pandas

USAGE:
  1. Set your Gemini API key in the GEMINI_API_KEY variable below
  2. Ensure the /data/ folder contains the four CSV files
  3. Run: python pregnancy_rag_assistant.py

=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️  IMPORTANT: Set your Gemini API key here or via environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCpmgqTLJ9Fzm2N5L3DowkK4jPS_FJaX_U")

# Model settings
GEMINI_MODEL = "gemini-3-flash-preview"           # Fast and cost-effective Gemini model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # 384-dim sentence embeddings, runs locally
TOP_K_RESULTS = 5                            # Number of documents to retrieve per query

# Data paths (relative to this script)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FOODS_CSV = os.path.join(DATA_DIR, "foods.csv")
EXERCISE_CSV = os.path.join(DATA_DIR, "exercise.csv")
SYMPTOMS_CSV = os.path.join(DATA_DIR, "symptoms.csv")
WEEKLY_CSV = os.path.join(DATA_DIR, "weekly_guidance.csv")

# FAISS index save path (persists between runs for faster startup)
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "pregnancy_knowledge.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "knowledge_chunks.json")

# Danger keywords that trigger emergency guidance
DANGER_KEYWORDS = [
    "bleeding", "heavy bleeding", "vaginal bleeding", "spotting",
    "severe pain", "severe abdominal pain", "sharp pain",
    "severe headache", "vision changes", "blurred vision", "flashing lights",
    "no fetal movement", "baby not moving", "decreased movement",
    "water broke", "leaking fluid", "gush of fluid",
    "preterm contractions", "contractions before 37 weeks",
    "high fever", "fever above 38", "difficulty breathing",
    "chest pain", "sudden swelling", "face swelling",
    "unconscious", "fainting", "seizure",
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets():
    """
    Load all four CSV knowledge base files into DataFrames.
    Returns a dict of DataFrames keyed by dataset name.
    """
    datasets = {}

    try:
        datasets["foods"] = pd.read_csv(FOODS_CSV)
        print(f"  ✓ Loaded foods.csv ({len(datasets['foods'])} rows)")
    except FileNotFoundError:
        print(f"  ✗ Warning: {FOODS_CSV} not found. Skipping.")

    try:
        datasets["exercise"] = pd.read_csv(EXERCISE_CSV)
        print(f"  ✓ Loaded exercise.csv ({len(datasets['exercise'])} rows)")
    except FileNotFoundError:
        print(f"  ✗ Warning: {EXERCISE_CSV} not found. Skipping.")

    try:
        datasets["symptoms"] = pd.read_csv(SYMPTOMS_CSV)
        print(f"  ✓ Loaded symptoms.csv ({len(datasets['symptoms'])} rows)")
    except FileNotFoundError:
        print(f"  ✗ Warning: {SYMPTOMS_CSV} not found. Skipping.")

    try:
        datasets["weekly"] = pd.read_csv(WEEKLY_CSV)
        print(f"  ✓ Loaded weekly_guidance.csv ({len(datasets['weekly'])} rows)")
    except FileNotFoundError:
        print(f"  ✗ Warning: {WEEKLY_CSV} not found. Skipping.")

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEXT CHUNK CREATION
# ─────────────────────────────────────────────────────────────────────────────

def create_text_chunks(datasets: dict) -> list[str]:
    """
    Convert structured CSV rows into descriptive natural language text chunks.
    Each chunk represents one piece of medical knowledge for embedding.

    Returns a list of text strings ready for vectorisation.
    """
    chunks = []

    # ── Foods ──────────────────────────────────────────────────────────────
    if "foods" in datasets:
        for _, row in datasets["foods"].iterrows():
            name = str(row.get("food_name", "")).replace("_", " ")
            safe = str(row.get("is_safe", "")).lower()
            trimester = str(row.get("trimester", "all"))
            notes = str(row.get("notes", ""))

            if safe == "yes":
                safety_text = "is safe to eat during pregnancy"
            elif safe == "no":
                safety_text = "is NOT safe to eat during pregnancy and should be avoided"
            else:
                safety_text = f"should be consumed in limited amounts during pregnancy"

            trimester_text = (
                "throughout all trimesters"
                if trimester == "all"
                else f"in trimester {trimester}"
            )

            chunk = (
                f"Food safety: {name} {safety_text} {trimester_text}. "
                f"Details: {notes}."
            )
            chunks.append(chunk)

    # ── Exercise ───────────────────────────────────────────────────────────
    if "exercise" in datasets:
        for _, row in datasets["exercise"].iterrows():
            exercise = str(row.get("exercise", "")).replace("_", " ")
            safe = str(row.get("is_safe", "")).lower()
            trimester = str(row.get("safe_trimester", "all"))
            notes = str(row.get("notes", ""))

            if safe == "yes":
                safety_text = "is safe and recommended during pregnancy"
            elif safe == "no":
                safety_text = "is NOT safe during pregnancy and should be avoided"
            else:
                safety_text = "may be safe with modifications during pregnancy"

            trimester_text = (
                "throughout all trimesters"
                if str(trimester) == "all"
                else f"for trimester {trimester}"
            )

            chunk = (
                f"Exercise guidance: {exercise} {safety_text} {trimester_text}. "
                f"Details: {notes}."
            )
            chunks.append(chunk)

    # ── Symptoms ───────────────────────────────────────────────────────────
    if "symptoms" in datasets:
        for _, row in datasets["symptoms"].iterrows():
            symptom = str(row.get("symptom", "")).replace("_", " ")
            risk = str(row.get("risk_level", "low")).upper()
            recommendation = str(row.get("recommendation", ""))

            chunk = (
                f"Pregnancy symptom: {symptom}. "
                f"Risk level: {risk}. "
                f"Medical guidance: {recommendation}."
            )
            chunks.append(chunk)

    # ── Weekly Guidance ────────────────────────────────────────────────────
    if "weekly" in datasets:
        for _, row in datasets["weekly"].iterrows():
            week = str(row.get("week", ""))
            trimester = str(row.get("trimester", ""))
            baby_dev = str(row.get("baby_development", ""))
            nutrition = str(row.get("nutrition_tip", ""))
            exercise = str(row.get("exercise_tip", ""))

            chunk = (
                f"Pregnancy week {week} (Trimester {trimester}): "
                f"Baby development: {baby_dev}. "
                f"Nutrition advice: {nutrition}. "
                f"Exercise advice: {exercise}."
            )
            chunks.append(chunk)

    print(f"  ✓ Created {len(chunks)} knowledge chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 5. VECTOR DATABASE (FAISS) CREATION & LOADING
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_database(chunks: list[str], embedding_model: SentenceTransformer):
    """
    Generate embeddings for all text chunks and build a FAISS index.

    Uses L2 (Euclidean) distance for similarity search. Normalising vectors
    before indexing converts L2 distance to cosine similarity.

    Args:
        chunks: List of text strings to embed
        embedding_model: Pre-loaded SentenceTransformer model

    Returns:
        faiss.Index: The built FAISS index
        np.ndarray: The embedding matrix (shape: n_chunks × embedding_dim)
    """
    print("  Generating embeddings (this may take a moment)...")
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,   # Normalise for cosine similarity via L2
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    print(f"  ✓ Embeddings shape: {embeddings.shape} (dim={dim})")

    # Build flat L2 index — exact search, ideal for datasets < 100k docs
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"  ✓ FAISS index built with {index.ntotal} vectors")

    return index, embeddings


def save_vector_database(index: faiss.Index, chunks: list[str]):
    """Persist FAISS index and chunk text to disk for faster subsequent loads."""
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✓ FAISS index saved to {FAISS_INDEX_PATH}")


def load_vector_database():
    """
    Load FAISS index and chunks from disk if they exist.
    Returns (index, chunks) or (None, None) if not found.
    """
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"  ✓ Loaded existing FAISS index ({index.ntotal} vectors)")
        return index, chunks
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 6. RETRIEVAL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_relevant_documents(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K_RESULTS,
) -> list[str]:
    """
    Convert user query to embedding and retrieve top-K most similar chunks.

    RETRIEVAL LOGIC:
    1. Embed the query using the same model used to build the index
    2. Normalise the query vector (consistent with normalised index)
    3. FAISS L2 search returns top-K nearest neighbours
    4. Return the corresponding text chunks

    Args:
        query: User's natural language question
        index: Built FAISS index
        chunks: List of all text chunks (parallel to index)
        embedding_model: SentenceTransformer model
        top_k: Number of results to retrieve

    Returns:
        List of most relevant text chunks
    """
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # -1 means no result found
            retrieved.append(chunks[idx])

    return retrieved


# ─────────────────────────────────────────────────────────────────────────────
# 7. DANGER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_danger_symptoms(question: str) -> tuple[bool, list[str]]:
    """
    Scan the user's question for dangerous pregnancy symptom keywords.

    Returns:
        is_dangerous (bool): True if any danger keywords detected
        matched_keywords (list[str]): Which danger keywords were found
    """
    question_lower = question.lower()
    matched = [kw for kw in DANGER_KEYWORDS if kw in question_lower]
    return bool(matched), matched


def get_danger_alert(matched_keywords: list[str]) -> str:
    """Generate an emergency alert message based on detected danger keywords."""
    keywords_str = ", ".join(matched_keywords[:3])  # Show up to 3 keywords
    return (
        f"⚠️  IMPORTANT SAFETY ALERT: Your question mentions symptoms that may "
        f"require immediate medical attention ({keywords_str}). "
        f"Please contact your healthcare provider, midwife, or call emergency services "
        f"immediately. Do not wait. Your safety and your baby's safety come first.\n\n"
        f"Emergency contacts to reach:\n"
        f"• Your midwife or obstetrician\n"
        f"• Your nearest maternity unit or hospital\n"
        f"• Emergency services (112 / 999 / 911 depending on your location)\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    user_profile: dict,
    retrieved_docs: list[str],
    is_dangerous: bool,
    matched_keywords: list[str],
) -> str:
    """
    Construct the full structured prompt for Gemini.

    PROMPTING STRATEGY:
    - System persona establishes the assistant's role, constraints, and tone
    - User profile personalises the response to their specific pregnancy context
    - Retrieved medical documents ground the answer in factual knowledge
    - Danger flag adds urgency guidance when emergency symptoms are detected
    - Output format instructions ensure consistent, structured responses

    Args:
        question: The user's question
        user_profile: Dict with pregnancy_week, trimester, allergies, illnesses
        retrieved_docs: Top-K relevant medical knowledge chunks
        is_dangerous: Whether danger keywords were detected
        matched_keywords: Which danger keywords were found

    Returns:
        Formatted prompt string ready for Gemini
    """
    # ── User Profile Section ───────────────────────────────────────────────
    week = user_profile.get("pregnancy_week", "unknown")
    trimester = user_profile.get("trimester", "unknown")
    trimester_names = {1: "First", 2: "Second", 3: "Third"}
    trimester_label = trimester_names.get(trimester, str(trimester))

    allergies = user_profile.get("allergies", [])
    allergies_str = ", ".join(allergies) if allergies else "None reported"

    illnesses = user_profile.get("illnesses", [])
    illnesses_str = ", ".join(illnesses) if illnesses else "None reported"

    # ── Retrieved Knowledge Section ────────────────────────────────────────
    if retrieved_docs:
        knowledge_section = "\n".join(
            f"  [{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)
        )
    else:
        knowledge_section = "  No specific knowledge retrieved for this query."

    # ── Danger Context ─────────────────────────────────────────────────────
    danger_context = ""
    if is_dangerous:
        kw_str = ", ".join(matched_keywords)
        danger_context = (
            f"\n⚠️  DANGER ALERT: The user's question contains potentially serious "
            f"pregnancy symptom keywords: [{kw_str}]. "
            f"You MUST include emergency medical guidance in your response "
            f"and strongly recommend they contact their healthcare provider immediately.\n"
        )

    # ── Full Prompt ────────────────────────────────────────────────────────
    prompt = f"""You are a Pregnancy Health Personal Assistant — a knowledgeable, \
calm, and compassionate guide for pregnant women.

YOUR ROLE:
- Provide evidence-based pregnancy nutrition advice
- Suggest safe exercises appropriate to trimester
- Explain pregnancy symptoms clearly and accurately
- Detect and escalate dangerous symptoms
- Recommend professional healthcare when appropriate
- Always prioritise the safety of mother and baby

YOUR RULES:
- Only use the medical knowledge provided in the context below
- Never invent, guess, or fabricate medical information
- If you are unsure, say so and recommend professional consultation
- Always include a brief medical disclaimer for health questions
- Be warm, supportive, and avoid causing unnecessary anxiety
- When danger symptoms are mentioned, always recommend urgent medical care
{danger_context}
────────────────────────────────────────────
USER PROFILE:
  Pregnancy Week  : {week}
  Trimester       : {trimester_label}
  Known Allergies : {allergies_str}
  Medical History : {illnesses_str}
────────────────────────────────────────────
RETRIEVED MEDICAL KNOWLEDGE:
{knowledge_section}
────────────────────────────────────────────
USER QUESTION:
{question}
────────────────────────────────────────────

Please provide your response in the following structured format:

**Answer:**
[Direct, evidence-based answer to the question]

**Safety Advice:**
[Important safety considerations and precautions]

**Recommendation:**
[What the user should do next, including when to contact a healthcare provider]

*Medical Disclaimer: This information is for general guidance only and does not \
replace professional medical advice. Always consult your midwife, obstetrician, \
or healthcare provider for personalised medical care.*"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 9. GEMINI API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """
    Send the structured prompt to Gemini and return the text response.

    Uses Gemini's safety settings to allow medical health content while
    maintaining appropriate safety filters.

    Args:
        prompt: The full structured prompt string

    Returns:
        Gemini's generated text response
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Configure generation for medical guidance: balanced, clear, not too long
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,        # Lower temperature = more factual, less creative
            top_p=0.8,
            top_k=40,
            max_output_tokens=1024, # Enough for structured medical guidance
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        return response.text

    except Exception as e:
        return (
            f"**Answer:** I'm sorry, I was unable to generate a response at this time.\n\n"
            f"**Safety Advice:** For any urgent pregnancy concerns, please contact "
            f"your healthcare provider immediately.\n\n"
            f"**Technical note:** {str(e)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN RAG CHAT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def ask_pregnancy_assistant(
    question: str,
    user_profile: dict,
    index: faiss.Index,
    chunks: list[str],
    embedding_model: SentenceTransformer,
) -> str:
    """
    Complete RAG pipeline: takes a user question and returns structured guidance.

    PIPELINE:
    1. Danger detection — check for emergency symptoms first
    2. Retrieval — find top-K relevant medical knowledge chunks
    3. Prompt building — combine context + profile + question
    4. Generation — call Gemini for the final answer

    Args:
        question: The user's pregnancy health question
        user_profile: Dict with pregnancy context (week, trimester, allergies, etc.)
        index: FAISS vector index
        chunks: Knowledge base text chunks
        embedding_model: SentenceTransformer for query embedding

    Returns:
        Structured text response with Answer / Safety Advice / Recommendation
    """
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Step 1: Danger detection
    is_dangerous, matched_keywords = detect_danger_symptoms(question)
    if is_dangerous:
        print(f"  ⚠️  DANGER KEYWORDS DETECTED: {matched_keywords}")
        # Prepend emergency alert to response
        danger_alert = get_danger_alert(matched_keywords)
    else:
        danger_alert = ""

    # Step 2: Retrieve relevant documents
    print(f"  → Retrieving top-{TOP_K_RESULTS} relevant documents...")
    retrieved_docs = retrieve_relevant_documents(
        question, index, chunks, embedding_model, TOP_K_RESULTS
    )
    print(f"  ✓ Retrieved {len(retrieved_docs)} documents")

    # Step 3: Build prompt
    prompt = build_prompt(
        question, user_profile, retrieved_docs, is_dangerous, matched_keywords
    )

    # Step 4: Call Gemini
    print(f"  → Calling Gemini ({GEMINI_MODEL})...")
    response = call_gemini(prompt)
    print(f"  ✓ Response generated")

    # Prepend danger alert if applicable
    full_response = danger_alert + response
    return full_response


# ─────────────────────────────────────────────────────────────────────────────
# 11. INTERACTIVE CHAT SESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_chat(
    index: faiss.Index,
    chunks: list[str],
    embedding_model: SentenceTransformer,
):
    """
    Run an interactive command-line chat session with the pregnancy assistant.
    Collects user profile at startup and loops until user types 'quit'.
    """
    print("\n" + "="*60)
    print("  🤰 PREGNANCY HEALTH ASSISTANT")
    print("  Powered by Gemini + RAG")
    print("="*60)
    print("  Type 'quit' to exit | Type 'profile' to update your profile\n")

    # ── Collect User Profile ───────────────────────────────────────────────
    print("Let's set up your profile first.\n")

    try:
        week = int(input("  What week of pregnancy are you in? (e.g. 18): ").strip())
    except ValueError:
        week = 12
        print(f"  Using default week: {week}")

    trimester = 1 if week <= 13 else (2 if week <= 26 else 3)
    print(f"  You are in trimester: {trimester}")

    allergies_input = input(
        "  Do you have any food allergies? (comma-separated, or press Enter for none): "
    ).strip()
    allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]

    illnesses_input = input(
        "  Any medical conditions? (e.g. gestational diabetes, hypertension, or Enter for none): "
    ).strip()
    illnesses = [i.strip() for i in illnesses_input.split(",") if i.strip()]

    user_profile = {
        "pregnancy_week": week,
        "trimester": trimester,
        "allergies": allergies,
        "illnesses": illnesses,
    }

    print(f"\n  ✓ Profile set: Week {week}, Trimester {trimester}")
    if allergies:
        print(f"  ✓ Allergies: {', '.join(allergies)}")
    if illnesses:
        print(f"  ✓ Medical conditions: {', '.join(illnesses)}")

    print("\n" + "─"*60)
    print("  You can now ask any pregnancy health question!")
    print("  Examples:")
    print("    • What food should I eat this week?")
    print("    • Is walking safe during pregnancy?")
    print("    • I feel severe headache and dizziness, is it dangerous?")
    print("    • Can I eat tuna?")
    print("─"*60 + "\n")

    # ── Chat Loop ──────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("\n  You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n  👋 Take care of yourself and your baby. Goodbye!\n")
                break

            if user_input.lower() == "profile":
                print(f"\n  Current profile: {json.dumps(user_profile, indent=4)}")
                continue

            # Get response from RAG pipeline
            response = ask_pregnancy_assistant(
                question=user_input,
                user_profile=user_profile,
                index=index,
                chunks=chunks,
                embedding_model=embedding_model,
            )

            print(f"\n  Assistant:\n")
            print(response)
            print("\n" + "─"*60)

        except KeyboardInterrupt:
            print("\n\n  👋 Session ended. Take care!\n")
            break


# ─────────────────────────────────────────────────────────────────────────────
# 12. PROGRAMMATIC API (for integration into mobile/web apps)
# ─────────────────────────────────────────────────────────────────────────────

class PregnancyRAGAssistant:
    """
    Clean Python class API for integration into mobile/web application backends.

    Usage:
        assistant = PregnancyRAGAssistant()
        response = assistant.ask(
            question="Is tuna safe to eat?",
            user_profile={"pregnancy_week": 18, "trimester": 2, "allergies": [], "illnesses": []}
        )
    """

    def __init__(self):
        """Initialise the RAG system: load models, datasets, build FAISS index."""
        self._initialised = False
        self.index = None
        self.chunks = None
        self.embedding_model = None

    def initialise(self):
        """
        Load all components. Call this once before using ask().
        Separated from __init__ to allow lazy loading in web frameworks.
        """
        if self._initialised:
            return

        print("\n[PregnancyRAGAssistant] Initialising...")

        # Configure Gemini
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "Please set your Gemini API key in GEMINI_API_KEY variable "
                "or GEMINI_API_KEY environment variable."
            )
        genai.configure(api_key=GEMINI_API_KEY)

        # Load embedding model
        print("  Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Try to load cached FAISS index
        self.index, self.chunks = load_vector_database()

        if self.index is None:
            # Build from scratch
            print("  Building knowledge base from CSV files...")
            datasets = load_datasets()
            self.chunks = create_text_chunks(datasets)
            self.index, _ = build_vector_database(self.chunks, self.embedding_model)
            save_vector_database(self.index, self.chunks)

        self._initialised = True
        print("[PregnancyRAGAssistant] Ready ✓\n")

    def ask(self, question: str, user_profile: dict) -> str:
        """
        Ask the pregnancy assistant a question.

        Args:
            question: Natural language health question
            user_profile: {
                "pregnancy_week": int,
                "trimester": int (1, 2, or 3),
                "allergies": list[str],
                "illnesses": list[str]
            }

        Returns:
            Structured text response with Answer / Safety Advice / Recommendation
        """
        if not self._initialised:
            self.initialise()

        return ask_pregnancy_assistant(
            question=question,
            user_profile=user_profile,
            index=self.index,
            chunks=self.chunks,
            embedding_model=self.embedding_model,
        )

    def get_weekly_guidance(self, week: int) -> str:
        """Convenience method: get guidance for a specific pregnancy week."""
        question = f"What should I know about pregnancy week {week}? What is my baby doing and what should I eat?"
        profile = {
            "pregnancy_week": week,
            "trimester": 1 if week <= 13 else (2 if week <= 26 else 3),
            "allergies": [],
            "illnesses": [],
        }
        return self.ask(question, profile)

    def check_food_safety(self, food: str, week: int) -> str:
        """Convenience method: check if a specific food is safe."""
        question = f"Is {food} safe to eat during pregnancy? I am {week} weeks pregnant."
        profile = {
            "pregnancy_week": week,
            "trimester": 1 if week <= 13 else (2 if week <= 26 else 3),
            "allergies": [],
            "illnesses": [],
        }
        return self.ask(question, profile)

    def check_symptom(self, symptom: str, week: int) -> str:
        """Convenience method: evaluate a pregnancy symptom."""
        question = f"I am {week} weeks pregnant and I am experiencing {symptom}. Is this dangerous?"
        profile = {
            "pregnancy_week": week,
            "trimester": 1 if week <= 13 else (2 if week <= 26 else 3),
            "allergies": [],
            "illnesses": [],
        }
        return self.ask(question, profile)


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main entry point for running the pregnancy assistant from the command line.

    Steps:
    1. Validate Gemini API key
    2. Configure Gemini
    3. Load embedding model
    4. Load or build FAISS vector database
    5. Launch interactive chat
    """
    print("\n" + "="*60)
    print("  PREGNANCY HEALTH RAG ASSISTANT — STARTUP")
    print("="*60)

    # ── Validate API Key ───────────────────────────────────────────────────
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n  ✗ ERROR: Gemini API key not set!")
        print("  Please set your API key in one of these ways:")
        print("  1. Edit this file: GEMINI_API_KEY = 'your-key-here'")
        print("  2. Set environment variable: export GEMINI_API_KEY='your-key-here'")
        print("\n  Get your free API key at: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)

    # ── Configure Gemini ───────────────────────────────────────────────────
    print("\n[1/4] Configuring Gemini API...")
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"  ✓ Gemini configured (model: {GEMINI_MODEL})")

    # ── Load Embedding Model ───────────────────────────────────────────────
    print(f"\n[2/4] Loading embedding model ({EMBEDDING_MODEL})...")
    print("  (First run will download ~90MB model — cached after that)")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Embedding model loaded")

    # ── Build/Load Vector Database ─────────────────────────────────────────
    print("\n[3/4] Loading knowledge base...")
    index, chunks = load_vector_database()

    if index is None:
        print("  No cached index found — building from CSV files...")
        datasets = load_datasets()

        if not datasets:
            print("\n  ✗ ERROR: No datasets found!")
            print(f"  Please ensure CSV files exist in: {DATA_DIR}")
            print("  Required files: foods.csv, exercise.csv, symptoms.csv, weekly_guidance.csv")
            sys.exit(1)

        chunks = create_text_chunks(datasets)
        index, _ = build_vector_database(chunks, embedding_model)
        save_vector_database(index, chunks)
    else:
        print(f"  ✓ Using cached FAISS index ({len(chunks)} knowledge chunks)")

    # ── Launch Chat ────────────────────────────────────────────────────────
    print("\n[4/4] Starting interactive session...")
    run_interactive_chat(index, chunks, embedding_model)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE (runs without Gemini API key for testing)
# ─────────────────────────────────────────────────────────────────────────────

def demo_retrieval_only():
    """
    Demo mode: test the retrieval pipeline without needing a Gemini API key.
    Useful for verifying the FAISS index and embeddings work correctly.
    """
    print("\n" + "="*60)
    print("  DEMO MODE — Retrieval Pipeline Test (no API key needed)")
    print("="*60)

    print(f"\nLoading embedding model ({EMBEDDING_MODEL})...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load or build index
    index, chunks = load_vector_database()
    if index is None:
        datasets = load_datasets()
        chunks = create_text_chunks(datasets)
        index, _ = build_vector_database(chunks, embedding_model)
        save_vector_database(index, chunks)

    # Test queries
    test_questions = [
        "What food should I eat this week?",
        "Is walking safe during pregnancy?",
        "I am 18 weeks pregnant, can I eat tuna?",
        "I feel dizziness and severe headache, is it dangerous?",
        "What happens to my baby in week 22?",
        "Is yoga safe in the second trimester?",
    ]

    for question in test_questions:
        print(f"\n{'─'*50}")
        print(f"QUERY: {question}")

        is_dangerous, keywords = detect_danger_symptoms(question)
        if is_dangerous:
            print(f"⚠️  DANGER DETECTED: {keywords}")

        docs = retrieve_relevant_documents(question, index, chunks, embedding_model)
        print(f"TOP {len(docs)} RETRIEVED DOCUMENTS:")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc[:120]}...")

    print("\n\n✓ Demo complete. Set GEMINI_API_KEY and run main() for full chat.\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check for demo flag
    if "--demo" in sys.argv:
        demo_retrieval_only()
    else:
        main()
