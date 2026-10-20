#projectsetup

# step-1:import packages
import os
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()

#step-2:gemini api configuration
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)

GEN_MODEL_NAME = "gemini-3.8-flash"
EMBED_MODEL_NAME = "gemini-embedding-001"

#STEP-3: PDF LOADER(PURPOSE: Extracting text from PDF)
#Responsible for extracting text(single pdf,multiple pdfs--We gonna parse(scraping) each page)
def load_pdf(file_path):
    """Extract text from a PDF page by page."""

    reader = PdfReader(file_path)
    source_name = os.path.basename(file_path)

    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages_data.append({
                "source": source_name,
                "page": page_number,
                "text": text
            })

    return pages_data


def load_multiple_pdfs(file_paths):
    """Load multiple PDFs."""

    all_pages = []

    for file_path in file_paths:
        all_pages.extend(load_pdf(file_path))

    return all_pages

#Module-2: chunking mechanism
#The text we have received from module-1--->chunking(breaking down complex text into small parts)

def chunk_pages(pages_data, chunk_size=250, overlap=50):
    chunks = []
    chunk_id = 0

    for page in pages_data:
        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": page["source"],
                        "page": page["page"],
                    }
                )

                chunk_id += 1

            if end >= len(words):
                break

            start = end - overlap

    return chunks


from sklearn.feature_extraction.text import TfidfVectorizer

# Store the vectorizer so questions use the same vocabulary as PDF chunks.
vectorizer = None

#module-3: Text Embeddings(converting text to vector)

# artificial intelligence(AI) can be tackled by any junior level software engineer,with just 2 things--Accountability--->[0.45.0.23,]
# Judge the output,You are not gonna lose job because of AI, YOu gonna lose if you blindly trust the output of AI

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    global vectorizer

    if task_type == "RETRIEVAL_DOCUMENT":
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
        return vectorizer.fit_transform(texts).toarray().astype("float32")

    if vectorizer is None:
        raise ValueError("Process the PDF documents first.")

    return vectorizer.transform(texts).toarray().astype("float32")

#module-4: Vector size(FAISS)

# question bank(10 concepts related to AI)

# user: What is AI??
# model: It have go through all these 10AI concepts

# Each concept will be assigend a score->Top 3 will be picked up, And the model will take care of the rest(blend,share optimal solution)

def build_vector_index(chunks):
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(
        texts,
        task_type="RETRIEVAL_DOCUMENT"
    )

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("LOCAL EMBEDDINGS SHAPE:", embeddings.shape)
    print("NUMBER OF CHUNKS:", len(chunks))

    return index

#module-5: Retriever(Top-k similarity search)

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=5,
    similarity_threshold=0.05
):
    query_embedding = embed_texts(
        [question],
        task_type="RETRIEVAL_QUERY"
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        score = float(score)

        if score < similarity_threshold:
            continue

        chunk = chunks[idx].copy()
        chunk["score"] = score
        results.append(chunk)

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results

#module-6: Prompt builder

#we are instructing what model have to do, what precautions need to taken, And how to hadnle edge cases(ai,ml,dbms...)
#ipl,,,our model should flag and reject that user input


def build_prompt(question, retrieved_chunks, chat_history=None):

    context_blocks = []

    for chunk in retrieved_chunks:

        context_blocks.append(
            f"[Source: {chunk['source']}, "
            f"Page: {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    history_text = ""

    if chat_history:

        for turn in chat_history[-6:]:

            history_text += (
                f"{turn['role'].capitalize()}: "
                f"{turn['content']}\n"
            )

    prompt = f"""
You are CampusGPT, an AI learning assistant for students.

Your job is to answer questions ONLY using the information
provided in the retrieved context.

Do NOT use outside knowledge.

If the answer cannot be found in the context, say:

"I couldn't find that information in the uploaded PDF(s)."

Always cite the source and page when giving information.

Previous conversation:
{history_text}

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    return prompt

#module-7: Answer generation

import time

def generate_answer(prompt):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL_NAME,
                contents=prompt
            )
            return response.text

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return (
                    "Gemini is temporarily unavailable because the model "
                    "is experiencing high demand. Please try again in a moment."
                )

#module-8:conversation memory

#module-9:full pipeline(UI)

def ask(question, index, chunks, chat_history=None):

    retrieved = retrieve_relevant_chunks(
        question,
        index,
        chunks
    )

    if not retrieved:

        answer = (
            "I couldn't find that information "
            "in the uploaded PDF(s)."
        )

        return answer, retrieved

    prompt = build_prompt(
        question,
        retrieved,
        chat_history
    )

    answer = generate_answer(prompt)

    return answer, retrieved