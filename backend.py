import os
import time
import uuid
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- DATA ----------------
os.makedirs("data", exist_ok=True)

file_path = "data/About_Ai.txt"

if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("AI ML DL fundamentals and applications in real world systems.")

loader = TextLoader(file_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# ---------------- VECTOR DB ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------------- METRICS ----------------
METRICS = {
    "latencies": [],
    "costs": [],
    "tokens": [],
    "quality": []
}

TRACE_LOGS = []

def create_trace_id():
    return str(uuid.uuid4())

# ---------------- LLM ----------------
llm = None

def get_llm():
    global llm
    if llm is None:
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline

        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct"
        )

        llm = HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={
                "max_new_tokens": 256,
                "temperature": 0.2,
                "do_sample": False,
                "eos_token_id": 25100,
                "pad_token_id": 25100
            }
        )

    return llm

# ---------------- QUALITY ----------------
from sentence_transformers import util

def retrieval_quality(question, docs):
    q_emb = embeddings.embed_query(question)

    scores = []
    for d in docs:
        d_emb = embeddings.embed_query(d.page_content)
        scores.append(util.cos_sim(q_emb, d_emb).item())

    return max(scores) if scores else 0.0

# ---------------- MAIN RAG ----------------
def rag_chain(question):

    trace_id = create_trace_id()
    start_time = time.perf_counter()

    docs = retriever.invoke(question)

    if not docs:
        return "I don't know"

    quality = retrieval_quality(question, docs)

    if quality < 0.30:
        return "I don't know"

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a strict AI assistant.

RULES:
- Use ONLY the context
- If not found, say exactly: I don't know
- DO NOT explain anything
- DO NOT add extra text
- Output must be ONLY the final answer (one line)

Context:
{context}

Question: {question}

Answer:"""

    llm_local = get_llm()
    answer = llm_local.invoke(prompt)

    latency = time.perf_counter() - start_time

    tokens = len(prompt.split()) + len(answer.split())
    cost = (tokens / 1_000_000) * 0.07

    METRICS["latencies"].append(latency)
    METRICS["costs"].append(cost)
    METRICS["tokens"].append(tokens)
    METRICS["quality"].append(quality)

    TRACE_LOGS.append({
        "trace_id": trace_id,
        "question": question,
        "answer": answer,
        "latency": latency,
        "cost": cost,
        "tokens": tokens,
        "quality": quality,
        "context_used": context
    })

    return answer

# ---------------- CI REGRESSION ----------------

test_cases = [
    ("What is Artificial Intelligence?", "intelligence"),
    ("What is Machine Learning?", "learning"),
    ("What is Deep Learning?", "neural")
]

def ci_regression():
    failures = 0

    for q, keyword in test_cases:
        ans = rag_chain(q)

        if ans is None or keyword.lower() not in ans.lower():
            failures += 1

    failure_rate = failures / len(test_cases)

    print("\n--- CI REGRESSION RESULT ---")
    print("FAILURE RATE:", failure_rate)

    if failure_rate > 0.3:
        raise Exception("REGRESSION FAILED")
    else:
        print("REGRESSION PASSED")


if __name__ == "__main__":
    ci_regression()
