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
                "do_sample": False
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

# ---------------- CLEAN OUTPUT ----------------
def clean_output(prompt, raw):

    if isinstance(raw, dict):
        text = raw.get("text", "")
    elif isinstance(raw, list):
        text = raw[0].get("generated_text", "")
    else:
        text = str(raw)

    answer = text

    if prompt in answer:
        answer = answer.replace(prompt, "")

    for stop in ["Context:", "Question:", "Rules:", "Answer:"]:
        if stop in answer:
            answer = answer.split(stop)[0]

    return answer.strip()

# ---------------- MAIN RAG ----------------
def rag_chain(question):

    trace_id = create_trace_id()
    start_time = time.perf_counter()

    docs = retriever.invoke(question)

    if not docs:
        return "I don't know"

    context = "\n\n".join([d.page_content[:500] for d in docs])

    quality = retrieval_quality(question, docs)

    prompt = f"""You are a strict QA system.

Rules:
- Answer ONLY using context
- If answer is not in context, say "I don't know"
- No explanation

Context:
{context}

Question: {question}

Answer:"""

    llm_local = get_llm()
    raw = llm_local.invoke(prompt)

    answer = clean_output(prompt, raw)

    if len(answer.strip()) == 0 or quality < 0.3:
        answer = "I don't know"

    latency = time.perf_counter() - start_time

    # ---------------- FIXED COST CALCULATION ----------------
    tokens = len(prompt.split()) + len(answer.split())

    cost_per_1k_tokens = 0.0001   # simulated cost rate
    cost = (tokens / 1000) * cost_per_1k_tokens

    # ---------------- METRICS ----------------
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

        if keyword.lower() not in ans.lower():
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
