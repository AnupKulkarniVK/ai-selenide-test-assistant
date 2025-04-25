import os
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------- Config ---------------
TEST_DIR      = "test"
INDEX_PATH    = "model/embeddings_index.faiss"
META_PATH     = "model/chunks.json"
TOP_K         = 5
CODE_MODEL    = "codellama/CodeLlama-7b-Instruct-hf"
MAX_NEW_TOKENS= 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load metadata + docs + TF-IDF
with open(META_PATH, "r", encoding="utf-8") as f:
    file_paths = json.load(f)
file_paths = [p[3:] if p.startswith("../") else p for p in file_paths]

docs = []
for p in file_paths:
    with open(p, "r", encoding="utf-8", errors="ignore") as fd:
        docs.append(fd.read())

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(docs).toarray().astype("float32")

# 2. Load FAISS index
index = faiss.read_index(INDEX_PATH)

def retrieve_snippets(query: str, top_k=TOP_K):
    q_vec = vectorizer.transform([query]).toarray().astype("float32")
    D, I = index.search(q_vec, top_k)
    snippets = []
    for idx in I[0]:
        # grab first 50 lines for context
        lines = open(file_paths[idx], "r", encoding="utf-8", errors="ignore")\
                    .read().splitlines()
        snippets.append("\n".join(lines[:50]))
    return snippets

# 3. Load the code model
tokenizer = AutoTokenizer.from_pretrained(
     CODE_MODEL,
     trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
     CODE_MODEL,
     trust_remote_code=True,
     device_map="auto",
     torch_dtype=torch.float16
 ).to(DEVICE)

gen_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.2,
    top_p=0.95,
    do_sample=True,
)

# 4. Generation function
def generate_test(query: str):
    # a) Retrieve similar examples
    examples = retrieve_snippets(query)
    context  = "\n\n".join(examples)

    # b) Build prompt
    prompt = (
        f"### Examples of Selenide @Test methods:\n{context}\n\n"
        f"### Task:\nWrite a Java Selenide @Test method that:\n{query}\n\n"
        "### Answer:\n"
    )

    # c) Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, generation_config=gen_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Natural-language test scenario")
    args = parser.parse_args()

    print("\n===== Generated Test =====\n")
    print(generate_test(args.scenario))

RAG?
Finetuning?
