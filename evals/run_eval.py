import json
import sys
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8000/ask"
REFUSAL_TEXT = "i don't know based on the provided documents"
MAX_REFUSAL_LEN = 120
MIN_ANSWER_RATE = 0.80
MIN_RETRIEVAL_RATE = 0.80

items = json.loads(Path("evals/datasets/golden.json").read_text(encoding="utf-8"))
results = []

for item in items:
    start = time.time()
    r = requests.post(API, json={"question": item["question"]}, timeout=300).json()
    latency = round(time.time() - start, 1)
    answer = r["answer"].lower()
    context = " ".join(r["sources"]).lower()

    if item.get("should_refuse"):
        retrieval_ok = None
        # Must be the exact refusal sentence and nothing much else
        answer_ok = REFUSAL_TEXT in answer and len(answer) < MAX_REFUSAL_LEN
    else:
        kws = [k.lower() for k in item["keywords"]]
        retrieval_ok = all(k in context for k in kws)
        answer_ok = sum(k in answer for k in kws) >= max(1, len(kws) / 2)

    results.append({
        "question": item["question"],
        "should_refuse": item.get("should_refuse", False),
        "retrieval_ok": retrieval_ok,
        "answer_ok": answer_ok,
        "latency_s": latency,
        "answer": r["answer"],
    })
    flag = "PASS" if answer_ok else "FAIL"
    print(f"{flag}  retrieval={retrieval_ok}  {latency}s  {item['question']}")
    if item.get("should_refuse"):
        print(f"      answer: {r['answer'][:150]}")

answerable = [x for x in results if not x["should_refuse"]]
refusals = [x for x in results if x["should_refuse"]]
retrieval_rate = sum(x["retrieval_ok"] for x in answerable) / len(answerable)
answer_rate = sum(x["answer_ok"] for x in answerable) / len(answerable)
refusal_rate = sum(x["answer_ok"] for x in refusals) / len(refusals)
latencies = sorted(x["latency_s"] for x in results)
median_latency = latencies[len(latencies) // 2]

summary = {
    "retrieval_rate": round(retrieval_rate, 2),
    "answer_rate": round(answer_rate, 2),
    "refusal_rate": round(refusal_rate, 2),
    "median_latency_s": median_latency,
}
print("\nSUMMARY", summary)

Path("evals/results").mkdir(parents=True, exist_ok=True)
Path("evals/results/latest.json").write_text(
    json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
)

if answer_rate < MIN_ANSWER_RATE or retrieval_rate < MIN_RETRIEVAL_RATE or refusal_rate < MIN_ANSWER_RATE:
    print("QUALITY GATE: FAILED")
    sys.exit(1)
print("QUALITY GATE: PASSED")