"""Fire a stream of questions at the agent so the dashboard has live data.

    python scripts/simulate_traffic.py --n 20 --delay 3
"""
import argparse
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import DocsAgent  # noqa: E402
from app.metrics import MetricsStore  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

QUESTIONS = [
    "How much does the Team plan cost?",
    "Can I get a refund after 20 days?",
    "How do I export all my notes?",
    "My notes aren't syncing on my phone, what should I try?",
    "Is my data encrypted?",
    "What's the file size limit for attachments?",
    "How do I cancel my subscription?",
    "Do you offer discounts for students?",
    "Where is my data stored?",
    "How do I recover a deleted notebook?",
    "What happens to my notes if I downgrade to Free?",
    "Does Nimbus support two-factor authentication?",
    # off-topic questions, to show weak-retrieval answers on the dashboard
    "What's the capital of Australia?",
    "Who won the 2018 World Cup?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of questions")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between questions")
    args = ap.parse_args()

    agent = DocsAgent(VectorStore(), MetricsStore())
    if agent.store.count() == 0:
        print("Vector DB is empty. Run: python scripts/ingest_folder.py data/sample_docs")
        return

    session = f"sim-{uuid.uuid4().hex[:6]}"
    for i in range(args.n):
        q = random.choice(QUESTIONS)
        r = agent.run(q, session_id=session)
        flag = "✓" if r.grounded else "~"
        print(f"[{i + 1}/{args.n}] {flag} {r.latency_ms / 1000:5.1f}s  score={r.top_score or 0:.2f}  {q}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
