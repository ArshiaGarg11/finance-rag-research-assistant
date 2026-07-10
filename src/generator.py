"""
generator.py
------------
Takes retrieved chunks from the retriever and generates a grounded,
cited answer using Groq's Llama 3 model.

Key design decisions:
- System prompt explicitly forbids answering from outside the context
- Model must cite every factual claim with [Source N]
- Returns both the answer and which sources were actually used
- Temperature=0 for factual consistency (no creativity needed here)

Usage:
    from generator import FinanceGenerator
    gen = FinanceGenerator()
    answer = gen.generate(query="What is Apple's services revenue?", context_chunks=results)
    print(answer["answer"])
    print(answer["sources_used"])
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL         = "llama-3.3-70b-versatile"   # best free model on Groq
MAX_TOKENS    = 1024
TEMPERATURE   = 0.0   # deterministic — we want facts, not creativity
MAX_CONTEXT_CHUNKS = 5  # max chunks to include in prompt

SYSTEM_PROMPT = """You are a financial research assistant that answers questions about company filings and earnings calls.

STRICT RULES you must always follow:
1. Answer ONLY using the information provided in the numbered sources below. Do not use any outside knowledge.
2. Every factual claim in your answer must be followed by a citation like [Source 1] or [Source 2].
3. If the provided sources do not contain enough information to answer the question, say exactly: "The provided documents do not contain sufficient information to answer this question."
4. Be concise and precise. Prefer specific numbers and facts over vague statements.
5. Do not speculate or infer beyond what is explicitly stated in the sources.
6. If multiple sources say conflicting things, mention the conflict explicitly."""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered source list for the prompt.
    Each source includes company, document type, section, and text.
    """
    parts = []
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], 1):
        meta = chunk["metadata"]

        # Build a human-readable source label
        company  = meta.get("company", "Unknown")
        doc_type = meta.get("doc_type", "")
        period   = meta.get("year", meta.get("quarter", ""))
        section  = meta.get("section", meta.get("speakers", ""))

        if doc_type == "10-K":
            label = f"{company} 10-K ({period}) — {section}"
        elif doc_type == "earnings_transcript":
            label = f"{company} Earnings Call ({period}) — {section}"
        else:
            label = f"{company} {doc_type} ({period})"

        parts.append(f"[Source {i}: {label}]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def extract_cited_sources(answer: str, num_chunks: int) -> list[int]:
    """
    Parse the answer text to find which source numbers were actually cited.
    Returns list of 1-indexed source numbers.
    """
    cited = set()
    for match in re.finditer(r'\[Source (\d+)\]', answer):
        n = int(match.group(1))
        if 1 <= n <= num_chunks:
            cited.add(n)
    return sorted(cited)


# ── Main class ─────────────────────────────────────────────────────────────────

class FinanceGenerator:
    def __init__(self, model: str = MODEL):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Create a .env file in your project root with: GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=api_key)
        self.model  = model
        print(f"Generator ready — model: {self.model}")

    def generate(
        self,
        query:          str,
        context_chunks: list[dict],
        verbose:        bool = False,
    ) -> dict:
        """
        Generate a grounded answer for the query using the retrieved chunks.

        Returns:
            {
                "query":         str,
                "answer":        str,
                "sources_used":  list[int],   # 1-indexed source numbers cited
                "sources":       list[dict],  # full metadata for cited sources
                "model":         str,
                "tokens_used":   int,
            }
        """
        if not context_chunks:
            return {
                "query":        query,
                "answer":       "No relevant documents were retrieved for this query.",
                "sources_used": [],
                "sources":      [],
                "model":        self.model,
                "tokens_used":  0,
            }

        # Build the user message: context + question
        context_str = format_context(context_chunks)
        user_message = f"""Here are the relevant excerpts from financial documents:

{context_str}

---

Based ONLY on the sources above, answer the following question. Cite sources using [Source N] after each claim.

Question: {query}"""

        if verbose:
            print(f"\n{'='*60}")
            print(f"PROMPT SENT TO GROQ:")
            print(f"{'='*60}")
            print(user_message[:2000] + "..." if len(user_message) > 2000 else user_message)

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        answer      = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        cited_nums  = extract_cited_sources(answer, len(context_chunks))

        # Build source metadata for cited sources only
        cited_sources = [
            context_chunks[i - 1]["metadata"]
            for i in cited_nums
            if i <= len(context_chunks)
        ]

        return {
            "query":        query,
            "answer":       answer,
            "sources_used": cited_nums,
            "sources":      cited_sources,
            "model":        self.model,
            "tokens_used":  tokens_used,
        }

    def pretty_print(self, result: dict) -> None:
        """Print a formatted answer with citations."""
        print(f"\n{'='*60}")
        print(f"Q: {result['query']}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        print(f"\n--- Sources cited: {result['sources_used']} ---")
        for i, src in enumerate(result['sources'], 1):
            company  = src.get('company', '?')
            doc_type = src.get('doc_type', '?')
            period   = src.get('year', src.get('quarter', '?'))
            section  = src.get('section', src.get('speakers', ''))
            print(f"  [{i}] {company} {doc_type} ({period}) — {section}")
        print(f"\nTokens used: {result['tokens_used']}")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from retriever import FinanceRetriever

    # Load retriever (uses saved indexes — fast)
    retriever = FinanceRetriever()
    retriever.build_index()   # loads from disk if already built

    # Initialize generator
    gen = FinanceGenerator()

    # Test queries
    test_queries = [
        "What is Apple's total revenue and how has it changed year over year?",
        "What are Southwest Airlines' main fuel hedging strategies?",
        "What are the biggest risk factors Apple faces in its supply chain?",
        "How does Apple generate revenue from its Services segment?",
    ]

    for query in test_queries:
        # Retrieve
        results = retriever.search(query, top_k=5)
        # Generate
        answer  = gen.generate(query, results)
        # Print
        gen.pretty_print(answer)
        print()
