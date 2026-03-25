from collections import defaultdict
from typing import Dict, List
import re

from app.services.llm_service import ask_huggingface_llm


def clean_snippet(text: str, max_length: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_symbol_label(match: Dict) -> str:
    symbol_name = match.get("symbol_name")
    symbol_type = match.get("symbol_type")
    parent_symbol = match.get("parent_symbol")

    if symbol_name and parent_symbol:
        return f"{symbol_type}: {parent_symbol}.{symbol_name}"

    if symbol_name:
        return f"{symbol_type}: {symbol_name}"

    return "symbol: unknown"


def build_prompt(question: str, matches: List[Dict]) -> str:
    context_blocks = []

    for idx, match in enumerate(matches, start=1):
        file_path = match.get("file_path", "unknown")
        source_type = match.get("source_type", "unknown")
        chunk_index = match.get("chunk_index", "unknown")
        symbol_name = match.get("symbol_name")
        symbol_type = match.get("symbol_type")
        parent_symbol = match.get("parent_symbol")
        content = match.get("content", "")

        symbol_line = f"Symbol: {symbol_type or 'unknown'}"
        if symbol_name:
            if parent_symbol:
                symbol_line = f"Symbol: {symbol_type} {parent_symbol}.{symbol_name}"
            else:
                symbol_line = f"Symbol: {symbol_type} {symbol_name}"

        context_blocks.append(
            f"""[Context {idx}]
File: {file_path}
Type: {source_type}
Chunk: {chunk_index}
{symbol_line}

{content}
"""
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
Use the following retrieved codebase context to answer the question.

Rules:
- Answer only from the context below.
- If the answer is not clearly present, say: "I could not find this in the indexed codebase."
- Mention relevant file paths and symbols when possible.
- Be concise but useful.

Question:
{question}

Context:
{context_text}
"""
    return prompt.strip()


def build_grounded_answer(question: str, matches: List[Dict]) -> Dict:
    if not matches:
        return {
            "answer": "I could not find relevant code or documentation for this question in the indexed knowledge base.",
            "key_files": [],
            "evidence": []
        }

    file_scores = defaultdict(list)

    for match in matches:
        file_path = match.get("file_path", "unknown")
        file_scores[file_path].append(match)

    ranked_files = sorted(
        file_scores.items(),
        key=lambda item: min(m.get("distance", 999) for m in item[1])
    )

    top_files = [file_path for file_path, _ in ranked_files[:3]]

    evidence = []
    for file_path, file_matches in ranked_files[:5]:
        best_match = sorted(file_matches, key=lambda x: x.get("distance", 999))[0]
        evidence.append(
            {
                "file_path": file_path,
                "file_name": best_match.get("file_name"),
                "source_type": best_match.get("source_type"),
                "chunk_index": best_match.get("chunk_index"),
                "symbol_name": best_match.get("symbol_name"),
                "symbol_type": best_match.get("symbol_type"),
                "parent_symbol": best_match.get("parent_symbol"),
                "symbol_label": format_symbol_label(best_match),
                "distance": best_match.get("distance"),
                "snippet": clean_snippet(best_match.get("content", ""))
            }
        )

    prompt = build_prompt(question, matches[:5])

    try:
        llm_answer = ask_huggingface_llm(prompt)
    except Exception as e:
        llm_answer = f"LLM answer generation failed: {str(e)}"

    return {
        "answer": llm_answer,
        "key_files": top_files,
        "evidence": evidence
    }