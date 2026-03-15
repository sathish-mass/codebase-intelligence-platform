from collections import defaultdict
from typing import Dict, List
import re


def clean_snippet(text: str, max_length: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


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
    for file_path, file_matches in ranked_files[:3]:
        best_match = sorted(file_matches, key=lambda x: x.get("distance", 999))[0]
        evidence.append(
            {
                "file_path": file_path,
                "file_name": best_match.get("file_name"),
                "source_type": best_match.get("source_type"),
                "chunk_index": best_match.get("chunk_index"),
                "distance": best_match.get("distance"),
                "snippet": clean_snippet(best_match.get("content", ""))
            }
        )

    answer_lines = []
    answer_lines.append(f"Question: {question}")
    answer_lines.append("")
    answer_lines.append("Most relevant files found in the indexed knowledge base:")

    for idx, item in enumerate(evidence, start=1):
        answer_lines.append(
            f"{idx}. {item['file_path']} [{item['source_type']}]"
        )

    answer_lines.append("")
    answer_lines.append("Most relevant evidence:")
    for item in evidence:
        answer_lines.append(
            f"- {item['file_path']} [{item['source_type']}] "
            f"(chunk {item['chunk_index']}): {item['snippet']}"
        )

    return {
        "answer": "\n".join(answer_lines),
        "key_files": top_files,
        "evidence": evidence
    }