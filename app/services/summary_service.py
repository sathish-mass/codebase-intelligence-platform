from typing import Dict, List

from app.services.llm_service import ask_huggingface_llm


def build_summary_context(matches: List[Dict], max_items: int = 8) -> str:
    blocks = []

    for idx, match in enumerate(matches[:max_items], start=1):
        blocks.append(
            f"""[File {idx}]
Path: {match.get("file_path", "unknown")}
Type: {match.get("source_type", "unknown")}
Chunk: {match.get("chunk_index", "unknown")}

{match.get("content", "")}
"""
        )

    return "\n\n".join(blocks)


def summarize_codebase(matches: List[Dict]) -> Dict:
    if not matches:
        return {
            "summary": "I could not find enough indexed content to summarize this codebase.",
            "important_files": []
        }

    important_files = []
    seen = set()

    for match in matches:
        path = match.get("file_path")
        if path and path not in seen:
            seen.add(path)
            important_files.append(path)

    context = build_summary_context(matches)

    prompt = f"""
You are an AI codebase analyst.

Based only on the retrieved code context below, write a high-level summary of this codebase.

Rules:
- Explain what the codebase appears to do
- Mention the major responsibilities or modules
- Mention important files when possible
- Keep the answer concise but useful
- If the context is limited, say that the summary is partial

Retrieved context:
{context}
""".strip()

    try:
        summary = ask_huggingface_llm(prompt)
    except Exception as e:
        summary = f"Summary generation failed: {str(e)}"

    return {
        "summary": summary,
        "important_files": important_files[:8]
    }