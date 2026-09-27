"""RAG + LLM generation with grounding."""
from typing import List, Dict, Any
from src.config import OPENAI_API_KEY, OPENAI_MODEL, USE_MOCK_LLM


def build_context(retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return "No relevant knowledge found."
    parts = []
    for i, r in enumerate(retrieved, 1):
        label = r.get("label", "unknown")
        parts.append(f"[{i}] (label={label}, score={r.get('score', 0):.2f})\n{r['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a grounded answer using retrieved context.
    Uses OpenAI if key is set, otherwise a deterministic mock LLM.
    """
    context = build_context(retrieved)

    if USE_MOCK_LLM or not OPENAI_API_KEY:
        return _mock_llm(query, retrieved, context)

    try:
        from openai import OpenAI
        import json

        client = OpenAI(api_key=OPENAI_API_KEY)
        system = """You are a careful analyst. Answer ONLY using the provided knowledge context.
Return valid JSON with keys:
- label: "scam" or "safe"
- confidence: float 0-1
- explanation: short grounded explanation
- used_sources: list of source indices you relied on
If context is insufficient, say so and lower confidence."""

        user = f"MESSAGE:\n{query}\n\nKNOWLEDGE CONTEXT:\n{context}\n\nRespond with JSON only."

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["context_used"] = context
        result["mode"] = "llm"
        return result
    except Exception as e:
        out = _mock_llm(query, retrieved, context)
        out["llm_error"] = str(e)
        return out


def _mock_llm(query: str, retrieved: List[Dict[str, Any]], context: str) -> Dict[str, Any]:
    """Deterministic fallback that still uses retrieved labels."""
    if not retrieved:
        return {
            "label": "unknown",
            "confidence": 0.3,
            "explanation": "No relevant knowledge retrieved. Cannot make a reliable decision.",
            "used_sources": [],
            "context_used": context,
            "mode": "mock",
        }

    # Majority vote weighted by score
    scam_score = 0.0
    safe_score = 0.0
    for r in retrieved:
        s = r.get("score", 0.5)
        if r.get("label") == "scam":
            scam_score += s
        else:
            safe_score += s

    if scam_score > safe_score:
        label = "scam"
        conf = min(0.95, 0.55 + (scam_score - safe_score) * 0.3)
        explanation = (
            f"Retrieved knowledge suggests this is a scam. "
            f"Top match: \"{retrieved[0]['text'][:80]}...\""
        )
    else:
        label = "safe"
        conf = min(0.95, 0.55 + (safe_score - scam_score) * 0.3)
        explanation = (
            f"Retrieved knowledge suggests this is relatively safe. "
            f"Top match: \"{retrieved[0]['text'][:80]}...\""
        )

    return {
        "label": label,
        "confidence": round(conf, 3),
        "explanation": explanation,
        "used_sources": list(range(1, min(3, len(retrieved)) + 1)),
        "context_used": context,
        "mode": "mock",
    }
