"""
LLM-based risk explanation and retention strategy from customer profile.
Uses OpenAI API if key is set; otherwise returns mock insights.
"""
from pathlib import Path
import sys
import os
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings


def _mock_insight(customer_id: str, churn_prob: float, profile_summary: dict) -> dict:
    return {
        "customer_id": customer_id,
        "risk_summary": (
            f"Customer shows {'elevated' if churn_prob > 0.5 else 'moderate'} churn risk (score: {churn_prob:.2f}). "
            "Key factors: recency of last transaction, session frequency, and support ticket count."
        ),
        "retention_strategy": (
            "Recommend: personalized outreach, loyalty offer, or proactive support contact. "
            "Consider win-back campaign if recency_days is high."
        ),
        "model_used": "mock (no API key)",
    }


def _openai_insight(
    customer_id: str,
    churn_prob: float,
    profile_summary: dict,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        prompt = f"""You are a customer success analyst. Given this customer profile and churn risk score, provide:
1. A short "risk_summary" (2-3 sentences) explaining why they might churn.
2. A short "retention_strategy" (2-3 sentences) with concrete actions.

Customer ID: {customer_id}
Churn probability: {churn_prob:.2f}
Profile (key metrics): {json.dumps(profile_summary, indent=0)}

Respond with valid JSON only: {"risk_summary": "...", "retention_strategy": "..."}"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        data = json.loads(text)
        data["customer_id"] = customer_id
        data["model_used"] = model
        return data
    except Exception as e:
        return _mock_insight(
            customer_id,
            churn_prob,
            {**profile_summary, "error": str(e)},
        )


def generate_risk_insight(
    customer_id: str,
    churn_probability: float,
    profile: dict,
    openai_api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Generate risk explanation and retention strategy."""
    api_key = openai_api_key or getattr(settings, "openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    use_mock = getattr(settings, "llm_mock_if_no_key", True)
    if not api_key and use_mock:
        return _mock_insight(customer_id, churn_probability, profile)
    if api_key:
        return _openai_insight(
            customer_id,
            churn_probability,
            profile,
            api_key,
            model or getattr(settings, "openai_model", "gpt-4o-mini"),
        )
    return _mock_insight(customer_id, churn_probability, profile)
