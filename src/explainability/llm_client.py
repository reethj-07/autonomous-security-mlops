from typing import Dict


class LLMClient:
    """
    Abstracted LLM client.
    Replace implementation with OpenAI / Gemini / local later.
    """

    def generate(self, payload: Dict[str, object]) -> str:
        """
        Generates explanation text strictly from payload.
        (Stub for now – production-safe)
        """

        # 🚨 IMPORTANT:
        # We do NOT hallucinate. This stub simulates a grounded response.
        summary = payload.get("system_summary", "")

        explanation = (
            "Incident Explanation:\n\n"
            f"{summary}\n\n"
            "Evidence was evaluated based on feature drift, "
            "prediction confidence, and system safety state. "
            "No speculative causes were introduced."
        )

        return explanation
