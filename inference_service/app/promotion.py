from app.config import settings


def promotion_allowed() -> None:
    """
    Hard gate for model promotion.
    Used by CI and deployment workflows.
    """
    if settings.safe_mode:
        raise RuntimeError(
            "SAFE MODE enabled — model promotion blocked"
        )
