from dataclasses import dataclass


@dataclass
class EnsembleResult:
    nlp_score: float
    vision_score: float
    final_score: float
    verdict: str


def weighted_ensemble(nlp_score: float, vision_score: float, nlp_weight: float = 0.55) -> EnsembleResult:
    nlp_score = float(max(0.0, min(1.0, nlp_score)))
    vision_score = float(max(0.0, min(1.0, vision_score)))
    nlp_weight = float(max(0.0, min(1.0, nlp_weight)))
    vision_weight = 1.0 - nlp_weight

    final_score = (nlp_weight * nlp_score) + (vision_weight * vision_score)
    verdict = "phishing" if final_score >= 0.5 else "legitimate"

    return EnsembleResult(
        nlp_score=nlp_score,
        vision_score=vision_score,
        final_score=final_score,
        verdict=verdict,
    )
