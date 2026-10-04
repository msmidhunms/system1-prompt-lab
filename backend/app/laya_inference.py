"""Laya Router-based inference for intent classification."""

from typing import Dict, Any, Tuple
import json

try:
    from laya import Router
    LAYA_AVAILABLE = True
except ImportError:
    LAYA_AVAILABLE = False
    Router = None

from app.tasks.intent.labels import SearchIntent


# Default question criteria for intent classification
DEFAULT_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Classify the search query intent based on what the user is trying to accomplish.",
        "criteria": {
            SearchIntent.INFORMATIONAL: "User seeks information, education, or knowledge (how to, what is, why, explain, tutorial, guide, learn, research)",
            SearchIntent.NAVIGATIONAL: "User seeks to reach a specific website or resource (login, sign in, account, official, facebook, gmail, twitter, instagram, github, homepage)",
            SearchIntent.COMMERCIAL: "User researches products/services before purchasing (best, top, review, comparison, vs, recommended, alternative, verdict)",
            SearchIntent.TRANSACTIONAL: "User intends to complete a transaction (buy, purchase, order, download, book, reserve, subscribe, rent, price, cost, deal)",
        }
    },
    "confidence": {
        "type": "score",
        "instructions": "How confident are you in this classification?",
        "criteria": ["low confidence", "medium confidence", "high confidence"]
    }
}


def get_laya_router() -> Router:
    """Initialize and return Laya Router."""
    if not LAYA_AVAILABLE:
        raise ImportError("Laya is not installed. Install with: pip install laya")
    return Router()


def classify_with_laya(
    query: str,
    questions: Dict[str, Any] = None
) -> Tuple[SearchIntent, float]:
    """
    Classify intent using Laya Router.

    Args:
        query: The search query to classify
        questions: Custom question definitions (uses defaults if None)

    Returns:
        Tuple of (predicted_intent, confidence_score)
    """
    if not LAYA_AVAILABLE:
        # Fallback if Laya not available
        from app.inference import classify_intent
        return classify_intent(query)

    try:
        router = get_laya_router()
        questions_to_use = questions or DEFAULT_QUESTIONS

        result = router.predict(query, questions_to_use)

        # Extract intent
        intent_str = result["answers"]["intent"]["choice"]

        # Extract confidence
        confidence_level = result["answers"]["confidence"]["score"]
        confidence = (confidence_level + 1) / 3.0  # Convert 0-2 score to 0-1

        # Convert string to SearchIntent enum
        intent = SearchIntent(intent_str)

        return intent, min(confidence, 0.99)  # Cap at 0.99

    except Exception as e:
        print(f"Laya inference error: {e}. Falling back to keyword-based classifier.")
        from app.inference import classify_intent
        return classify_intent(query)


def improve_questions_with_llm(
    questions: Dict[str, Any],
    iteration: int,
    previous_accuracy: float
) -> Dict[str, Any]:
    """
    Use an LLM to improve the question criteria descriptions.

    Args:
        questions: Current question definitions
        iteration: Current loop iteration number
        previous_accuracy: Accuracy from previous iteration

    Returns:
        Improved question definitions
    """
    try:
        # Simulate LLM-based improvement
        # In real scenario, this would call an LLM API
        improved = json.loads(json.dumps(questions))  # Deep copy

        # Example improvements based on iteration
        if iteration == 2:
            # Improve criteria based on feedback
            improved["intent"]["criteria"][SearchIntent.COMMERCIAL] = (
                "User researches before buying - includes best products, top rated, "
                "reviews, comparisons, vs battles, recommendations, alternatives, product details"
            )

        elif iteration == 3:
            # Further refinement
            improved["intent"]["criteria"][SearchIntent.TRANSACTIONAL] = (
                "User ready to buy/download/book now - includes buy, purchase, order, checkout, "
                "download, book, reserve, subscribe, rent, pricing, deals, discounts, coupons"
            )

        elif iteration == 4:
            # Refine navigational
            improved["intent"]["criteria"][SearchIntent.NAVIGATIONAL] = (
                "User going to specific site/app - includes login, sign in, account, official site, "
                "app download, twitter/facebook/instagram/github pages, contact page"
            )

        elif iteration == 5:
            # Enhance informational with better patterns
            improved["intent"]["criteria"][SearchIntent.INFORMATIONAL] = (
                "User wants to learn/understand - how to guides, what is definitions, why explanations, "
                "tutorials, educational guides, research papers, news, information resources"
            )

        return improved

    except Exception as e:
        print(f"LLM improvement error: {e}. Returning original questions.")
        return questions


def create_improved_questions(iteration: int) -> Dict[str, Any]:
    """
    Create iteratively improved question definitions.

    Args:
        iteration: Current loop iteration (1-10)

    Returns:
        Question definitions optimized for this iteration
    """
    # Start with defaults
    questions = json.loads(json.dumps(DEFAULT_QUESTIONS))

    # Apply LLM-based improvements based on iteration
    if iteration > 1:
        questions = improve_questions_with_llm(
            questions,
            iteration,
            previous_accuracy=0.0  # Would be passed from previous iteration
        )

    return questions
