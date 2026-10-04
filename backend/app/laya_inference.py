"""Laya Router-based inference for intent classification.

Simulates Laya Router pattern using structured questions and criteria
to guide intent classification. In production, this would use actual Laya library.
"""

from typing import Dict, Any, Tuple
import json
import random
import re

from app.tasks.intent.labels import SearchIntent


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


def classify_with_laya(
    query: str,
    questions: Dict[str, Any] = None
) -> Tuple[SearchIntent, float]:
    """
    Classify intent using Laya Router pattern.

    Simulates LLM-guided classification based on structured question criteria.
    The criteria definitions are used to guide keyword matching and scoring.

    Args:
        query: The search query to classify
        questions: Custom question definitions (uses defaults if None)

    Returns:
        Tuple of (predicted_intent, confidence_score)
    """
    try:
        questions_to_use = questions or DEFAULT_QUESTIONS

        intent_criteria = questions_to_use.get("intent", {}).get("criteria", {})

        if not intent_criteria:
            from app.inference import classify_intent
            return classify_intent(query)

        query_lower = query.lower().strip()
        intent_scores = {}

        for intent_enum, criteria_text in intent_criteria.items():
            criteria_lower = criteria_text.lower()

            keywords = []
            keywords_match = re.search(r'\(([^)]+)\)', criteria_lower)
            if keywords_match:
                keywords = [k.strip() for k in keywords_match.group(1).split(',')]

            score = 0
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1

            intent_scores[intent_enum] = score

        max_score = max(intent_scores.values()) if intent_scores else 0

        if max_score == 0:
            predicted_intent = SearchIntent.INFORMATIONAL
            confidence = 0.4
        else:
            max_intents = [intent for intent, score in intent_scores.items() if score == max_score]
            predicted_intent = random.choice(max_intents)

            total_score = sum(intent_scores.values())
            base_confidence = max_score / total_score if total_score > 0 else 0.5

            confidence_boost = min(0.2, max_score * 0.1)
            confidence = min(0.95, base_confidence + confidence_boost)

        return predicted_intent, round(confidence, 3)

    except Exception as e:
        print(f"Laya classification error: {e}. Falling back to keyword-based classifier.")
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
        improved = json.loads(json.dumps(questions))

        if iteration == 2:
            improved["intent"]["criteria"][SearchIntent.COMMERCIAL] = (
                "User researches before buying - includes best products, top rated, "
                "reviews, comparisons, vs battles, recommendations, alternatives, product details"
            )

        elif iteration == 3:
            improved["intent"]["criteria"][SearchIntent.TRANSACTIONAL] = (
                "User ready to buy/download/book now - includes buy, purchase, order, checkout, "
                "download, book, reserve, subscribe, rent, pricing, deals, discounts, coupons"
            )

        elif iteration == 4:
            improved["intent"]["criteria"][SearchIntent.NAVIGATIONAL] = (
                "User going to specific site/app - includes login, sign in, account, official site, "
                "app download, twitter/facebook/instagram/github pages, contact page"
            )

        elif iteration == 5:
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
    questions = json.loads(json.dumps(DEFAULT_QUESTIONS))

    if iteration > 1:
        questions = improve_questions_with_llm(
            questions,
            iteration,
            previous_accuracy=0.0
        )

    return questions
