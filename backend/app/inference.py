"""Inference module for System 1 model."""

import random
from typing import Tuple
from app.tasks.intent.labels import SearchIntent

# Simple keyword-based heuristics for intent classification
INTENT_KEYWORDS = {
    SearchIntent.INFORMATIONAL: [
        "how", "what", "why", "when", "where", "tutorial", "guide", "explain",
        "definition", "meaning", "learn", "understand", "research"
    ],
    SearchIntent.NAVIGATIONAL: [
        "login", "sign in", "account", "official", "site:", "facebook", "gmail",
        "twitter", "instagram", "linkedin", "github", "homepage", "contact"
    ],
    SearchIntent.COMMERCIAL: [
        "best", "top", "review", "comparison", "vs", "alternative", "recommended",
        "vs vs", "better than", "pros and cons", "verdict"
    ],
    SearchIntent.TRANSACTIONAL: [
        "buy", "purchase", "order", "download", "book", "reserve", "subscribe",
        "rent", "price", "cost", "deal", "discount", "coupon", "free"
    ],
}


def classify_intent(query: str) -> Tuple[SearchIntent, float]:
    """
    Classify search intent based on query keywords.
    Returns (predicted_intent, confidence_score).

    This is a simple heuristic-based classifier.
    In production, replace with actual neural network model.
    """
    query_lower = query.lower().strip()

    # Score each intent based on keyword matches
    intent_scores = {intent: 0 for intent in SearchIntent.all_intents()}

    query_words = query_lower.split()

    for intent, keywords in INTENT_KEYWORDS.items():
        for word in query_words:
            if any(keyword in word for keyword in keywords):
                intent_scores[intent] += 1

    # Find the intent with highest score
    max_score = max(intent_scores.values())

    if max_score == 0:
        # No keywords matched, default to informational with low confidence
        predicted_intent = SearchIntent.INFORMATIONAL
        confidence = 0.3 + random.random() * 0.2  # 0.3-0.5
    else:
        # Find all intents with max score
        max_intents = [intent for intent, score in intent_scores.items() if score == max_score]
        predicted_intent = random.choice(max_intents)

        # Confidence based on score ratio
        total_score = sum(intent_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        confidence = min(0.95, max(0.5, confidence))  # Clip to 0.5-0.95

    return predicted_intent, round(confidence, 3)


def classify_intent_v2(query: str) -> Tuple[SearchIntent, float]:
    """
    Improved version of intent classifier.
    Can be used after Karpathy loop improvements.
    """
    query_lower = query.lower().strip()

    # V2 has more comprehensive patterns
    v2_patterns = {
        SearchIntent.INFORMATIONAL: [
            r"how\s+to", r"what\s+is", r"explain", r"tutorial",
            r"definition", r"guide", r"learn", r"research"
        ],
        SearchIntent.NAVIGATIONAL: [
            r"login|sign\s+in|account", r"official", r"facebook", r"gmail",
            r"twitter", r"instagram", r"linkedin", r"github"
        ],
        SearchIntent.COMMERCIAL: [
            r"best\s+", r"top\s+", r"review", r"comparison", r"vs\.",
            r"recommended", r"alternative", r"better\s+than"
        ],
        SearchIntent.TRANSACTIONAL: [
            r"buy\s+", r"purchase", r"order", r"download", r"book\s+",
            r"reserve", r"subscribe", r"price\s+", r"cost\s+of"
        ],
    }

    intent_scores = {intent: 0 for intent in SearchIntent.all_intents()}

    for intent, patterns in v2_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                intent_scores[intent] += 2

    max_score = max(intent_scores.values())

    if max_score == 0:
        predicted_intent = SearchIntent.INFORMATIONAL
        confidence = 0.4 + random.random() * 0.15
    else:
        max_intents = [intent for intent, score in intent_scores.items() if score == max_score]
        predicted_intent = random.choice(max_intents)

        total_score = sum(intent_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        confidence = min(0.98, max(0.55, confidence))

    return predicted_intent, round(confidence, 3)
