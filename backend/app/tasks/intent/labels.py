"""Intent labels for SERP classification task."""

from enum import Enum

class SearchIntent(str, Enum):
    """SEO search intent taxonomy."""

    INFORMATIONAL = "informational"
    """User seeks information, education, or knowledge (e.g., 'how to', 'what is')."""

    NAVIGATIONAL = "navigational"
    """User seeks to reach a specific website or resource (e.g., 'facebook login', 'gmail')."""

    COMMERCIAL = "commercial"
    """User researches products/services before purchasing (e.g., 'best laptop', 'product reviews')."""

    TRANSACTIONAL = "transactional"
    """User intends to complete a transaction (e.g., 'buy laptop', 'download pdf')."""

    @classmethod
    def all_labels(cls) -> list[str]:
        """Return all intent labels."""
        return [intent.value for intent in cls]

    @classmethod
    def all_intents(cls) -> list["SearchIntent"]:
        """Return all intent enum values."""
        return list(cls)
