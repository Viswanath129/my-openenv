"""
InboxIQ Email Classifier v3.0
- TF-IDF + Multinomial Naive Bayes for spam/ham classification
- Keyword + heuristic sentiment analysis
- Urgency detection from content signals
- Online accuracy tracking from agent feedback
- Trains on SpamAssassin dataset at startup
"""

import re
import math
import os
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Lightweight TF-IDF + Naive Bayes (no sklearn hard dependency)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
        "re",
        "subject",
        "http",
        "www",
    ]
)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with stopword removal."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior: Dict[int, float] = {}
        self.feature_log_prob: Dict[int, Dict[str, float]] = {}
        self.vocab: set = set()
        self._is_trained = False

    def fit(self, texts: List[str], labels: List[int]) -> "NaiveBayesClassifier":
        class_counts: Dict[int, int] = Counter(labels)
        total = len(labels)
        word_counts: Dict[int, Counter] = defaultdict(Counter)
        class_totals: Dict[int, int] = defaultdict(int)

        for text, label in zip(texts, labels):
            tokens = _tokenize(text)
            word_counts[label].update(tokens)
            class_totals[label] += len(tokens)
            self.vocab.update(tokens)

        V = len(self.vocab)

        for cls in class_counts:
            self.class_log_prior[cls] = math.log(class_counts[cls] / total)
            self.feature_log_prob[cls] = {}
            denom = class_totals[cls] + self.alpha * V
            for word in self.vocab:
                self.feature_log_prob[cls][word] = math.log(
                    (word_counts[cls].get(word, 0) + self.alpha) / denom
                )

        self._is_trained = True
        print(
            f"[CLASSIFIER] Trained on {total} samples | vocab={V} | classes={dict(class_counts)}"
        )
        return self

    def predict_proba(self, text: str) -> Dict[int, float]:
        """Return log-probabilities per class."""
        tokens = _tokenize(text)
        scores = {}
        for cls in self.class_log_prior:
            score = self.class_log_prior[cls]
            for token in tokens:
                if token in self.feature_log_prob[cls]:
                    score += self.feature_log_prob[cls][token]
            scores[cls] = score
        return scores

    def predict(self, text: str) -> int:
        scores = self.predict_proba(text)
        return max(scores, key=scores.get) if scores else 0

    def predict_with_confidence(self, text: str) -> Tuple[int, float]:
        """Return (predicted_class, confidence 0-1)."""
        scores = self.predict_proba(text)
        if not scores:
            return 0, 0.5
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}
        pred = max(probs, key=probs.get)
        return pred, probs[pred]

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

_AGGRESSIVE_WORDS = frozenset(
    [
        "urgent",
        "immediately",
        "asap",
        "demand",
        "threaten",
        "angry",
        "furious",
        "unacceptable",
        "terrible",
        "worst",
        "horrible",
        "disgusting",
        "outraged",
        "complaint",
        "lawsuit",
        "warning",
        "final notice",
        "overdue",
        "penalty",
        "critical",
        "emergency",
        "deadline",
        "escalate",
        "disappointed",
        "failure",
        "incompetent",
        "ridiculous",
        "fraud",
        "scam",
        "stolen",
        "hack",
        "breach",
        "attack",
        "alert",
        "danger",
        "risk",
        "violation",
        "suspended",
        "blocked",
    ]
)

_PROFESSIONAL_WORDS = frozenset(
    [
        "regards",
        "sincerely",
        "dear",
        "please",
        "kindly",
        "appreciate",
        "meeting",
        "schedule",
        "project",
        "update",
        "review",
        "report",
        "proposal",
        "agreement",
        "contract",
        "invoice",
        "quarterly",
        "budget",
        "milestone",
        "deliverable",
        "stakeholder",
        "collaborate",
        "strategy",
        "objective",
        "compliance",
        "policy",
        "procedure",
        "documentation",
        "attached",
        "forwarding",
        "following",
        "agenda",
        "minutes",
    ]
)

_CASUAL_WORDS = frozenset(
    [
        "hey",
        "hi",
        "hello",
        "thanks",
        "cool",
        "awesome",
        "great",
        "nice",
        "lol",
        "haha",
        "btw",
        "fyi",
        "gonna",
        "wanna",
        "gotta",
        "yeah",
        "nah",
        "sure",
        "okay",
        "ok",
        "yep",
        "nope",
        "cheers",
        "mate",
        "dude",
        "bro",
        "sup",
        "chill",
        "hang",
        "weekend",
    ]
)


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Determine sentiment: Aggressive / Professional / Casual."""
    text_lower = text.lower()
    scores = {
        "Aggressive": sum(1 for w in _AGGRESSIVE_WORDS if w in text_lower),
        "Professional": sum(1 for w in _PROFESSIONAL_WORDS if w in text_lower),
        "Casual": sum(1 for w in _CASUAL_WORDS if w in text_lower),
    }
    total = sum(scores.values()) or 1
    best = max(scores, key=scores.get)
    confidence = scores[best] / total if total > 0 else 0.33

    if scores[best] == 0:
        return "Professional", 0.33  # Default
    return best, round(min(confidence, 1.0), 2)


# ---------------------------------------------------------------------------
# Urgency Detector
# ---------------------------------------------------------------------------

_URGENCY_HIGH = [
    "urgent",
    "asap",
    "immediately",
    "critical",
    "emergency",
    "deadline",
    "action required",
    "time sensitive",
    "expiring",
    "final notice",
    "security alert",
    "password reset",
    "account suspended",
    "breach",
]

_URGENCY_LOW = [
    "newsletter",
    "unsubscribe",
    "promotion",
    "sale",
    "discount",
    "offer",
    "weekly digest",
    "monthly update",
    "no reply needed",
    "fyi",
]


def detect_urgency(text: str) -> str:
    """Returns HIGH / MEDIUM / LOW."""
    text_lower = text.lower()
    high = sum(1 for s in _URGENCY_HIGH if s in text_lower)
    low = sum(1 for s in _URGENCY_LOW if s in text_lower)

    if high >= 2 or (high == 1 and low == 0):
        return "HIGH"
    if low >= 2 or (low == 1 and high == 0):
        return "LOW"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Unified Pipeline
# ---------------------------------------------------------------------------


class EmailClassifier:
    """
    Full pipeline: spam detection + sentiment + urgency + online learning.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        self.nb = NaiveBayesClassifier(alpha=1.0)
        self._accuracy: float = 0.0
        self._total_classified: int = 0
        self._correct_predictions: int = 0
        self._reward_history: List[float] = []

        if dataset_path and os.path.exists(dataset_path):
            self._train_from_csv(dataset_path)

    def _train_from_csv(self, path: str):
        """Load SpamAssassin CSV and train NB classifier."""
        texts, labels = [], []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("text") and row.get("target") is not None:
                        texts.append(row["text"])
                        labels.append(int(row["target"]))

            if len(texts) < 10:
                print(f"[CLASSIFIER] Dataset too small ({len(texts)}), skipping.")
                return

            # 80/20 split
            split = int(len(texts) * 0.8)
            train_t, test_t = texts[:split], texts[split:]
            train_l, test_l = labels[:split], labels[split:]

            self.nb.fit(train_t, train_l)

            correct = sum(1 for t, l in zip(test_t, test_l) if self.nb.predict(t) == l)
            self._accuracy = correct / len(test_l) if test_l else 0
            print(
                f"[CLASSIFIER] Test accuracy: {self._accuracy:.2%} on {len(test_l)} samples"
            )

        except Exception as e:
            print(f"[CLASSIFIER ERROR] Training failed: {e}")

    def classify(self, text: str, subject: str = "") -> Dict:
        """Full classification: type, urgency, sentiment, confidence, spam_score."""
        full_text = f"{subject} {text}" if subject else text

        if self.nb.is_trained:
            pred, confidence = self.nb.predict_with_confidence(full_text)
            is_spam = pred == 1
            spam_score = confidence if is_spam else (1 - confidence)
        else:
            is_spam, spam_score = self._keyword_spam_check(full_text)
            confidence = spam_score

        sentiment, sent_conf = analyze_sentiment(full_text)
        urgency = detect_urgency(full_text)

        if is_spam:
            email_type = "SPAM"
            urgency = "LOW"
        elif urgency == "HIGH":
            email_type = "SUPPORT"
        else:
            # Sub-classify work vs support
            support_kw = [
                "help",
                "issue",
                "problem",
                "support",
                "ticket",
                "bug",
                "error",
            ]
            if any(kw in full_text.lower() for kw in support_kw):
                email_type = "SUPPORT"
            else:
                email_type = "WORK"

        self._total_classified += 1

        return {
            "type": email_type,
            "urgency": urgency,
            "sentiment": sentiment,
            "sentiment_confidence": sent_conf,
            "spam_score": round(spam_score, 3),
            "confidence": round(confidence, 3),
            "is_spam": is_spam,
        }

    def _keyword_spam_check(self, text: str) -> Tuple[bool, float]:
        """Fallback heuristic."""
        spam_keywords = [
            "offer",
            "unsubscribe",
            "promo",
            "claim",
            "prize",
            "discount",
            "winner",
            "congratulations",
            "click here",
            "free",
            "limited time",
            "act now",
            "buy now",
            "order now",
            "credit card",
            "earn money",
        ]
        text_lower = text.lower()
        hits = sum(1 for kw in spam_keywords if kw in text_lower)
        score = min(hits / 3, 1.0)
        return score > 0.5, round(score, 3)

    def record_reward(self, reward: float):
        """Track reward for performance monitoring."""
        self._reward_history.append(reward)

    def update_feedback(self, predicted_spam: bool, actual_spam: bool):
        """Online accuracy from user corrections."""
        if predicted_spam == actual_spam:
            self._correct_predictions += 1

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def avg_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        return sum(self._reward_history[-50:]) / len(self._reward_history[-50:])

    @property
    def stats(self) -> Dict:
        return {
            "model_trained": self.nb.is_trained,
            "accuracy": round(self._accuracy, 4),
            "total_classified": self._total_classified,
            "correct_predictions": self._correct_predictions,
            "vocab_size": len(self.nb.vocab) if self.nb.is_trained else 0,
            "avg_reward_last_50": round(self.avg_reward, 2),
            "total_rewards_tracked": len(self._reward_history),
        }


# Global singleton
brain = EmailClassifier()
