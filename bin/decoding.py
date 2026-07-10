"""Signal-to-intent decoding helpers shared across BCI device controllers.

These primitives sit between the windowed EEG readings produced by ``bin.eeg``
and the hardware actions issued by a device controller. They are deliberately
dependency-free (stdlib only) so they stay fast and fully unit-testable:

* :class:`Decision` — a decoded intent plus its confidence and a human reason.
* :func:`gate_by_confidence` — turn class probabilities into a :class:`Decision`,
  falling back to a reject/idle intent when the top probability is too low.
* :class:`VoteWindow` — sliding-window majority vote that adds temporal
  hysteresis so a single noisy window cannot move the hardware.

The intent strings are opaque to this module: each device chooses its own
vocabulary (e.g. ``"base_left"`` for the arm, ``"left"`` for the car) and only
has to agree on which string means "do nothing" (the idle/reject intent).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Sequence


@dataclass
class Decision:
    """One decoded control intent.

    ``intent`` is an opaque string chosen by the caller. ``confidence`` is in
    ``[0, 1]`` when it comes from a probabilistic source, or ``1.0`` for
    rule-based decisions. ``reason`` is a short, log-friendly explanation.
    """

    intent: str
    confidence: float = 1.0
    reason: str = ""


def gate_by_confidence(
    probabilities: Sequence[float],
    labels: Sequence[str],
    threshold: float,
    *,
    reject_intent: str = "rest",
) -> Decision:
    """Return the most likely label only when it clears ``threshold``.

    Below the threshold (or for an empty distribution) ``reject_intent`` is
    returned instead, so an uncertain classifier stays idle rather than
    emitting a random direction. This is the "rest band" that a bare ``argmax``
    lacks.
    """

    if not probabilities:
        return Decision(reject_intent, 0.0, "empty distribution")
    best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    confidence = float(probabilities[best_index])
    if confidence < threshold:
        return Decision(
            reject_intent,
            confidence,
            f"confidence {confidence:.2f} < {threshold:.2f}",
        )
    label = labels[best_index] if best_index < len(labels) else reject_intent
    return Decision(label, confidence, f"confidence {confidence:.2f} >= {threshold:.2f}")


class VoteWindow:
    """Sliding-window majority vote with an explicit idle intent.

    Each :meth:`push` appends one raw intent and returns the *committed*
    intent. A non-idle intent is only committed once it wins at least
    ``min_votes`` of the last ``size`` observations; otherwise the idle intent
    is returned. This debounces per-window classifier/rule noise: a lone
    spurious window cannot flip the hardware, while a genuinely sustained
    intent commits after ``min_votes`` windows.

    Set ``size=1, min_votes=1`` to disable smoothing (every window acts).
    """

    def __init__(self, size: int = 3, min_votes: int = 2, *, idle_intent: str = "rest") -> None:
        self.size = max(1, int(size))
        self.min_votes = max(1, int(min_votes))
        self.idle_intent = idle_intent
        self._buffer: Deque[str] = deque(maxlen=self.size)

    def push(self, intent: str) -> str:
        self._buffer.append(intent)
        counts = Counter(self._buffer)
        best_intent = self.idle_intent
        best_votes = 0
        for candidate, votes in counts.items():
            if candidate == self.idle_intent:
                continue
            if votes > best_votes:
                best_intent, best_votes = candidate, votes
        if best_votes >= self.min_votes:
            return best_intent
        return self.idle_intent

    def reset(self) -> None:
        self._buffer.clear()
