"""Voice PE timers, handled by Smart Listening without the conversation agent.

An accepted request that mentions a time is put to the classifier as one choice question
(what does the speaker want?). A timer action with a high enough score is run as Home
Assistant's own timer intent on the Voice PE that heard it, with the duration and timer
name read here. Everything else goes to the conversation agent as usual. English only.

The question and criteria below were measured on Clef (2026-11-05, 60 labelled requests,
"instructions" field): at 0.6, no wrong timer actions and one timer request left to the
agent. Keep their wording as it is.
"""

from __future__ import annotations

import re
from typing import Any

from homeassistant.helpers import intent

# Requests without any of these never reach the classifier (no extra delay).
TIMER_WORDS = re.compile(
    r"\b(timers?|minutes?|mins?|seconds?|secs?|hours?|hrs?|countdown|count down|"
    r"how long|time left|half an hour|quarter of an hour)\b",
    re.IGNORECASE,
)

TIMER_KEY = "t"
TIMER_QUESTION = (
    "A voice assistant in a family home was asked the request below. "
    "What does the speaker want the voice assistant to do?"
)
TIMER_CRITERIA = {
    "start": "Start a new countdown timer for a length of time",
    "cancel": "Cancel or stop a timer that is running",
    "add": "Add time to a timer that is running",
    "remove": "Take time off a timer that is running",
    "pause": "Pause a timer that is running",
    "resume": "Resume a paused timer",
    "status": "Ask how much time is left on a timer",
    "reminder": "Be reminded of something specific at a later time, with a message or task to remember",
    "other": "Anything else, including questions or talk that only mention a time or a timer",
}
TIMER_ACTIONS = ("start", "cancel", "add", "remove", "pause", "resume", "status")
NEEDS_DURATION = ("start", "add", "remove")

ACTION_INTENTS = {
    "start": intent.INTENT_START_TIMER,
    "cancel": intent.INTENT_CANCEL_TIMER,
    "cancel_all": intent.INTENT_CANCEL_ALL_TIMERS,
    "add": intent.INTENT_INCREASE_TIMER,
    "remove": intent.INTENT_DECREASE_TIMER,
    "pause": intent.INTENT_PAUSE_TIMER,
    "resume": intent.INTENT_UNPAUSE_TIMER,
    "status": intent.INTENT_TIMER_STATUS,
}


def mentions_time(text: str) -> bool:
    return bool(TIMER_WORDS.search(text))


def timer_state(text: str) -> str:
    """The classifier state, as measured."""
    return f'Request: "{text}"'


# Durations ------------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_UNIT = r"(hours?|hrs?|minutes?|mins?|seconds?|secs?)"
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}
_FIXED = [
    (r"\bthree quarters of an hour\b", "45 minutes"),
    (r"\b(?:a )?quarter of an hour\b|\bquarter hour\b", "15 minutes"),
    (r"\bhalf an hour\b|\bhalf a hour\b|\bhalf hour\b", "30 minutes"),
    (r"\bhalf a minute\b", "30 seconds"),
]


def _words_to_digits(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in _TENS:
            value = _TENS[word]
            if i + 1 < len(tokens) and tokens[i + 1] in _ONES and 0 < _ONES[tokens[i + 1]] < 10:
                value += _ONES[tokens[i + 1]]
                i += 1
            out.append(str(value))
        elif word in _ONES:
            out.append(str(_ONES[word]))
        elif word in ("a", "an") and i + 1 < len(tokens) and re.fullmatch(_UNIT, tokens[i + 1]):
            out.append("1")  # "an hour", "a minute"
        else:
            out.append(word)
        i += 1
    return " ".join(out)


def parse_duration(text: str) -> int | None:
    """Total seconds named in the request, or None. "an hour and a half" -> 5400."""
    s = text.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9.\s]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    for pattern, replacement in _FIXED:
        s = re.sub(pattern, replacement, s)
    s = _words_to_digits(" ".join(s.split()))
    total = 0.0
    for number, half_before, unit, half_after in re.findall(
        r"(\d+(?:\.\d+)?)\s+(and a half\s+)?(?:(?:more|extra|additional)\s+)?" + _UNIT + r"\b(\s+and a half)?", s
    ):
        value = float(number) + (0.5 if half_before or half_after else 0)
        total += value * _UNIT_SECONDS[unit[0]]
    seconds = round(total)
    return seconds if seconds > 0 else None


def slots_for(seconds: int) -> dict[str, dict[str, int]]:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    slots = {}
    if hours:
        slots["hours"] = {"value": hours}
    if minutes:
        slots["minutes"] = {"value": minutes}
    if secs or not slots:
        slots["seconds"] = {"value": secs}
    return slots


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def say_duration(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [_plural(n, u) for n, u in ((hours, "hour"), (minutes, "minute"), (secs, "second")) if n]
    if not parts:
        return "0 seconds"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# Names and "all" ------------------------------------------------------------------------

_NOT_NAMES = {
    "a", "an", "the", "my", "this", "that", "your", "our", "new", "another", "all", "every",
    "minute", "minutes", "second", "seconds", "hour", "hours", "min", "mins", "sec", "secs",
    "set", "start", "cancel", "stop", "pause", "resume", "unpause", "delete", "clear", "on",
    "off", "of", "for", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
}


def parse_name(text: str) -> str | None:
    """The word naming the timer: "set a pasta timer" -> "pasta"."""
    match = re.search(r"\b([a-z']+)\s+timers?\b", text.lower())
    if not match:
        return None
    word = match.group(1).strip("'")
    if word.endswith("'s"):
        word = word[:-2]
    if not word or word in _NOT_NAMES or word in _ONES or word in _TENS:
        return None
    return word


def is_all(text: str) -> bool:
    return bool(re.search(r"\b(all (of )?(the |my )?timers|every timer|all timers)\b", text.lower()))


# What is said ---------------------------------------------------------------------------


def confirmation(action: str, seconds: int | None, name: str | None, speech_slots: dict[str, Any]) -> str:
    named = f"{name.capitalize()} timer" if name else "Timer"
    the = f"the {name} timer" if name else "the timer"
    if action == "start":
        return f"{named} set for {say_duration(seconds or 0)}."
    if action == "cancel":
        return f"{named} cancelled."
    if action == "cancel_all":
        count = int(speech_slots.get("canceled", 0))
        if count == 0:
            return "There were no timers to cancel."
        return "Cancelled the timer." if count == 1 else f"Cancelled {count} timers."
    if action == "add":
        return f"Added {say_duration(seconds or 0)} to {the}."
    if action == "remove":
        return f"Took {say_duration(seconds or 0)} off {the}."
    if action == "pause":
        return f"{named} paused."
    if action == "resume":
        return f"{named} resumed."
    if action == "status":
        return status_speech(speech_slots.get("timers") or [])
    return "Done."


def _left(timer: dict[str, Any]) -> str:
    seconds = (
        int(timer.get("rounded_hours_left", 0)) * 3600
        + int(timer.get("rounded_minutes_left", 0)) * 60
        + int(timer.get("rounded_seconds_left", 0))
    )
    return say_duration(seconds) if seconds else "less than a second"


def status_speech(timers: list[dict[str, Any]]) -> str:
    if not timers:
        return "There are no timers running."
    if len(timers) == 1:
        timer = timers[0]
        the = f"the {timer['name']} timer" if timer.get("name") else "the timer"
        if not timer.get("is_active", True):
            return f"{the[0].upper()}{the[1:]} is paused with {_left(timer)} left."
        return f"{_left(timer)[0].upper()}{_left(timer)[1:]} left on {the}."
    parts = []
    for timer in timers:
        label = timer.get("name") or "a timer"
        paused = ", paused" if not timer.get("is_active", True) else ""
        parts.append(f"{label}, {_left(timer)} left{paused}")
    return f"You have {len(timers)} timers: " + "; ".join(parts) + "."
