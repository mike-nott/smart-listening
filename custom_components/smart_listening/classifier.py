"""Classifier backends: does overheard speech need an answer from the assistant?

Each backend answers "choice" questions and returns, per question, a probability for
each choice: {"ms": 67, "answers": {"q2": {"yes": 0.93, "no": 0.07}}}, or
{"ms": ..., "error": "..."} on failure. Callers never retry.

The question keys, wording and order are the ones the default thresholds were measured
with on Clef (2026-11-03/04). Clef scores shift when keys, field names or the order of
questions change, so keep them as they are. Jev has its own wording (see JevClassifier).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CLASSIFIER_CLEF,
    CLASSIFIER_CLEF_CLOUD,
    CLASSIFIER_JEV,
    CLASSIFIER_OPENAI,
    CLASSIFIER_TIMEOUT,
    JEV_MODEL,
    JEV_URL,
)


ADDRESS = "q2"
KIND = "q3"
FOLLOWUP = "f"
JEV_ADDRESS = "n1"


def address_questions(assistant: str) -> dict[str, Any]:
    """Is the speech a request for the assistant? (the one the thresholds use)"""
    return {
        ADDRESS: {
            "type": "choice",
            "question": f"A smart speaker overheard this speech. Is someone giving the voice assistant named {assistant} a request or question to act on?",
            "criteria": {
                "yes": f"Yes, it is a request or question for the voice assistant {assistant}",
                "no": "No, it is conversation, TV, radio, or talk about the assistant",
            },
        },
    }


def kind_question(assistant: str) -> dict[str, Any]:
    """Diagnostic only (logged, not used for decisions): command, mention or other."""
    return {
        KIND: {
            "type": "choice",
            "question": f"This is a transcript of speech overheard by a smart speaker in a family home. The home's voice assistant is called {assistant}. Which best describes the speech?",
            "criteria": {
                "command": f"The speaker is talking directly to the voice assistant {assistant}, asking it to do something or answer a question",
                "mention": f"{assistant} is mentioned or talked about, or a character or person with a similar name is spoken to, but nobody is asking the voice assistant to do anything",
                "other": "Ordinary conversation, TV, radio or background speech not directed at the voice assistant",
            },
        },
    }


FOLLOWUP_QUESTIONS = {
    FOLLOWUP: {
        "type": "choice",
        "question": "A voice assistant just replied to someone in a family home, and the smart speaker then heard the new speech below, with no wake word. Is the new speech a follow-up said to the assistant?",
        "criteria": {
            "followup": "Yes, it continues the conversation with the assistant: a correction, an addition or a related request or question for it",
            "not": "No, it is said to another person, is a reaction, or is TV, radio or background speech",
        },
    }
}


class Classifier:
    """Base class."""

    # Ask the diagnostic "kind" question in the same request. On Clef the thresholds were
    # measured with it included; elsewhere it would be a second model call, so skip it.
    asks_kind = False

    def __init__(self, hass: HomeAssistant, url: str, model: str = "", api_key: str = "") -> None:
        self.hass = hass
        self.url = url.strip()
        self.model = model.strip()
        self.api_key = api_key.strip()

    async def address(self, text: str, assistant: str) -> dict[str, Any]:
        """Probability ("p") that the speech is a request for the assistant."""
        questions = address_questions(assistant)
        if self.asks_kind:
            # Kind first: on Clef this order let 1/28 named side talk through at 0.8,
            # against 5/28 the other way round (2026-11-04 test set).
            questions = kind_question(assistant) | questions
        rec = await self.ask(f'Transcript: "{text}"', questions)
        rec["p"] = ((rec.get("answers") or {}).get(ADDRESS) or {}).get("yes", 0.0)
        return rec

    async def followup(self, said: str, text: str, assistant: str) -> dict[str, Any]:
        """Probability ("p") that the speech continues the conversation after a reply."""
        rec = await self.ask(f'Assistant said: "{said[:500]}"\nNew speech: "{text}"', FOLLOWUP_QUESTIONS)
        rec["p"] = ((rec.get("answers") or {}).get(FOLLOWUP) or {}).get("followup", 0.0)
        return rec

    async def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        try:
            answers = await self._ask(state, questions)
            return {"ms": round((time.monotonic() - start) * 1000), "answers": answers}
        except Exception as err:  # noqa: BLE001  (never retry; the server may be shared)
            return {"ms": round((time.monotonic() - start) * 1000), "error": repr(err)[:200]}

    async def _ask(self, state: Any, questions: dict[str, Any]) -> dict[str, dict[str, float]]:
        raise NotImplementedError

    async def check(self) -> None:
        """Raise if the server cannot be reached (used by the setup screen)."""
        raise NotImplementedError


class SystemOneClassifier(Classifier):
    """A System One decision server: POST <url> with a state and choice questions."""

    asks_kind = True
    # Field that carries the question text.
    text_field = "instructions"

    def _payload(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        qs = {}
        for k, q in questions.items():
            qs[k] = {"type": q["type"], self.text_field: q["question"]}
            if q.get("criteria") is not None:
                qs[k]["criteria"] = q["criteria"]
        return {"model": self.model, "state": state, "questions": qs}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _ask(self, state: Any, questions: dict[str, Any]) -> dict[str, dict[str, float]]:
        resp = await get_async_client(self.hass).post(
            self.url, json=self._payload(state, questions), headers=self._headers(), timeout=CLASSIFIER_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        if "answers" not in data and isinstance(data.get("result"), dict):
            data = data["result"]  # Cloudflare's REST API wraps replies in "result"
        answers: dict[str, dict[str, float]] = {}
        for k, v in (data.get("answers") or {}).items():
            # Choice answers carry "probabilities"; yes/no (noul) answers a single "noul".
            answers[k] = v.get("probabilities") or ({"noul": v["noul"]} if "noul" in v else {})
        return answers


class ClefClassifier(SystemOneClassifier):
    """Self-hosted Clef (Cloudflare/clef or a quantised build, served with SGLang or another
    System One server), with an empty "images" list. The question text goes in
    `text_field`: the documented "instructions" for new entries, or "question" for entries
    tuned before 0.2.0. Clef scores the two differently, so each has its own threshold."""

    def __init__(
        self, hass: HomeAssistant, url: str, model: str = "", api_key: str = "", text_field: str = "question"
    ) -> None:
        super().__init__(hass, url, model or "clef", api_key)
        self.text_field = text_field

    def _payload(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return {**super()._payload(state, questions), "images": []}

    async def check(self) -> None:
        base = self.url.split("/v1/")[0].rstrip("/")
        resp = await get_async_client(self.hass).get(f"{base}/health", timeout=10)
        resp.raise_for_status()


class ClefCloudClassifier(SystemOneClassifier):
    """Clef on Cloudflare Workers AI. The URL holds the account id (built by the setup
    screen) and the key is a Cloudflare API token with Workers AI access. Uses the
    documented "instructions" field, which is what its default threshold was measured on."""

    def __init__(self, hass: HomeAssistant, url: str, model: str = "", api_key: str = "") -> None:
        super().__init__(hass, url, model or "clef", api_key)

    async def check(self) -> None:
        # Workers AI has no free endpoint that proves the token can run the model, so
        # make one tiny decision (a few dozen input tokens).
        resp = await get_async_client(self.hass).post(
            self.url,
            json={"model": self.model, "state": "ok",
                  "questions": {"ok": {"type": "noul", "instructions": "Is this a test?"}}},
            headers=self._headers(),
            timeout=CLASSIFIER_TIMEOUT,
        )
        resp.raise_for_status()


class JevClassifier(SystemOneClassifier):
    """TypeSafe Jev, the hosted System One model (needs an API key)."""

    def __init__(self, hass: HomeAssistant, url: str, model: str = "", api_key: str = "") -> None:
        super().__init__(hass, url or JEV_URL, model or JEV_MODEL, api_key)

    async def check(self) -> None:
        base = self.url.split("/v1/")[0].rstrip("/")
        resp = await get_async_client(self.hass).get(f"{base}/v1/models", headers=self._headers(), timeout=10)
        resp.raise_for_status()

    # Measured on jev-1.14.0 (2026-11-04, same test set as Clef): the yes/no question with
    # the setting in a structured state answered 37/38 requests with 1/28 named side
    # talk at 0.7; follow-ups 7/7 with 0/5 false at 0.5-0.8.

    async def address(self, text: str, assistant: str) -> dict[str, Any]:
        state = {
            "transcript": text,
            "setting": f"Speech overheard by a smart speaker in a family home, transcribed automatically. "
            f"The home's voice assistant is called {assistant}. Transcripts can contain recognition "
            "errors, so the assistant's name may be misspelt.",
        }
        questions = kind_question(assistant) | {
            JEV_ADDRESS: {
                "type": "noul",
                "question": f"Is the speaker addressing the voice assistant {assistant} directly, asking it to do something or to answer a question?",
                "criteria": {
                    "true": f"The speech is said to {assistant}: a command, a request or a question for it, even a short one such as 'stop' or 'goodnight'",
                    "false": f"The speech is said to another person, is TV, radio or background talk, or only mentions {assistant} without asking it for anything",
                },
            }
        }
        rec = await self.ask(state, questions)
        rec["p"] = ((rec.get("answers") or {}).get(JEV_ADDRESS) or {}).get("noul", 0.0)
        return rec

    async def followup(self, said: str, text: str, assistant: str) -> dict[str, Any]:
        state = {
            "assistant_said": said[:500],
            "new_speech": text,
            "setting": f"A smart speaker in a family home. Its voice assistant, {assistant}, has just replied, "
            "and the speaker then heard the new speech with no wake word. Transcripts are automatic "
            "and can contain errors.",
        }
        rec = await self.ask(state, FOLLOWUP_QUESTIONS)
        rec["p"] = ((rec.get("answers") or {}).get(FOLLOWUP) or {}).get("followup", 0.0)
        return rec


_JSON = re.compile(r"\{.*\}", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


class OpenAIClassifier(Classifier):
    """Any OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp, LM Studio, cloud APIs).

    The model picks one choice and gives its confidence; the other choices share the rest.
    """

    def _endpoint(self) -> str:
        url = self.url.rstrip("/")
        return url if url.endswith("/chat/completions") else f"{url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _ask(self, state: str, questions: dict[str, Any]) -> dict[str, dict[str, float]]:
        client = get_async_client(self.hass)
        answers: dict[str, dict[str, float]] = {}
        for key, q in questions.items():
            choices = list(q["criteria"])
            listing = "\n".join(f'- "{c}": {desc}' for c, desc in q["criteria"].items())
            resp = await client.post(
                self._endpoint(),
                headers=self._headers(),
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 60,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You classify speech overheard by a smart speaker. Reply with JSON only, "
                            'in the form {"choice": "<one of the choice keys>", "confidence": <number from 0 to 1>}.',
                        },
                        {"role": "user", "content": f"{state}\n\n{q['question']}\n\nChoices:\n{listing}"},
                    ],
                },
                timeout=CLASSIFIER_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            match = _JSON.search(_THINK.sub("", content))
            if not match:
                raise ValueError(f"no JSON in reply: {content[:80]!r}")
            parsed = json.loads(match.group(0))
            choice = str(parsed.get("choice", "")).strip().strip('"').lower()
            if choice not in choices:
                raise ValueError(f"unknown choice {choice!r}")
            conf = min(max(float(parsed.get("confidence", 1.0)), 0.0), 1.0)
            rest = (1.0 - conf) / max(len(choices) - 1, 1)
            answers[key] = {c: round(conf if c == choice else rest, 4) for c in choices}
        return answers

    async def check(self) -> None:
        base = self._endpoint().rsplit("/chat/completions", 1)[0]
        resp = await get_async_client(self.hass).get(f"{base}/models", headers=self._headers(), timeout=10)
        resp.raise_for_status()


def make_classifier(
    hass: HomeAssistant, kind: str, url: str, model: str, api_key: str, clef_field: str = "question"
) -> Classifier | None:
    """The configured backend, or None for name-only mode."""
    if kind == CLASSIFIER_CLEF:
        return ClefClassifier(hass, url, model, api_key, clef_field)
    if kind == CLASSIFIER_CLEF_CLOUD:
        return ClefCloudClassifier(hass, url, model, api_key)
    if kind == CLASSIFIER_JEV:
        return JevClassifier(hass, url, model, api_key)
    if kind == CLASSIFIER_OPENAI:
        return OpenAIClassifier(hass, url, model, api_key)
    return None
