"""Voice PE timer handling (0.4.0): parsing, the classifier question, and the timer flow.

The phrasings are the labelled set the timer threshold was measured on (Clef, 2026-11-05).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from homeassistant.components.intent.timers import (  # noqa: E402
    MultipleTimersMatchedError,
    TimerNotFoundError,
    TimersNotSupportedError,
)
from homeassistant.helpers import intent as ha_intent  # noqa: E402

from custom_components.smart_listening import classifier as cl  # noqa: E402
from custom_components.smart_listening import conversation as slc  # noqa: E402
from custom_components.smart_listening import timers as t  # noqa: E402
from custom_components.smart_listening.const import (  # noqa: E402
    CLASSIFIER_CLEF,
    CONF_CLASSIFIER,
    CONF_NAMES,
    CONF_TARGET_AGENT,
)

# (label, request without the name, duration in seconds or None)
SAMPLES = [
    ("start", "set a timer for 10 seconds", 10),
    ("start", "set a timer for 15 minutes", 900),
    ("start", "set a 5 minute timer", 300),
    ("start", "start a timer for one hour", 3600),
    ("start", "timer for 20 minutes", 1200),
    ("start", "set a pasta timer for 12 minutes", 720),
    ("start", "can you set a timer for fifteen minutes?", 900),
    ("start", "give me a shout in ten minutes", 600),
    ("start", "let me know when 20 minutes is up", 1200),
    ("start", "time 8 minutes for me", 480),
    ("start", "could you count down three minutes please", 180),
    ("start", "put a timer on for half an hour", 1800),
    ("start", "I need a timer, 25 minutes", 1500),
    ("start", "tell me when an hour and a half has passed", 5400),
    ("start", "set me an alarm for 30 minutes from now", 1800),
    ("start", "Timer. 4 minutes.", 240),
    ("start", "can you time the eggs, 6 minutes", 360),
    ("start", "set a timer for ninety seconds", 90),
    ("cancel", "cancel the timer", None),
    ("cancel", "stop the timer", None),
    ("cancel", "cancel my pasta timer", None),
    ("cancel", "never mind the timer, get rid of it", None),
    ("cancel", "turn the timer off", None),
    ("cancel", "delete all the timers", None),
    ("add", "add 5 minutes to the timer", 300),
    ("add", "give the timer another two minutes", 120),
    ("add", "can you put 10 more minutes on it", 600),
    ("remove", "take 2 minutes off the timer", 120),
    ("remove", "remove one minute from the timer", 60),
    ("remove", "knock five minutes off the pasta timer", 300),
    ("pause", "pause the timer", None),
    ("pause", "hold the timer a second", 1),
    ("resume", "resume the timer", None),
    ("resume", "unpause the timer", None),
    ("resume", "carry on with the timer", None),
    ("status", "how much time is left on the timer?", None),
    ("status", "how long left on my timer", None),
    ("status", "what's the timer at?", None),
    ("status", "how long until the pasta's done?", None),
    ("status", "is the timer still going?", None),
]


@pytest.mark.parametrize(("label", "text", "seconds"), SAMPLES)
def test_every_timer_request_passes_the_word_filter(label, text, seconds):
    assert t.mentions_time(text)


@pytest.mark.parametrize(("label", "text", "seconds"), [s for s in SAMPLES if s[0] in t.NEEDS_DURATION])
def test_durations_of_the_measured_requests(label, text, seconds):
    assert t.parse_duration(text) == seconds


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("an hour and a half", 5400),
        ("2 and a half minutes", 150),
        ("one and a half hours", 5400),
        ("1 hour 20 minutes", 4800),
        ("an hour and 20 minutes", 4800),
        ("1.5 hours", 5400),
        ("a minute", 60),
        ("twenty five minutes", 1500),
        ("forty-five seconds", 45),
        ("three quarters of an hour", 2700),
        ("a quarter of an hour", 900),
        ("half a minute", 30),
        ("90 secs", 90),
        ("set a timer for 1 hr", 3600),
        ("add 5 extra minutes", 300),
        ("cancel the timer", None),
        ("give me a shout", None),
    ],
)
def test_duration_edge_cases(text, seconds):
    assert t.parse_duration(text) == seconds


def test_words_without_a_time_skip_the_classifier():
    for text in ("what's the news today?", "who won the Formula One?", "turn on the kitchen lights", "what time is it?"):
        assert not t.mentions_time(text)


def test_names_and_all():
    assert t.parse_name("set a pasta timer for 12 minutes") == "pasta"
    assert t.parse_name("cancel my pasta timer") == "pasta"
    assert t.parse_name("how long until the eggs timer is done") == "eggs"
    assert t.parse_name("knock five minutes off the pasta timer") == "pasta"
    for text in ("set a 5 minute timer", "cancel the timer", "set a timer for 10 seconds", "delete all the timers",
                 "start another timer", "set a five minute timer"):
        assert t.parse_name(text) is None, text
    assert t.is_all("delete all the timers") and t.is_all("cancel all timers") and t.is_all("cancel every timer")
    assert not t.is_all("cancel the timer")


def test_slots_and_speech():
    assert t.slots_for(10) == {"seconds": {"value": 10}}
    assert t.slots_for(5400) == {"hours": {"value": 1}, "minutes": {"value": 30}}
    assert t.say_duration(90) == "1 minute and 30 seconds"
    assert t.say_duration(3661) == "1 hour, 1 minute and 1 second"
    assert t.confirmation("start", 900, None, {}) == "Timer set for 15 minutes."
    assert t.confirmation("start", 720, "pasta", {}) == "Pasta timer set for 12 minutes."
    assert t.confirmation("add", 300, None, {}) == "Added 5 minutes to the timer."
    assert t.confirmation("remove", 120, "pasta", {}) == "Took 2 minutes off the pasta timer."
    assert t.confirmation("cancel_all", None, None, {"canceled": 0}) == "There were no timers to cancel."
    assert t.confirmation("cancel_all", None, None, {"canceled": 2}) == "Cancelled 2 timers."


def _status(name="", active=True, h=0, m=0, s=0):
    return {"name": name, "is_active": active, "rounded_hours_left": h, "rounded_minutes_left": m, "rounded_seconds_left": s}


def test_status_speech():
    assert t.status_speech([]) == "There are no timers running."
    assert t.status_speech([_status(m=5)]) == "5 minutes left on the timer."
    assert t.status_speech([_status("pasta", active=False, m=5)]) == "The pasta timer is paused with 5 minutes left."
    assert (t.status_speech([_status("pasta", m=5), _status("eggs", m=2)])
            == "You have 2 timers: pasta, 5 minutes left; eggs, 2 minutes left.")


# The classifier question -------------------------------------------------------------


class _Resp:
    def __init__(self, answers):
        self._answers = answers

    def raise_for_status(self):
        pass

    def json(self):
        return {"answers": self._answers}


class _Client:
    def __init__(self):
        self.sent = []

    async def post(self, url, **kw):
        self.sent.append(kw["json"])
        probs = {c: 0.01 for c in kw["json"]["questions"][t.TIMER_KEY]["criteria"]}
        probs["start"] = 0.92
        return _Resp({t.TIMER_KEY: {"type": "choice", "probabilities": probs}})


@pytest.mark.asyncio
async def test_timer_question_is_sent_as_measured(monkeypatch):
    client = _Client()
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    # An entry still on the "question" field for its address check uses "instructions" here.
    c = cl.ClefClassifier(None, "http://clef:8080/v1/systemone", text_field="question")
    r = await c.timer_action("set a timer for 10 seconds")
    body = client.sent[0]
    q = body["questions"][t.TIMER_KEY]
    assert body["state"] == 'Request: "set a timer for 10 seconds"' and body["images"] == []
    assert q == {"type": "choice", "instructions": t.TIMER_QUESTION, "criteria": t.TIMER_CRITERIA}
    assert r["best"] == "start" and r["p"] == 0.92
    # The address check on the same classifier still uses its own field.
    await c.ask("Transcript: x", cl.address_questions("Nova"))
    assert "question" in client.sent[1]["questions"][cl.ADDRESS]


# The timer flow ----------------------------------------------------------------------


class TimerClassifier(cl.Classifier):
    def __init__(self, best="start", p=0.95, error=None):
        self.best, self.p, self.error, self.calls = best, p, error, []

    async def timer_action(self, text):
        self.calls.append(text)
        if self.error:
            return {"ms": 5, "error": self.error}
        return {"ms": 5, "best": self.best, "p": self.p, "probs": {}}


class Flow:
    def __init__(self, monkeypatch, classifier, raises=None, speech_slots=None, speech_run=True):
        self.intents, self.forwarded, self.logged = [], [], []
        entry = SimpleNamespace(entry_id="e1", title="Nova", data={}, options={
            CONF_TARGET_AGENT: "conversation.main", CONF_NAMES: "Nova, Alexa", CONF_CLASSIFIER: CLASSIFIER_CLEF})
        self.agent = slc.SmartListeningAgent(entry)
        self.agent.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: None))
        self.agent.entity_id = "conversation.smart_listening_nova"
        self.agent._classifier = classifier
        self.agent._log = self.logged.append
        self.agent._is_speech_run = lambda device_id: speech_run
        self.agent._show_thinking = lambda device_id: None
        self.agent._wake_hint = lambda device_id: None

        async def accepted(user_input):  # the address check has accepted the request
            return {"kind": "speech", "device_id": user_input.device_id, "text": user_input.text, "answer": True}

        self.agent._assess = accepted

        async def forward(user_input):
            self.forwarded.append(user_input.text)
            return slc._spoken(user_input, "agent answer"), {"answered_by": "conversation.main"}

        self.agent._forward = forward

        async def handle(hass, platform, intent_type, slots, text_input, context, language, assistant,
                         device_id, satellite_id, conversation_agent_id):
            self.intents.append({"type": intent_type, "slots": slots, "device_id": device_id, "text": text_input})
            if raises:
                raise raises
            response = ha_intent.IntentResponse(language="en")
            response.async_set_speech_slots(speech_slots or {})
            return response

        monkeypatch.setattr(slc.intent, "async_handle", handle)
        monkeypatch.setattr(slc.ir, "async_create_issue", lambda *a, **k: None)
        monkeypatch.setattr(slc.ir, "async_delete_issue", lambda *a, **k: None)

    async def say(self, text):
        user_input = SimpleNamespace(text=text, device_id="dev", conversation_id="c1", context=None, language="en",
                                     satellite_id="assist_satellite.kitchen", extra_system_prompt=None)
        result = await self.agent.async_process(user_input)
        return result.response.speech["plain"]["speech"], self.logged[-1]


@pytest.mark.asyncio
async def test_start_timer_on_the_device_that_heard_it(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier("start", 0.97))
    speech, rec = await f.say("Nova, set a timer for 10 seconds")
    assert speech == "Timer set for 10 seconds." and f.forwarded == []
    assert f.intents == [{"type": "HassStartTimer", "slots": {"seconds": {"value": 10}}, "device_id": "dev",
                          "text": "set a timer for 10 seconds"}]
    assert rec["answered_by"] == "timers" and rec["timer"]["action"] == "start" and rec["timer"]["seconds"] == 10
    assert f.agent._last_reply["dev"][1] == "Timer set for 10 seconds."  # echo check


@pytest.mark.asyncio
async def test_named_timer_and_loose_wording(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier("start", 0.95))
    speech, _ = await f.say("Nova, can you time the pasta timer, 12 minutes")
    assert f.intents[0]["slots"] == {"minutes": {"value": 12}, "name": {"value": "pasta"}}
    assert speech == "Pasta timer set for 12 minutes."


@pytest.mark.asyncio
async def test_requests_without_time_words_never_ask_the_classifier(monkeypatch):
    c = TimerClassifier()
    f = Flow(monkeypatch, c)
    speech, rec = await f.say("Nova, what's the news today?")
    assert c.calls == [] and f.forwarded == ["Nova, what's the news today?"] and "timer" not in rec


@pytest.mark.asyncio
@pytest.mark.parametrize(("best", "p"), [("reminder", 0.97), ("other", 0.9), ("start", 0.55)])
async def test_reminders_others_and_low_scores_go_to_the_agent(monkeypatch, best, p):
    f = Flow(monkeypatch, TimerClassifier(best, p))
    speech, rec = await f.say("Nova, remind me in 15 minutes to take the washing out")
    assert speech == "agent answer" and f.intents == [] and rec["timer"]["action"] == best


@pytest.mark.asyncio
async def test_start_without_a_readable_duration_goes_to_the_agent(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier("start", 0.95))
    speech, rec = await f.say("Nova, set a timer for a bit")
    assert speech == "agent answer" and rec["timer"]["error"] == "no duration"


@pytest.mark.asyncio
async def test_classifier_error_goes_to_the_agent(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier(error="timeout"))
    speech, rec = await f.say("Nova, set a timer for 10 seconds")
    assert speech == "agent answer" and rec["timer"]["error"] == "timeout"


@pytest.mark.asyncio
async def test_timer_errors(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier("cancel", 0.95), raises=TimerNotFoundError())
    speech, rec = await f.say("Nova, cancel the pasta timer")
    assert speech == "I couldn't find the pasta timer." and rec["timer"]["error"] == "timer not found"

    f = Flow(monkeypatch, TimerClassifier("pause", 0.95), raises=MultipleTimersMatchedError())
    speech, _ = await f.say("Nova, pause the timer")
    assert speech == "There's more than one timer. Please say which one."

    f = Flow(monkeypatch, TimerClassifier("start", 0.95), raises=TimersNotSupportedError("dev"))
    speech, rec = await f.say("Nova, set a timer for 10 seconds")
    assert speech == "agent answer" and rec["timer"]["error"] == "device has no timers"


@pytest.mark.asyncio
async def test_cancel_all_and_status(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier("cancel", 0.95), speech_slots={"canceled": 2})
    speech, _ = await f.say("Nova, delete all the timers")
    assert f.intents[0]["type"] == "HassCancelAllTimers" and f.intents[0]["slots"] == {}
    assert speech == "Cancelled 2 timers."

    f = Flow(monkeypatch, TimerClassifier("status", 0.95), speech_slots={"timers": [_status(m=5)]})
    speech, _ = await f.say("Nova, how long left on my timer?")
    assert f.intents[0]["type"] == "HassTimerStatus" and speech == "5 minutes left on the timer."


@pytest.mark.asyncio
async def test_button_requests_get_timers_too(monkeypatch):
    f = Flow(monkeypatch, TimerClassifier("start", 0.95), speech_run=False)
    speech, rec = await f.say("set a timer for 2 minutes")
    assert speech == "Timer set for 2 minutes." and rec["kind"] == "passthrough" and rec["answered_by"] == "timers"


@pytest.mark.asyncio
async def test_timers_switched_off(monkeypatch):
    c = TimerClassifier()
    f = Flow(monkeypatch, c)
    f.agent._handle_timers = False
    speech, _ = await f.say("Nova, set a timer for 10 seconds")
    assert c.calls == [] and speech == "agent answer"
