"""Open-mic follow-ups (0.4.2): HA may keep listening after a reply that asks a question,
and Smart Listening judges that run before passing it on."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_components.smart_listening import conversation as slc  # noqa: E402
from custom_components.smart_listening.const import (  # noqa: E402
    CLASSIFIER_CLEF,
    CONF_CLASSIFIER,
    CONF_NAMES,
    CONF_TARGET_AGENT,
)
from test_decisions import FakeClassifier  # noqa: E402

SAT = "assist_satellite.kitchen"


class Room:
    def __init__(self, monkeypatch, followup=0.95, address=0.95, ask_back=True):
        self.forwarded, self.logged = [], []
        entry = SimpleNamespace(entry_id="e1", title="Nova", data={}, options={
            CONF_TARGET_AGENT: "conversation.main", CONF_NAMES: "Nova, Alexa", CONF_CLASSIFIER: CLASSIFIER_CLEF})
        a = slc.SmartListeningAgent(entry)
        a.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: None))
        a._classifier = FakeClassifier(address=address, followup=followup)
        a._classifier_cleared = True
        a._log = self.logged.append
        a._show_thinking = lambda d: None
        a._wake_hint = lambda d: None
        a._satellite = lambda d: SAT
        a._handle_timers = False
        self.speech_run = True
        a._is_speech_run = lambda d: self.speech_run
        self.ask_back = ask_back

        async def forward(user_input):
            self.forwarded.append(user_input.text)
            response = slc.intent.IntentResponse(language="en")
            response.async_set_speech("It's 5:28. Anything else?")
            result = slc.conversation.ConversationResult(
                response=response, conversation_id="c1", continue_conversation=self.ask_back)
            a._last_reply["dev"] = (slc.time.monotonic(), "It's 5:28. Anything else?")
            return result, {"answered_by": "conversation.main"}

        a._forward = forward
        self.agent = a

    async def say(self, text):
        ui = SimpleNamespace(text=text, device_id="dev", conversation_id="c1", context=None, language="en",
                             satellite_id=SAT, extra_system_prompt=None)
        result = await self.agent.async_process(ui)
        return result, self.logged[-1]


@pytest.mark.asyncio
async def test_agent_asking_back_keeps_the_microphone_open(monkeypatch):
    room = Room(monkeypatch)
    result, rec = await room.say("Nova, what time is it?")
    assert result.continue_conversation is True and rec["continue"] is True
    assert "dev" in room.agent._invited


@pytest.mark.asyncio
async def test_open_mic_run_is_judged_and_answered(monkeypatch):
    room = Room(monkeypatch, followup=0.95)
    await room.say("Nova, what time is it?")
    room.speech_run = False                      # HA's open-mic run, not a speech run
    room.agent._run_after_reply[SAT] = True      # started straight after the reply
    result, rec = await room.say("What time is sunrise tomorrow?")
    assert rec["kind"] == "open_mic" and rec["answer"] and rec["reason"] == "follow-up"
    assert room.forwarded[-1] == "What time is sunrise tomorrow?"
    assert "dev" in room.agent._invited          # the new reply asked back again


@pytest.mark.asyncio
async def test_open_mic_side_talk_is_dropped(monkeypatch):
    room = Room(monkeypatch, followup=0.05)
    await room.say("Nova, what time is it?")
    room.speech_run = False
    room.agent._run_after_reply[SAT] = True
    result, rec = await room.say("Yeah.")
    assert rec["kind"] == "open_mic" and not rec["answer"]
    assert result.response.speech["plain"]["speech"] == "" and room.forwarded == ["Nova, what time is it?"]


@pytest.mark.asyncio
async def test_button_press_after_a_question_is_passed_through(monkeypatch):
    room = Room(monkeypatch, followup=0.05)
    await room.say("Nova, what time is it?")
    room.speech_run = False
    room.agent._run_after_reply[SAT] = False     # idle first, then listening: a button press
    result, rec = await room.say("Turn on the lights")
    assert rec["kind"] == "passthrough" and room.forwarded[-1] == "Turn on the lights"


@pytest.mark.asyncio
async def test_no_invitation_means_passthrough(monkeypatch):
    room = Room(monkeypatch, ask_back=False)
    room.agent._keep_listening = False
    await room.say("Nova, what time is it?")
    assert "dev" not in room.agent._invited
    room.speech_run = False
    room.agent._run_after_reply[SAT] = True
    result, rec = await room.say("hello")
    assert rec["kind"] == "passthrough"


@pytest.mark.asyncio
async def test_old_invitation_expires(monkeypatch):
    room = Room(monkeypatch)
    await room.say("Nova, what time is it?")
    when, why = room.agent._invited["dev"]
    room.agent._invited["dev"] = (when - slc.OPEN_MIC_WINDOW - 1, why)
    room.speech_run = False
    room.agent._run_after_reply[SAT] = True
    result, rec = await room.say("hello")
    assert rec["kind"] == "passthrough"


@pytest.mark.asyncio
async def test_keep_listening_after_every_answer(monkeypatch):
    room = Room(monkeypatch, ask_back=False)       # the agent did not ask anything
    result, rec = await room.say("Nova, what time is it?")
    assert result.continue_conversation is True and rec["continue"] is True
    assert room.agent._invited["dev"][1] == "reply"


@pytest.mark.asyncio
async def test_name_alone_ends_quietly_then_greets(monkeypatch):
    room = Room(monkeypatch, address=0.05)         # the classifier is not asked
    result, rec = await room.say("Nova.")
    assert result.response.speech["plain"]["speech"] == "" and rec["reason"] == "name only"
    assert room.forwarded == [] and "dev" in room.agent._summon_pending
    for text in ("Hey Nova", "Nova?"):
        _, rec = await room.say(text)
        assert rec["reason"] == "name only", text


def test_greeting_on_the_voice_pe(monkeypatch):
    room = Room(monkeypatch)
    calls = []

    async def async_call(domain, service, data, blocking=False):
        calls.append((domain, service, data))

    def create_task(coro):
        coro.close()
        calls.append("task")

    room.agent.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: None),
                                      services=SimpleNamespace(async_call=async_call),
                                      async_create_task=create_task)
    room.agent._summon("dev", "wake word, no speech")
    assert calls == ["task"] and room.agent._invited["dev"][1] == "summoned"
    assert room.agent._last_reply["dev"][1] == "Hi, how can I help?"
    assert room.logged[-1] == {"kind": "summon", "device_id": "dev", "reason": "wake word, no speech",
                               "reply": "Hi, how can I help?"}


@pytest.mark.asyncio
async def test_request_after_the_name_is_answered_without_judging(monkeypatch):
    room = Room(monkeypatch, followup=0.0, address=0.0)
    await room.say("Nova.")
    room.agent._invited["dev"] = (slc.time.monotonic(), "summoned")  # greeted when the run ended
    room.speech_run = False
    room.agent._run_after_reply[SAT] = True
    result, rec = await room.say("turn the kitchen lights off")
    assert rec["kind"] == "open_mic" and rec["reason"] == "after the name"
    assert room.forwarded == ["turn the kitchen lights off"]
