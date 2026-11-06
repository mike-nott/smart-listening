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
async def test_name_alone_is_answered_like_any_reply_and_listens(monkeypatch):
    room = Room(monkeypatch, address=0.05)         # the classifier is not asked
    result, rec = await room.say("Nova.")
    assert result.response.speech["plain"]["speech"] == "Hi, how can I help?"
    assert result.continue_conversation is True and rec["reason"] == "name only"
    assert room.forwarded == [] and room.agent._invited["dev"][1] == "summoned"
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



def _summoned_room(monkeypatch):
    room = Room(monkeypatch, followup=0.0, address=0.0)
    room.agent._invited["dev"] = (slc.time.monotonic(), "summoned")
    room.speech_run = False
    room.agent._run_after_reply[SAT] = True
    return room


@pytest.mark.asyncio
async def test_greeting_heard_back_is_taken_off_the_request(monkeypatch):
    room = _summoned_room(monkeypatch)
    result, rec = await room.say("Hi, how can I help? Hi, how can I help? What time is it?")
    assert rec["reason"] == "after the name" and rec["request"] == "What time is it?"
    assert room.forwarded == ["What time is it?"]


@pytest.mark.asyncio
async def test_greeting_heard_back_alone_is_ignored(monkeypatch):
    room = _summoned_room(monkeypatch)
    result, rec = await room.say("Hi, how can I help? Hi, how can I help?")
    assert rec["answer"] is False and rec["reason"] == "greeting echo only"
    assert room.forwarded == []
    assert result.response.speech["plain"]["speech"] == ""


@pytest.mark.parametrize("heard, rest", [
    ("Hi, how can I help?", ""),
    ("Hi how can I help you?", ""),
    ("How can I help? Turn the lights off.", "Turn the lights off."),
    ("Turn the lights off.", "Turn the lights off."),
    ("Hi, what's the weather?", "Hi, what's the weather?"),
])
def test_strip_echo(heard, rest):
    assert slc._strip_echo(heard, "Hi, how can I help?")[0] == rest

class _Services:
    def __init__(self):
        self.calls = []

    async def async_call(self, domain, service, data, blocking=False):
        self.calls.append((domain, service, data))


def _greeter(monkeypatch, replies_switch, script):
    room = Room(monkeypatch)
    services = _Services()
    states = {"switch.replies": SimpleNamespace(state=replies_switch) if replies_switch else None,
              "media_player.vpe": SimpleNamespace(state="idle", attributes={"volume_level": 0.6}),
              "script.room": SimpleNamespace(state="off") if script else None}
    a = room.agent
    a.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: states.get(e)), services=services,
                             async_create_task=lambda coro: coro.close())
    a._reply_script = "script.room" if script else ""
    a._device_entity = lambda d, suffix: "switch.replies" if replies_switch else None
    a._esphome_name = lambda sat: "home-assistant-voice-kitchen"
    a._tts_url = lambda sat, msg: "http://ha:8123/api/tts_proxy/abc.mp3"
    monkeypatch.setattr(slc.er, "async_get", lambda hass: None)
    monkeypatch.setattr(slc.er, "async_entries_for_device", lambda reg, d: [
        SimpleNamespace(entity_id="media_player.vpe", domain="media_player", platform="esphome")])
    monkeypatch.setattr(slc, "async_call_later", lambda hass, delay, cb: None)
    return a, services


@pytest.mark.asyncio
async def test_greeting_plays_on_the_room_speaker_like_replies(monkeypatch):
    a, services = _greeter(monkeypatch, replies_switch="on", script=True)
    await a._greet("dev", SAT)
    assert services.calls == [
        ("media_player", "volume_set", {"entity_id": "media_player.vpe", "volume_level": 0}),
        ("script", "room", {"url": "http://ha:8123/api/tts_proxy/abc.mp3", "satellite": "home-assistant-voice-kitchen"}),
        ("assist_satellite", "start_conversation",
         {"entity_id": SAT, "start_media_id": "http://ha:8123/api/tts_proxy/abc.mp3", "preannounce": False}),
    ]
    assert a._restore_volume[SAT] == ("media_player.vpe", 0.6)


@pytest.mark.asyncio
@pytest.mark.parametrize(("switch", "script"), [("off", True), (None, False)])
async def test_greeting_plays_on_the_voice_pe_when_replies_do(monkeypatch, switch, script):
    a, services = _greeter(monkeypatch, replies_switch=switch, script=script)
    await a._greet("dev", SAT)
    assert services.calls == [("assist_satellite", "start_conversation",
                               {"entity_id": SAT, "start_message": "Hi, how can I help?", "preannounce": False})]


def _pause_agent(monkeypatch, value, states, uses_me=True):
    room = Room(monkeypatch)
    a = room.agent
    services = _Services()
    tasks = []
    a.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: states.get(e)), services=services,
                             async_create_task=lambda coro: tasks.append(coro))
    a._name_pause = value
    ents = {e: SimpleNamespace(entity_id=e, platform="esphome", device_id="dev",
                               unique_id="20:F8:3B:09:06:5B/0/number/Name pause") for e in states}
    monkeypatch.setattr(slc.er, "async_get", lambda hass: SimpleNamespace(entities=ents, async_get=ents.get))
    a._satellite = lambda d: SAT
    a._uses_me = lambda sat: uses_me
    return a, services, tasks


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "state", "uses_me", "pushed"), [
    (500, "0.0", True, True),
    (500, "500.0", True, False),
    (500, "unavailable", True, False),
    (500, "0.0", False, False),
])
async def test_name_pause_is_pushed_to_voice_pes(monkeypatch, value, state, uses_me, pushed):
    a, services, tasks = _pause_agent(monkeypatch, value, {"number.lr_name_pause": SimpleNamespace(state=state)}, uses_me)
    a._push_name_pause()
    for t in tasks:
        await t
    expected = [("number", "set_value", {"entity_id": "number.lr_name_pause", "value": 500})] if pushed else []
    assert services.calls == expected


def _ended_run(monkeypatch, hint_after, handled=False):
    room = Room(monkeypatch)
    a = room.agent
    import datetime as _dt
    start = _dt.datetime(2026, 11, 6, 8, 0, 50, tzinfo=_dt.timezone.utc)
    hint = SimpleNamespace(state="Alexa", last_changed=start + _dt.timedelta(seconds=hint_after))
    a.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: hint if e == "text_sensor.hint" else None))
    ent = SimpleNamespace(platform="esphome", unique_id="x" + slc.DEVICE_RUN, device_id="dev")
    monkeypatch.setattr(slc.er, "async_get", lambda hass: SimpleNamespace(async_get=lambda e: ent))
    a._device_entity = lambda d, suffix: "text_sensor.hint"
    a._names = ["Lorelai", "Alexa"]
    if handled:
        a._last_process["dev"] = start + _dt.timedelta(seconds=1)
    summoned = []
    a._summon = lambda d, why: summoned.append((d, why))
    event = SimpleNamespace(data={"entity_id": "binary_sensor.run", "old_state": SimpleNamespace(last_changed=start)})
    a._on_run_ended(event)
    return summoned


@pytest.mark.parametrize(("hint_after", "handled", "greets"), [
    (0.3, False, True),    # "Alexa" on its own
    (3.3, False, False),   # "Alexa" heard 3 s into someone talking
    (0.3, True, False),    # Smart Listening judged the run
])
def test_run_ended_greets_only_when_the_name_opened_the_speech(monkeypatch, hint_after, handled, greets):
    summoned = _ended_run(monkeypatch, hint_after, handled)
    assert summoned == ([("dev", "wake word, no speech")] if greets else [])
