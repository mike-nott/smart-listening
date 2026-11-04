"""Decision tests, replaying real Voice PE transcripts and classifier scores (2026-11-04).

Run: pytest tests (needs the homeassistant package installed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.smart_listening import classifier as cl  # noqa: E402
from custom_components.smart_listening.const import (  # noqa: E402
    CLASSIFIER_CLEF,
    CLASSIFIER_NAME_ONLY,
    CONF_CLASSIFIER,
    CONF_NAMES,
    CONF_TARGET_AGENT,
)
from custom_components.smart_listening.conversation import (  # noqa: E402
    SmartListeningAgent,
    _name_match,
    parse_names,
)


class FakeClassifier(cl.Classifier):
    asks_kind = True

    def __init__(self, address=None, followup=None, error=False):
        self.p_address, self.p_followup, self.error = address, followup, error
        self.calls: list[str] = []

    async def ask(self, state, questions):
        key = cl.ADDRESS if cl.ADDRESS in questions else next(iter(questions))
        self.calls.append(key)
        if self.error:
            return {"ms": 1, "error": "boom"}
        if key == cl.ADDRESS:
            return {"ms": 1, "answers": {cl.ADDRESS: {"yes": self.p_address, "no": 1 - self.p_address}}}
        return {"ms": 1, "answers": {cl.FOLLOWUP: {"followup": self.p_followup, "not": 1 - self.p_followup}}}


def make_agent(classifier=None, hint=None, kind=CLASSIFIER_CLEF, last_reply=None):
    entry = SimpleNamespace(
        entry_id="e1",
        title="Nova",
        data={},
        options={CONF_TARGET_AGENT: "conversation.llm", CONF_NAMES: "Nova, Alexa", CONF_CLASSIFIER: kind},
    )
    agent = SmartListeningAgent(entry)
    agent._classifier = classifier
    agent._wake_hint = lambda device_id: hint
    if last_reply is not None:
        agent._last_reply["dev"] = (time.monotonic() - last_reply[0], last_reply[1])
    return agent


def speech(text):
    return SimpleNamespace(text=text, device_id="dev")


async def decide(agent, text):
    return await agent._assess(speech(text))


def test_parse_names():
    assert parse_names("Nova, Alexa") == ("Nova", "Alexa")
    assert parse_names(" Sam \n") == ("Sam",)
    assert parse_names("") == ()


def test_name_match_fuzzy():
    score, name, pos, rest = _name_match("Novva, what's the temperature outside?", ("Nova", "Alexa"))
    assert score >= 0.75 and name == "Nova" and pos == 0


@pytest.mark.asyncio
async def test_clipped_name_restored_from_wake_word():
    # Transcript lost "Nova"; the wake word model heard it. Clef 0.94.
    r = await decide(make_agent(FakeClassifier(address=0.9421), hint="Nova"), "What time is it?")
    assert r["answer"] and r["text"] == "Nova, What time is it?" and r["reason"] == "name + classifier"


@pytest.mark.asyncio
async def test_wake_word_lowers_the_bar():
    # Scored 0.74 and was refused under the old single threshold.
    r = await decide(make_agent(FakeClassifier(address=0.7432), hint="Nova"), "Nova, Do you know where Sam is?")
    assert r["answer"] and r["reason"] == "wake word + classifier"


@pytest.mark.asyncio
async def test_same_score_without_wake_word_is_refused():
    r = await decide(make_agent(FakeClassifier(address=0.7432)), "Nova, Do you know where Sam is?")
    assert not r["answer"]


@pytest.mark.asyncio
async def test_side_talk_without_name_skips_classifier():
    c = FakeClassifier(address=0.99)
    r = await decide(make_agent(c), "Yeah.")
    assert not r["answer"] and c.calls == []


@pytest.mark.asyncio
async def test_weak_name_asks_and_refuses():
    r = await decide(make_agent(FakeClassifier(address=0.2476, followup=0.2602), last_reply=(6, "It's Wednesday.")), "Thank you. No more.")
    assert not r["answer"]


@pytest.mark.asyncio
async def test_followup_without_name():
    c = FakeClassifier(address=0.5, followup=0.8418)
    r = await decide(make_agent(c, last_reply=(16.6, "I wish I could count the leaves.")), "Look at one of the security cameras.")
    assert r["answer"] and r["reason"] == "follow-up" and cl.FOLLOWUP in c.calls


@pytest.mark.asyncio
async def test_followup_window_closed():
    c = FakeClassifier(address=0.5, followup=0.99)
    r = await decide(make_agent(c, last_reply=(40, "It's ten fifty-seven.")), "And the weather?")
    assert not r["answer"] and cl.FOLLOWUP not in c.calls


@pytest.mark.asyncio
async def test_echo_of_reply_is_ignored():
    reply = "It's ten fifty-seven in the morning. You've got the whole day ahead of you."
    r = await decide(make_agent(FakeClassifier(address=0.99, followup=0.99), last_reply=(3, reply)), reply)
    assert not r["answer"] and r["reason"] == "echo of last reply"


@pytest.mark.asyncio
async def test_short_command_needs_no_classifier():
    c = FakeClassifier(address=0.0)
    r = await decide(make_agent(c), "Nova, stop.")
    assert r["answer"] and c.calls == []


@pytest.mark.asyncio
async def test_classifier_failure_falls_back_to_name_position():
    assert (await decide(make_agent(FakeClassifier(error=True)), "Alexa, what time is it?"))["answer"]
    r = await decide(make_agent(FakeClassifier(error=True)), "I was telling my sister that Alexa is useless")
    assert not r["answer"]


@pytest.mark.asyncio
async def test_name_only_mode():
    agent = make_agent(None, kind=CLASSIFIER_NAME_ONLY)
    assert (await decide(agent, "Nova, turn the lights off"))["answer"]
    assert not (await decide(agent, "the kids said Nova was wrong again"))["answer"]
    assert not (await decide(agent, "Turn the lights off"))["answer"]


class _Resp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _Client:
    def __init__(self, content):
        self.content, self.sent = content, []

    async def post(self, url, **kw):
        self.sent.append((url, kw))
        return _Resp(self.content)


@pytest.mark.asyncio
async def test_openai_classifier_parses_choice(monkeypatch):
    client = _Client('<think>hmm</think> {"choice": "yes", "confidence": 0.9}')
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    c = cl.OpenAIClassifier(None, "http://llm:11434/v1", "qwen", "k")
    r = await c.ask('Transcript: "Nova, what time is it?"', cl.address_questions("Nova"))
    assert r["answers"][cl.ADDRESS] == {"yes": 0.9, "no": 0.1}
    url, kw = client.sent[0]
    assert url == "http://llm:11434/v1/chat/completions" and kw["headers"] == {"Authorization": "Bearer k"}


@pytest.mark.asyncio
async def test_openai_classifier_bad_reply_is_an_error(monkeypatch):
    monkeypatch.setattr(cl, "get_async_client", lambda hass: _Client("I think so!"))
    r = await cl.OpenAIClassifier(None, "http://llm/v1", "m").ask("x", cl.address_questions("Nova"))
    assert "error" in r


def _run_agent(run_state, run_changed, switch_state, started):
    agent = make_agent(FakeClassifier(address=0.5))
    ents = {"dev": {"run": "binary_sensor.run", "switch": "switch.sl"}}
    states = {
        "binary_sensor.run": SimpleNamespace(state=run_state, last_changed=run_changed),
        "switch.sl": SimpleNamespace(state=switch_state, last_changed=run_changed),
    }
    agent._device_entity = lambda d, suffix: ents[d]["run" if "run" in suffix else "switch"]
    agent._satellite = lambda d: "assist_satellite.kitchen"
    agent.hass = SimpleNamespace(states=SimpleNamespace(get=states.get))
    if started is not None:
        agent._run_started["assist_satellite.kitchen"] = started
    return agent


def test_flag_cleared_mid_run_still_counts():
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 11, 4, 12, 45, 0, 445000, tzinfo=timezone.utc)
    # Kitchen 12:45:00: run started at .445, previous run's end cleared the flag at .518.
    assert _run_agent("off", t0 + timedelta(milliseconds=73), "on", t0)._is_speech_run("dev")
    # Button request: flag cleared before the run started -> passed through.
    assert not _run_agent("off", t0 - timedelta(milliseconds=300), "on", t0)._is_speech_run("dev")
    # Switch off: stock behaviour, never treated as a speech run when the flag is off.
    assert not _run_agent("off", t0 + timedelta(milliseconds=73), "off", t0)._is_speech_run("dev")
    # Flag on with the switch on: speech run, however old.
    assert _run_agent("on", t0 - timedelta(hours=3), "on", None)._is_speech_run("dev")


class _SOResp:
    def __init__(self, answers):
        self._answers = answers

    def raise_for_status(self):
        pass

    def json(self):
        return {"model": "x", "answers": self._answers, "usage": {}}


class _SOClient:
    def __init__(self):
        self.sent = []

    async def post(self, url, **kw):
        self.sent.append((url, kw))
        return _SOResp({k: {"type": "choice", "probabilities": {c: 0.5 for c in q["criteria"]}}
                        for k, q in kw["json"]["questions"].items()})


@pytest.mark.asyncio
async def test_clef_payload_is_as_measured(monkeypatch):
    client = _SOClient()
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    c = cl.ClefClassifier(None, "http://clef:8080/v1/systemone")
    await c.ask("Transcript: x", cl.kind_question("Nova") | cl.address_questions("Nova"))
    url, kw = client.sent[0]
    body = kw["json"]
    assert body["model"] == "clef" and body["images"] == [] and "Authorization" not in kw["headers"]
    assert list(body["questions"]) == [cl.KIND, cl.ADDRESS]
    assert "question" in body["questions"][cl.ADDRESS] and "instructions" not in body["questions"][cl.ADDRESS]


class _JevClient(_SOClient):
    async def post(self, url, **kw):
        self.sent.append((url, kw))
        answers = {}
        for k, q in kw["json"]["questions"].items():
            if q["type"] == "noul":
                answers[k] = {"type": "noul", "noul": 0.83}
            else:
                answers[k] = {"type": "choice", "probabilities": {c: 0.5 for c in q["criteria"]}}
        return _SOResp(answers)


@pytest.mark.asyncio
async def test_jev_address_request(monkeypatch):
    client = _JevClient()
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    c = cl.make_classifier(None, "jev", "", "", "secret")
    r = await c.address("Nova, what time is it?", "Nova")
    url, kw = client.sent[0]
    body = kw["json"]
    assert url == "https://api.typesafe.ai/v1/systemone" and body["model"] == "jev-1.14.0"
    assert kw["headers"] == {"Authorization": "Bearer secret"} and "images" not in body
    assert body["state"]["transcript"] == "Nova, what time is it?" and "setting" in body["state"]
    assert list(body["questions"]) == [cl.KIND, cl.JEV_ADDRESS]
    n1 = body["questions"][cl.JEV_ADDRESS]
    assert n1["type"] == "noul" and set(n1) == {"type", "instructions", "criteria"} and set(n1["criteria"]) == {"true", "false"}
    assert r["p"] == 0.83


@pytest.mark.asyncio
async def test_jev_followup_request(monkeypatch):
    client = _JevClient()
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    r = await cl.make_classifier(None, "jev", "", "", "k").followup("Timer set.", "Make it fifteen.", "Nova")
    body = client.sent[0][1]["json"]
    assert body["state"]["assistant_said"] == "Timer set." and body["state"]["new_speech"] == "Make it fifteen."
    assert list(body["questions"]) == [cl.FOLLOWUP] and "instructions" in body["questions"][cl.FOLLOWUP]
    assert r["p"] == 0.5


@pytest.mark.asyncio
async def test_kind_question_is_asked_first():
    seen = []

    class Recorder(FakeClassifier):
        async def ask(self, state, questions):
            seen.append(list(questions))
            return await super().ask(state, {cl.ADDRESS: None})

    await decide(make_agent(Recorder(address=0.9)), "Nova, what time is it?")
    assert seen[0] == [cl.KIND, cl.ADDRESS]


class _CFResp(_SOResp):
    def json(self):
        return {"result": {"model": "clef", "answers": self._answers}, "success": True, "errors": []}


class _CFClient(_SOClient):
    async def post(self, url, **kw):
        self.sent.append((url, kw))
        answers = {}
        for k, q in kw["json"]["questions"].items():
            if q["type"] == "noul":
                answers[k] = {"type": "noul", "noul": 0.5}
            else:
                answers[k] = {"type": "choice", "probabilities": {c: 0.9 if c in ("yes", "command") else 0.05 for c in q["criteria"]}}
        return _CFResp(answers)


@pytest.mark.asyncio
async def test_clef_cloud_request_and_reply(monkeypatch):
    from custom_components.smart_listening.config_flow import _validate_classifier

    client = _CFClient()
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    data = {"classifier": "clef_cloud", "cloudflare_account_id": " abc123 ", "classifier_model": "",
            "classifier_api_key": "cf-token"}
    errors = {}
    await _validate_classifier(None, data, errors)  # sends one tiny test decision
    assert errors == {}
    url = "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/cloudflare/clef"
    assert data["classifier_url"] == url and data["classifier_model"] == "clef"

    c = cl.make_classifier(None, "clef_cloud", data["classifier_url"], data["classifier_model"], data["classifier_api_key"])
    r = await c.address("Nova, what time is it?", "Nova")
    sent_url, kw = client.sent[-1]
    body = kw["json"]
    assert sent_url == url and kw["headers"] == {"Authorization": "Bearer cf-token"}
    assert body["model"] == "clef" and "images" not in body
    assert list(body["questions"]) == [cl.KIND, cl.ADDRESS]
    assert "instructions" in body["questions"][cl.ADDRESS] and "question" not in body["questions"][cl.ADDRESS]
    assert r["p"] == 0.9  # unwrapped from Cloudflare's {"result": ...}


@pytest.mark.asyncio
async def test_clef_cloud_validation_errors():
    from custom_components.smart_listening.config_flow import _validate_classifier

    for data, field, code in (
        ({"classifier": "clef_cloud", "classifier_api_key": "t"}, "cloudflare_account_id", "no_account"),
        ({"classifier": "clef_cloud", "cloudflare_account_id": "a", "classifier_model": "llama", "classifier_api_key": "t"},
         "classifier_model", "invalid_clef_model"),
        ({"classifier": "clef_cloud", "cloudflare_account_id": "a"}, "classifier_api_key", "no_api_key"),
    ):
        errors = {}
        await _validate_classifier(None, data, errors)
        assert errors == {field: code}


def test_switching_classifier_resets_endpoint_and_threshold():
    from custom_components.smart_listening.config_flow import _switch_classifier

    current = {"classifier": "clef", "classifier_url": "http://clef:8080/v1/systemone", "classifier_model": "",
               "address_threshold": 0.8}
    to_jev = {**current, "classifier": "jev", "classifier_api_key": "k"}
    _switch_classifier(current, to_jev)
    assert to_jev["classifier_url"] == "" and to_jev["address_threshold"] == 0.7

    to_cf = {**current, "classifier": "clef_cloud", "address_threshold": 0.65}
    _switch_classifier(current, to_cf)
    assert to_cf["address_threshold"] == 0.65  # changed on the same screen: kept

    back = {"classifier": "clef", "classifier_url": "http://clef:8080/v1/systemone", "address_threshold": 0.7}
    _switch_classifier({**back, "classifier": "jev"}, back)
    assert back["address_threshold"] == 0.5 and back["classifier_url"] == "http://clef:8080/v1/systemone"


@pytest.mark.asyncio
async def test_new_clef_entries_use_instructions(monkeypatch):
    from custom_components.smart_listening.const import CLASSIFIER_DEFAULTS

    client = _SOClient()
    monkeypatch.setattr(cl, "get_async_client", lambda hass: client)
    field = CLASSIFIER_DEFAULTS["clef"]["clef_question_field"]
    assert field == "instructions" and CLASSIFIER_DEFAULTS["clef"]["address_threshold"] == 0.5
    c = cl.make_classifier(None, "clef", "http://clef:30000/v1/systemone", "Cloudflare/clef", "", field)
    await c.address("Nova, what time is it?", "Nova")
    body = client.sent[0][1]["json"]
    assert body["model"] == "Cloudflare/clef" and body["images"] == []
    assert list(body["questions"]) == [cl.KIND, cl.ADDRESS]
    assert "instructions" in body["questions"][cl.ADDRESS] and "question" not in body["questions"][cl.ADDRESS]


def test_switching_to_clef_sets_its_field():
    from custom_components.smart_listening.config_flow import _switch_classifier

    current = {"classifier": "jev", "classifier_url": "https://api.typesafe.ai/v1/systemone", "address_threshold": 0.7}
    to_clef = {**current, "classifier": "clef", "classifier_url": "http://clef:30000/v1/systemone"}
    _switch_classifier(current, to_clef)
    assert to_clef["clef_question_field"] == "instructions" and to_clef["address_threshold"] == 0.5


def test_agent_named_like_mcp_assist():
    agent = make_agent()
    assert agent.name == "Smart Listening - Nova"
    assert agent._attr_suggested_object_id == "smart_listening_nova"
    info = agent.device_info
    assert info["name"] == "Smart Listening - Nova" and info["manufacturer"] == "Smart Listening"
    assert info["model"] == "Clef" and ("smart_listening", "e1") in info["identifiers"]
