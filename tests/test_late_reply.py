"""smart_listening.reply: late replies on the speaker the room's replies come from."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components import smart_listening as sl  # noqa: E402
from custom_components.smart_listening import conversation as slc  # noqa: E402
from test_followups import SAT, Room, _Services  # noqa: E402

URL = "http://ha:8123/api/tts_proxy/abc.mp3"


def _agent(monkeypatch, replies_switch="off", script=False, sat_state="idle"):
    room = Room(monkeypatch)
    a = room.agent
    services = _Services()
    states = {"switch.replies": SimpleNamespace(state=replies_switch),
              "media_player.vpe": SimpleNamespace(state="idle", attributes={"volume_level": 0.6}),
              "script.room": SimpleNamespace(state="off") if script else None,
              SAT: SimpleNamespace(state=sat_state)}
    a.hass = SimpleNamespace(states=SimpleNamespace(get=lambda e: states.get(e)), services=services,
                             async_create_task=lambda coro: coro.close())
    a._reply_script = "script.room" if script else ""
    a._device_entity = lambda d, suffix: "switch.replies"
    a._esphome_name = lambda sat: "home-assistant-voice-kitchen"
    a._tts_url = lambda sat, msg: URL
    a._keep_listening = True
    monkeypatch.setattr(slc.er, "async_get", lambda hass: None)
    monkeypatch.setattr(slc.er, "async_entries_for_device", lambda reg, d: [
        SimpleNamespace(entity_id="media_player.vpe", domain="media_player", platform="esphome")])
    monkeypatch.setattr(slc, "async_call_later", lambda hass, delay, cb: None)
    return a, services, states, room


@pytest.mark.asyncio
@pytest.mark.parametrize(("listen", "call"), [
    (True, ("assist_satellite", "start_conversation", {"entity_id": SAT, "start_message": "It came.", "preannounce": False})),
    (False, ("assist_satellite", "announce", {"entity_id": SAT, "message": "It came.", "preannounce": False})),
])
async def test_late_reply_from_the_voice_pe(monkeypatch, listen, call):
    a, services, _, room = _agent(monkeypatch, replies_switch="off")
    out = await a.async_late_reply("dev", SAT, "It came.", listen, "Kitchen")
    assert out == {"satellite": SAT, "speaker": "voice_pe"}
    assert services.calls == [call]
    assert a._last_reply["dev"][1] == "It came."
    assert ("dev" in a._invited) is listen
    assert room.logged[-1]["kind"] == "late_reply" and room.logged[-1]["area"] == "Kitchen"


@pytest.mark.asyncio
async def test_late_reply_on_the_room_speaker_then_listens(monkeypatch):
    a, services, _, _ = _agent(monkeypatch, replies_switch="on", script=True)
    out = await a.async_late_reply("dev", SAT, "It came.", None, None)  # listen from keep_listening
    assert out["speaker"] == "room_speaker"
    assert services.calls == [
        ("media_player", "volume_set", {"entity_id": "media_player.vpe", "volume_level": 0}),
        ("script", "room", {"url": URL, "satellite": "home-assistant-voice-kitchen"}),
        ("assist_satellite", "start_conversation", {"entity_id": SAT, "start_media_id": URL, "preannounce": False}),
    ]
    assert a._invited["dev"][1] == "reply"


@pytest.mark.asyncio
async def test_late_reply_on_the_room_speaker_without_listening(monkeypatch):
    a, services, _, _ = _agent(monkeypatch, replies_switch="on", script=True)
    await a.async_late_reply("dev", SAT, "It came.", False, None)
    assert services.calls == [("script", "room", {"url": URL, "satellite": "home-assistant-voice-kitchen"})]
    assert "dev" not in a._invited


@pytest.mark.asyncio
async def test_late_reply_waits_for_the_reply_still_playing(monkeypatch):
    a, services, states, _ = _agent(monkeypatch, sat_state="responding")
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            states[SAT] = SimpleNamespace(state="listening")

    monkeypatch.setattr(slc.asyncio, "sleep", fake_sleep)
    await a.async_late_reply("dev", SAT, "It came.", True, None)
    assert len(sleeps) == 3 and len(services.calls) == 1


def _registries(monkeypatch, sat_area="kitchen"):
    sats = {SAT: SimpleNamespace(entity_id=SAT, domain="assist_satellite", device_id="dev", area_id=None)}
    monkeypatch.setattr(slc.er, "async_get", lambda hass: SimpleNamespace(entities=sats, async_get=sats.get))
    areas = [SimpleNamespace(id="kitchen", name="Kitchen", aliases=set()),
             SimpleNamespace(id="living_room", name="Living Room", aliases={"Lounge"})]
    monkeypatch.setattr(slc.ar, "async_get", lambda hass: SimpleNamespace(
        async_get_area=lambda i: next((x for x in areas if x.id == i), None), async_list_areas=lambda: areas))
    monkeypatch.setattr(slc.dr, "async_get", lambda hass: SimpleNamespace(
        async_get=lambda d: SimpleNamespace(area_id=sat_area)))


@pytest.mark.parametrize(("area", "found"), [
    ("Kitchen", True), ("kitchen", True), ("KITCHEN", True), ("Living Room", False), ("Lounge", False), ("Attic", False),
])
def test_find_voice_pe_by_area(monkeypatch, area, found):
    a = Room(monkeypatch).agent
    _registries(monkeypatch)
    a._uses_me = lambda sat: True
    assert (a.find_voice_pe(area, None) == ("dev", SAT, 0.0)) is found


def test_find_voice_pe_skips_satellites_on_other_assistants(monkeypatch):
    a = Room(monkeypatch).agent
    _registries(monkeypatch)
    a._uses_me = lambda sat: False
    assert a.find_voice_pe("Kitchen", None) is None


def test_find_voice_pe_by_last_request(monkeypatch):
    a = Room(monkeypatch).agent
    now = slc.dt_util.utcnow()
    a._last_process = {"old": now - dt.timedelta(hours=2), "dev": now - dt.timedelta(seconds=40)}
    a._satellite = lambda d: SAT if d == "dev" else "assist_satellite.old"
    device, sat, age = a.find_voice_pe(None, None)
    assert (device, sat) == ("dev", SAT) and 39 < age < 60
    a._last_process = {"old": now - dt.timedelta(hours=2)}
    assert a.find_voice_pe(None, None) is None


@pytest.mark.asyncio
async def test_reply_action_errors_when_no_voice_pe(monkeypatch):
    agent = SimpleNamespace(find_voice_pe=lambda area, sat: None)
    hass = SimpleNamespace(data={sl.DOMAIN: {"agents": {"conversation.sl": agent}}})
    with pytest.raises(sl.ServiceValidationError, match="area Attic"):
        await sl.async_reply(hass, {"message": "hi", "area": "Attic"})
    with pytest.raises(sl.ServiceValidationError, match="not a Smart Listening agent"):
        await sl.async_reply(hass, {"message": "hi", "agent": "conversation.other"})


@pytest.mark.asyncio
async def test_reply_action_uses_the_agent_with_the_latest_request(monkeypatch):
    said = []

    def make(name, age):
        async def late(device, sat, message, listen, area):
            said.append(name)
            return {"satellite": sat, "speaker": "voice_pe"}
        return SimpleNamespace(find_voice_pe=lambda area, sat: ("dev", SAT, age), async_late_reply=late)

    hass = SimpleNamespace(data={sl.DOMAIN: {"agents": {"a": make("old", 600.0), "b": make("new", 5.0)}}})
    out = await sl.async_reply(hass, {"message": "It came."})
    assert said == ["new"] and out["speaker"] == "voice_pe"
