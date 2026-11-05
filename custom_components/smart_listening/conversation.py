"""Smart Listening conversation agent.

Requests a Voice PE started by itself on hearing speech (its "Smart Listening run"
sensor is on) are judged: echo check, short commands, name match (with the wake word
model's hint when the start of the speech was clipped), then the classifier. Requests
meant for the assistant are passed to the target agent; the rest get an empty reply, so
HA skips text-to-speech and saves nothing to the chat history.

Wake word and button requests (run sensor off) are passed through unchanged.

If the target agent is missing, raises, times out or reports a server failure, the
request goes to the backup agent; if that fails too, the assistant says it isn't
available. Failures are written to the decision log, logged once in Home Assistant's
log, and shown under Settings > Repairs until the agent answers again.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import time
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL, STATE_ON
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er, intent
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.start import async_at_started
from homeassistant.util import dt as dt_util

from .classifier import make_classifier
from .timers import (
    ACTION_INTENTS,
    NEEDS_DURATION,
    TIMER_ACTIONS,
    confirmation,
    is_all,
    mentions_time,
    parse_duration,
    parse_name,
    slots_for,
)
from .const import (
    AGENT_ISSUE_AFTER,
    CLASSIFIER_DISPLAY_NAMES,
    CLASSIFIER_ISSUE_AFTER,
    CONF_ADDRESS_THRESHOLD,
    CONF_AGENT_TIMEOUT,
    CONF_BACKUP_AGENT,
    CONF_CLASSIFIER,
    CONF_CLASSIFIER_API_KEY,
    CONF_CLASSIFIER_MODEL,
    CONF_CLEF_FIELD,
    CONF_CLASSIFIER_URL,
    CONF_DECISION_LOG,
    CONF_FOLLOWUP_SECONDS,
    CONF_FOLLOWUP_THRESHOLD,
    CONF_NAMES,
    CONF_TARGET_AGENT,
    CONF_TIMER_THRESHOLD,
    CONF_TIMERS,
    CONF_WAKE_WORD_THRESHOLD,
    DEFAULTS,
    DEVICE_RUN,
    DEVICE_SWITCH,
    DEVICE_THINKING,
    DEVICE_WAKE_WORD,
    DOMAIN,
    ECHO_SIMILARITY,
    ECHO_WINDOW,
    OPEN_MIC_GAP,
    OPEN_MIC_WINDOW,
    LOG_FILE,
    LOG_MAX_BYTES,
    NAME_ASK_THRESHOLD,
    NAME_ONLY_POSITION,
    NAME_THRESHOLD,
    RUN_MAX_AGE,
    SHORT_COMMANDS,
    STARTUP_CHECK_DELAY,
    UNAVAILABLE_MESSAGE,
)

_LOGGER = logging.getLogger(__name__)

_FILLERS = {"okay", "ok", "hey", "um", "uh", "please", "oh"}


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    async_add_entities([SmartListeningAgent(entry)])


def parse_names(value: Any) -> tuple[str, ...]:
    """'Nova, Alexa' -> ('Nova', 'Alexa')."""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = re.split(r"[,\n]", str(value or ""))
    return tuple(n.strip() for n in items if n and n.strip())


def _name_match(text: str, names: tuple[str, ...]) -> tuple[float, str | None, int, str]:
    """Best fuzzy match of an assistant name: score, name, word position, words after it."""
    words = re.findall(r"[a-z]+", text.lower())
    best, best_name, pos, rest = 0.0, None, -1, ""
    for name in names:
        target = re.sub(r"[^a-z]", "", name.lower())
        for i in range(len(words)):
            for k in (1, 2):
                if i + k <= len(words):
                    r = difflib.SequenceMatcher(None, "".join(words[i : i + k]), target).ratio()
                    if r > best:
                        best, best_name, pos = r, name, i
                        rest = " ".join(
                            w for j, w in enumerate(words) if not i <= j < i + k and w not in _FILLERS
                        )
    return round(best, 2), best_name, pos, rest


def _speech(result: conversation.ConversationResult) -> str:
    try:
        return result.response.speech.get("plain", {}).get("speech", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _silent(user_input: conversation.ConversationInput) -> conversation.ConversationResult:
    """Empty reply: HA skips text-to-speech and does not save the chat log."""
    response = intent.IntentResponse(language=user_input.language)
    response.async_set_speech("")
    return conversation.ConversationResult(
        response=response, conversation_id=user_input.conversation_id, continue_conversation=False
    )


def _spoken(user_input: conversation.ConversationInput, speech: str) -> conversation.ConversationResult:
    response = intent.IntentResponse(language=user_input.language)
    response.async_set_speech(speech)
    return conversation.ConversationResult(
        response=response, conversation_id=user_input.conversation_id, continue_conversation=False
    )


_LEADING_FILLER = re.compile(r"^(?:hey|hi|ok|okay|oh|um|uh)\b[\s,.!]*", re.IGNORECASE)


def _strip_name(text: str, names: tuple[str, ...]) -> str:
    """The request without the assistant's name: "Lorelai, set a timer for 10 seconds"
    -> "set a timer for 10 seconds". Same fuzzy match as _name_match; digits are kept."""
    words = list(re.finditer(r"[A-Za-z]+", text))
    best, span = 0.0, None
    for name in names:
        target = re.sub(r"[^a-z]", "", name.lower())
        for i in range(len(words)):
            for k in (1, 2):
                if i + k <= len(words):
                    joined = "".join(w.group(0).lower() for w in words[i : i + k])
                    ratio = difflib.SequenceMatcher(None, joined, target).ratio()
                    if ratio > best:
                        best, span = ratio, (words[i].start(), words[i + k - 1].end())
    if span is None or best < NAME_THRESHOLD:
        return text.strip()
    out = text[: span[0]] + text[span[1] :]
    out = re.sub(r"^[\s,.!?:;-]+", "", out)
    out = _LEADING_FILLER.sub("", out)
    out = re.sub(r"\s*,\s*([.!?])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip(" ,")
    return out


def _unavailable(user_input: conversation.ConversationInput, name: str) -> conversation.ConversationResult:
    """Spoken reply when no agent could answer an accepted request."""
    response = intent.IntentResponse(language=user_input.language)
    response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, UNAVAILABLE_MESSAGE.format(name=name))
    return conversation.ConversationResult(
        response=response, conversation_id=user_input.conversation_id, continue_conversation=False
    )


def issue_id(kind: str, entry_id: str) -> str:
    """Repair issue id for an entry ("agent" or "classifier")."""
    return f"{kind}_unavailable_{entry_id}"


def _append(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > LOG_MAX_BYTES:
        os.replace(path, path + ".1")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class SmartListeningAgent(conversation.ConversationEntity):
    """Conversation agent that answers only the speech meant for its assistant."""

    _attr_has_entity_name = False
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        opts = {**DEFAULTS, **entry.data, **entry.options}
        self._target_agent: str = opts[CONF_TARGET_AGENT]
        self._backup_agent: str = opts.get(CONF_BACKUP_AGENT) or ""
        self._agent_timeout = float(opts[CONF_AGENT_TIMEOUT])
        self._handle_timers = bool(opts[CONF_TIMERS])
        self._timer_threshold = float(opts[CONF_TIMER_THRESHOLD])
        self._entry_id = entry.entry_id
        self._title = entry.title
        # Consecutive failures, and whether this agent raised a repair issue. An issue can
        # also be left from before a reload or restart, so the first success clears it.
        self._agent_failures = 0
        self._agent_issue = False
        self._agent_cleared = False
        self._classifier_failures = 0
        self._classifier_issue = False
        self._classifier_cleared = False
        self._cancel_startup_check = None
        self._names = parse_names(opts[CONF_NAMES])
        first, others = self._names[0], self._names[1:]
        # How the classifier is told the assistant's name.
        self._assistant = f"{first} (also called {', '.join(others)})" if others else first
        self._address_threshold = float(opts[CONF_ADDRESS_THRESHOLD])
        self._wake_word_threshold = float(opts[CONF_WAKE_WORD_THRESHOLD])
        self._followup_threshold = float(opts[CONF_FOLLOWUP_THRESHOLD])
        self._followup_seconds = float(opts[CONF_FOLLOWUP_SECONDS])
        self._decision_log = bool(opts[CONF_DECISION_LOG])
        self._classifier_kind = opts[CONF_CLASSIFIER]
        self._classifier_opts = (
            opts[CONF_CLASSIFIER_URL],
            opts[CONF_CLASSIFIER_MODEL],
            opts[CONF_CLASSIFIER_API_KEY],
            opts.get(CONF_CLEF_FIELD, "question"),
        )
        self._classifier = None
        self._attr_unique_id = entry.entry_id
        # Named like MCP Assist agents ("OpenClaw - Lorelai"), with a service device.
        self._attr_name = f"Smart Listening - {entry.title}"
        self._attr_suggested_object_id = f"smart_listening_{entry.title.lower().replace(' ', '_')}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"Smart Listening - {entry.title}",
            manufacturer="Smart Listening",
            model=CLASSIFIER_DISPLAY_NAMES.get(self._classifier_kind, self._classifier_kind),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # device_id -> (monotonic time of the last reply, its text)
        self._last_reply: dict[str, tuple[float, str]] = {}
        # assist_satellite entity_id -> when it last started listening (start of a run)
        self._run_started: dict[str, Any] = {}
        # assist_satellite entity_id -> when it last stopped "responding" (a reply ended),
        # and whether its latest run started straight after that (HA's open microphone).
        self._reply_ended: dict[str, Any] = {}
        self._run_after_reply: dict[str, bool] = {}
        # device_id -> when the agent last asked to keep listening (monotonic)
        self._invited: dict[str, float] = {}

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._classifier = make_classifier(self.hass, self._classifier_kind, *self._classifier_opts)

        @callback
        def _is_listening(event_data) -> bool:
            new = event_data.get("new_state")
            return (
                event_data["entity_id"].startswith("assist_satellite.")
                and new is not None
                and new.state == "listening"
            )

        @callback
        def _record(event: Event) -> None:
            entity, new = event.data["entity_id"], event.data["new_state"]
            old = event.data.get("old_state")
            self._run_started[entity] = new.last_changed
            ended = self._reply_ended.get(entity)
            self._run_after_reply[entity] = (old is not None and old.state == "responding") or (
                ended is not None and (new.last_changed - ended).total_seconds() <= OPEN_MIC_GAP
            )

        @callback
        def _left_responding(event_data) -> bool:
            old, new = event_data.get("old_state"), event_data.get("new_state")
            return (
                event_data["entity_id"].startswith("assist_satellite.")
                and old is not None
                and old.state == "responding"
                and new is not None
                and new.state != "responding"
            )

        @callback
        def _record_reply_end(event: Event) -> None:
            self._reply_ended[event.data["entity_id"]] = event.data["new_state"].last_changed

        self.async_on_remove(self.hass.bus.async_listen(EVENT_STATE_CHANGED, _record, event_filter=_is_listening))
        self.async_on_remove(
            self.hass.bus.async_listen(EVENT_STATE_CHANGED, _record_reply_end, event_filter=_left_responding)
        )

        # Agents (MCP Assist and others) can load late, so check the target a little
        # after Home Assistant has started rather than at setup.
        @callback
        def _started(_hass: HomeAssistant) -> None:
            self._cancel_startup_check = async_call_later(self.hass, STARTUP_CHECK_DELAY, self._startup_check)

        self.async_on_remove(async_at_started(self.hass, _started))
        self.async_on_remove(self._cancel_check)

    @callback
    def _cancel_check(self) -> None:
        if self._cancel_startup_check is not None:
            self._cancel_startup_check()
            self._cancel_startup_check = None

    @callback
    def _startup_check(self, _now: Any) -> None:
        self._cancel_startup_check = None
        if self._agent(self._target_agent) is None:
            _LOGGER.warning(
                "Smart Listening %s: the conversation agent %s was not found after startup",
                self._title,
                self._target_agent,
            )
            self._raise_agent_issue()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    def _agent(self, agent_id: str) -> Any:
        try:
            return async_get_agent(self.hass, agent_id)
        except Exception:  # noqa: BLE001
            return None

    def _target(self) -> Any:
        return self._agent(self._target_agent)

    @property
    def supports_streaming(self) -> bool:
        return bool(getattr(self._target(), "supports_streaming", False))

    async def async_prepare(self, language: str | None = None) -> None:
        if (target := self._target()) is not None and hasattr(target, "async_prepare"):
            await target.async_prepare(language)

    # Device entities ----------------------------------------------------------------

    def _device_entity(self, device_id: str | None, suffix: str) -> str | None:
        if not device_id:
            return None
        for ent in er.async_entries_for_device(er.async_get(self.hass), device_id):
            if ent.platform == "esphome" and ent.unique_id.endswith(suffix):
                return ent.entity_id
        return None

    def _satellite(self, device_id: str | None) -> str | None:
        if not device_id:
            return None
        for ent in er.async_entries_for_device(er.async_get(self.hass), device_id):
            if ent.domain == "assist_satellite":
                return ent.entity_id
        return None

    def _is_speech_run(self, device_id: str | None) -> bool:
        """True if the device started this request itself on hearing speech."""
        run = self._device_entity(device_id, DEVICE_RUN)
        state = self.hass.states.get(run) if run else None
        if state is None:
            return False
        switch = self._device_entity(device_id, DEVICE_SWITCH)
        sw = self.hass.states.get(switch) if switch else None
        switch_on = sw is not None and sw.state == STATE_ON
        if state.state != STATE_ON:
            # A flag switched off after this run started was cleared by the end of the
            # previous run (back-to-back runs), so this is still a speech run. Button
            # requests clear the flag before they start, so they are not caught here.
            started = self._run_started.get(self._satellite(device_id) or "")
            return switch_on and started is not None and state.last_changed > started
        # While the device's switch is on, an "on" flag always counts, however old: the
        # firmware keeps it on through back-to-back runs, and a run cut off by a restart
        # leaves it on, so its last_changed is not the start of this run.
        if switch_on:
            return True
        return (dt_util.utcnow() - state.last_changed).total_seconds() < RUN_MAX_AGE

    def _wake_hint(self, device_id: str | None) -> str | None:
        """Assistant name the device's wake word model heard during this run."""
        entity = self._device_entity(device_id, DEVICE_WAKE_WORD)
        state = self.hass.states.get(entity) if entity else None
        if state is None or (dt_util.utcnow() - state.last_changed).total_seconds() >= RUN_MAX_AGE:
            return None
        heard = state.state.strip().lower()
        for name in self._names:
            if name.lower() in heard:
                return name
        return None

    def _show_thinking(self, device_id: str | None) -> None:
        """Ask the device to show its thinking LEDs now that the request is accepted."""
        if entity := self._device_entity(device_id, DEVICE_THINKING):
            self.hass.async_create_task(
                self.hass.services.async_call("button", "press", {"entity_id": entity}, blocking=False)
            )

    # Requests -------------------------------------------------------------------------

    async def _call_agent(
        self, agent_id: str, user_input: conversation.ConversationInput
    ) -> tuple[conversation.ConversationResult | None, str | None]:
        """Ask one agent. Returns (result, None), or (None, why it failed)."""
        if self._agent(agent_id) is None:
            return None, "not found"
        try:
            async with asyncio.timeout(self._agent_timeout):
                result = await conversation.async_converse(
                    self.hass,
                    user_input.text,
                    user_input.conversation_id,
                    user_input.context,
                    user_input.language,
                    agent_id,
                    user_input.device_id,
                    user_input.satellite_id,
                    user_input.extra_system_prompt,
                )
        except TimeoutError:
            return None, "timed out"
        except Exception as err:  # noqa: BLE001
            return None, repr(err)[:200]
        response = result.response
        if (
            response.response_type == intent.IntentResponseType.ERROR
            and response.error_code == intent.IntentResponseErrorCode.UNKNOWN
        ):
            # How LLM agents such as MCP Assist report a server failure. Other error codes
            # ("no device called that") are real answers and are passed on as they are.
            return None, f"error response: {_speech(result)[:150]}"
        return result, None

    async def _forward(
        self, user_input: conversation.ConversationInput
    ) -> tuple[conversation.ConversationResult, dict[str, Any]]:
        """Hand the request to the target agent, then the backup; never fail silently."""
        info: dict[str, Any] = {}
        result, err = await self._call_agent(self._target_agent, user_input)
        if err is None:
            info["answered_by"] = self._target_agent
        else:
            info["agent_error"] = err
            if self._backup_agent and self._backup_agent != self._target_agent:
                result, backup_err = await self._call_agent(self._backup_agent, user_input)
                if backup_err is None:
                    info["answered_by"] = self._backup_agent
                else:
                    info["backup_error"] = backup_err
            if result is None:
                result = _unavailable(user_input, self._names[0])
                info["answered_by"] = None
        self._agent_health(err, info)
        if user_input.device_id:
            self._last_reply[user_input.device_id] = (time.monotonic(), _speech(result))
        return result, info

    def _is_open_mic_followup(self, device_id: str | None) -> bool:
        """True if this run is Home Assistant's open microphone after a reply that asked to
        keep listening (not a button press). The invitation is used up either way."""
        invited = self._invited.pop(device_id or "", None)
        if invited is None or time.monotonic() - invited > OPEN_MIC_WINDOW:
            return False
        return bool(self._run_after_reply.get(self._satellite(device_id) or ""))

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        if not self._is_speech_run(user_input.device_id):
            if self._is_open_mic_followup(user_input.device_id):
                # Judged like speech, with the follow-up check always asked.
                rec = await self._assess(user_input, invited=True)
                rec["kind"] = "open_mic"
                if not rec["answer"]:
                    self._log(rec)
                    return _silent(user_input)
                return await self._answer(user_input, rec)
            rec = {"kind": "passthrough", "device_id": user_input.device_id, "text": user_input.text}
            result, timer = await self._try_timer(user_input, user_input.text)
            if timer:
                rec["timer"] = timer
            if result is not None:
                rec["answered_by"] = "timers"
            else:
                result, info = await self._forward(user_input)
                rec.update(info)
            rec["reply"] = _speech(result)[:300]
            self._note_invitation(user_input, result, rec)
            self._log(rec)
            return result

        rec = await self._assess(user_input)
        if not rec["answer"]:
            self._log(rec)
            return _silent(user_input)
        return await self._answer(user_input, rec)

    def _note_invitation(
        self, user_input: conversation.ConversationInput, result: conversation.ConversationResult, rec: dict[str, Any]
    ) -> None:
        """The agent asked to keep listening: let HA reopen the microphone, and judge that run."""
        if result.continue_conversation and user_input.device_id:
            self._invited[user_input.device_id] = time.monotonic()
            rec["continue"] = True

    async def _answer(
        self, user_input: conversation.ConversationInput, rec: dict[str, Any]
    ) -> conversation.ConversationResult:
        self._show_thinking(user_input.device_id)
        start = time.monotonic()
        result, timer = await self._try_timer(user_input, rec["text"])
        if timer:
            rec["timer"] = timer
        if result is not None:
            rec["answered_by"] = "timers"
            info: dict[str, Any] = {}
        else:
            result, info = await self._forward(user_input)
        rec.update(info)
        rec["reply"] = _speech(result)[:300]
        rec["agent_ms"] = round((time.monotonic() - start) * 1000)
        self._note_invitation(user_input, result, rec)
        self._log(rec)
        return result

    # Timers ---------------------------------------------------------------------------

    async def _try_timer(
        self, user_input: conversation.ConversationInput, text: str
    ) -> tuple[conversation.ConversationResult | None, dict[str, Any] | None]:
        """Handle a Voice PE timer request here. Returns (result, log info); a None result
        means "pass it to the agent". Requests without a time word cost nothing."""
        if not self._handle_timers or self._classifier is None:
            return None, None
        request = _strip_name(text, self._names)
        if not mentions_time(request):
            return None, None
        answer = await self._classifier.timer_action(request)
        self._classifier_health({"timer": answer})
        info: dict[str, Any] = {"request": request, "ms": answer.get("ms")}
        if "error" in answer:
            info["error"] = answer["error"]
            return None, info
        action, p = answer["best"], answer["p"]
        info.update(action=action, p=p)
        if action not in TIMER_ACTIONS or p < self._timer_threshold:
            return None, info
        if action == "cancel" and is_all(request):
            action = "cancel_all"
        seconds = parse_duration(request) if action in NEEDS_DURATION else None
        if action in NEEDS_DURATION and not seconds:
            info["error"] = "no duration"
            return None, info
        name = parse_name(request) if action != "cancel_all" else None
        slots: dict[str, Any] = slots_for(seconds) if seconds else {}
        if name:
            slots["name"] = {"value": name}
        info.update(intent=ACTION_INTENTS[action], seconds=seconds, name=name)

        from homeassistant.components.intent.timers import (  # noqa: PLC0415
            MultipleTimersMatchedError,
            TimerNotFoundError,
            TimersNotSupportedError,
        )

        try:
            response = await intent.async_handle(
                self.hass,
                DOMAIN,
                ACTION_INTENTS[action],
                slots,
                request,
                user_input.context,
                user_input.language,
                "conversation",
                user_input.device_id,
                user_input.satellite_id,
                self.entity_id,
            )
        except TimersNotSupportedError:
            info["error"] = "device has no timers"
            return None, info
        except TimerNotFoundError:
            speech = f"I couldn't find the {name} timer." if name else "I couldn't find that timer."
            info["error"] = "timer not found"
        except MultipleTimersMatchedError:
            speech = "There's more than one timer. Please say which one."
            info["error"] = "several timers matched"
        except intent.IntentError as err:
            info["error"] = repr(err)[:200]
            return None, info
        else:
            speech = confirmation(action, seconds, name, response.speech_slots)
        result = _spoken(user_input, speech)
        if user_input.device_id:
            self._last_reply[user_input.device_id] = (time.monotonic(), speech)
        return result, info

    # Health: warnings and repair issues -----------------------------------------------

    def _agent_name(self, agent_id: str) -> str:
        state = self.hass.states.get(agent_id)
        return (state.attributes.get("friendly_name") if state else None) or agent_id

    def _agent_health(self, err: str | None, info: dict[str, Any]) -> None:
        if err is None:
            if self._agent_failures:
                _LOGGER.info("Smart Listening %s: %s is answering again", self._title, self._target_agent)
            self._agent_failures = 0
            if self._agent_issue or not self._agent_cleared:
                ir.async_delete_issue(self.hass, DOMAIN, issue_id("agent", self._entry_id))
                self._agent_issue = False
                self._agent_cleared = True
            return
        self._agent_failures += 1
        if self._agent_failures == 1:
            answered = info.get("answered_by")
            _LOGGER.warning(
                "Smart Listening %s: %s failed (%s); %s",
                self._title,
                self._target_agent,
                err,
                f"answered by the backup {answered}" if answered else f"replied that {self._names[0]} isn't available",
            )
        if not self._agent_issue and (err == "not found" or self._agent_failures >= AGENT_ISSUE_AFTER):
            self._raise_agent_issue()

    def _raise_agent_issue(self) -> None:
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            issue_id("agent", self._entry_id),
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="agent_unavailable",
            translation_placeholders={"assistant": self._title, "agent": self._agent_name(self._target_agent)},
        )
        self._agent_issue = True

    def _classifier_health(self, rec: dict[str, Any]) -> None:
        results = [r for r in (rec.get("address"), rec.get("followup"), rec.get("timer")) if r]
        if not results:
            return
        errors = [r["error"] for r in results if "error" in r]
        if not errors:
            if self._classifier_failures:
                _LOGGER.info("Smart Listening %s: the classifier is answering again", self._title)
            self._classifier_failures = 0
            if self._classifier_issue or not self._classifier_cleared:
                ir.async_delete_issue(self.hass, DOMAIN, issue_id("classifier", self._entry_id))
                self._classifier_issue = False
                self._classifier_cleared = True
            return
        self._classifier_failures += 1
        if self._classifier_failures == 1:
            _LOGGER.warning(
                "Smart Listening %s: the classifier failed (%s); judging by the name position until it answers",
                self._title,
                errors[0],
            )
        if not self._classifier_issue and self._classifier_failures >= CLASSIFIER_ISSUE_AFTER:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                issue_id("classifier", self._entry_id),
                is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key="classifier_unavailable",
                translation_placeholders={"assistant": self._title},
            )
            self._classifier_issue = True

    async def _assess(self, user_input: conversation.ConversationInput, invited: bool = False) -> dict[str, Any]:
        text = user_input.text.strip()
        score, name, pos, rest = _name_match(text, self._names)
        hint = self._wake_hint(user_input.device_id)
        if hint and score < NAME_THRESHOLD:
            # The start of the speech (the name) was clipped; the wake word model heard it.
            text = f"{hint}, {text}"
            score, name, pos, rest = _name_match(text, self._names)
        rec: dict[str, Any] = {
            "kind": "speech",
            "device_id": user_input.device_id,
            "text": text,
            "name": name,
            "name_score": score,
            "name_pos": pos,
            "wake_hint": hint,
            "classifier": self._classifier_kind,
        }

        prev = self._last_reply.get(user_input.device_id or "")
        in_followup = False
        if prev is not None:
            rec["since_reply"] = round(time.monotonic() - prev[0], 1)
            # An open-mic run after a reply that asked to keep listening is always a
            # possible follow-up, however long the reply took to play.
            in_followup = bool(prev[1]) and (invited or rec["since_reply"] <= self._followup_seconds)
            # An echo is the reply heard back soon after it was spoken. Speech that starts
            # with the name, or that the wake word model heard the name in, is a person
            # talking to the assistant ("Lorelai, set a timer for 10 seconds" after
            # "Timer set for 10 seconds."), not the reply.
            addressed = hint is not None or (score >= NAME_THRESHOLD and pos == 0)
            if (
                prev[1]
                and not addressed
                and rec["since_reply"] <= ECHO_WINDOW
                and difflib.SequenceMatcher(None, text.lower(), prev[1].lower()).ratio() >= ECHO_SIMILARITY
            ):
                return {**rec, "answer": False, "reason": "echo of last reply"}

        named = score >= NAME_THRESHOLD
        if named and rest.strip(" .,!?") in SHORT_COMMANDS:
            return {**rec, "answer": True, "reason": "short command"}

        if self._classifier is None:
            if named and 0 <= pos < NAME_ONLY_POSITION:
                return {**rec, "answer": True, "reason": "name at start"}
            return {**rec, "answer": False}

        if score >= NAME_ASK_THRESHOLD:
            rec["address"] = await self._classifier.address(text, self._assistant)
        if in_followup:
            rec["followup"] = await self._classifier.followup(prev[1], text, self._assistant)
        self._classifier_health(rec)

        yes = (rec.get("address") or {}).get("p", 0.0)
        fup = (rec.get("followup") or {}).get("p", 0.0)
        if named and "error" in (rec.get("address") or {}):
            return {**rec, "answer": 0 <= pos < NAME_ONLY_POSITION, "reason": "classifier failed, name position"}
        if named and yes >= self._address_threshold:
            return {**rec, "answer": True, "reason": "name + classifier"}
        if hint and named and yes >= self._wake_word_threshold:
            # The wake word model heard the name too: direct address is very likely.
            return {**rec, "answer": True, "reason": "wake word + classifier"}
        if fup >= self._followup_threshold:
            return {**rec, "answer": True, "reason": "follow-up"}
        return {**rec, "answer": False}

    # Decision log ---------------------------------------------------------------------

    def _log(self, rec: dict[str, Any]) -> None:
        _LOGGER.debug("decision: %s", rec)
        if not self._decision_log:
            return
        line = json.dumps(
            {"ts": dt_util.now().isoformat(timespec="seconds"), "agent": self.entity_id, **rec},
            ensure_ascii=False,
        )
        self.hass.async_create_background_task(self._write(line), "smart_listening log")

    async def _write(self, line: str) -> None:
        try:
            await self.hass.async_add_executor_job(_append, self.hass.config.path(LOG_FILE), line)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Could not write the decision log")
