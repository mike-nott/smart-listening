"""Config and options flow for Smart Listening.

One entry per assistant: the conversation agent that answers, the names it answers to,
and how overheard speech is judged.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import selector

from .classifier import make_classifier
from .const import (
    CLASSIFIER_CLEF,
    CLASSIFIER_CLEF_CLOUD,
    CLASSIFIER_DEFAULTS,
    CLASSIFIER_JEV,
    CLASSIFIER_NAME_ONLY,
    CLASSIFIER_OPENAI,
    CLASSIFIERS,
    CLEF_CLOUD_MODEL,
    CLEF_CLOUD_MODELS,
    CLEF_CLOUD_URL,
    CONF_ADDRESS_THRESHOLD,
    CONF_CLASSIFIER,
    CONF_CLASSIFIER_ACCOUNT,
    CONF_CLASSIFIER_API_KEY,
    CONF_CLASSIFIER_MODEL,
    CONF_CLASSIFIER_URL,
    CONF_CLEF_FIELD,
    CONF_DECISION_LOG,
    CONF_FOLLOWUP_SECONDS,
    CONF_FOLLOWUP_THRESHOLD,
    CONF_NAMES,
    CONF_TARGET_AGENT,
    CONF_WAKE_WORD_THRESHOLD,
    DEFAULTS,
    DOMAIN,
    JEV_MODEL,
    JEV_URL,
)
from .conversation import parse_names

_AGENT = selector.EntitySelector(selector.EntitySelectorConfig(domain="conversation"))
_TEXT = selector.TextSelector()
_URL = selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL))
_PASSWORD = selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD))
_CLASSIFIER = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=CLASSIFIERS, translation_key="classifier", mode=selector.SelectSelectorMode.LIST
    )
)
_PROBABILITY = selector.NumberSelector(
    selector.NumberSelectorConfig(min=0, max=1, step=0.01, mode=selector.NumberSelectorMode.BOX)
)
_SECONDS = selector.NumberSelector(
    selector.NumberSelectorConfig(
        min=0, max=120, step=1, unit_of_measurement="s", mode=selector.NumberSelectorMode.BOX
    )
)
_BOOL = selector.BooleanSelector()


def _validate_basics(data: dict[str, Any], errors: dict[str, str]) -> None:
    if not parse_names(data.get(CONF_NAMES)):
        errors[CONF_NAMES] = "no_names"
    if data.get(CONF_TARGET_AGENT, "").startswith("conversation.smart_listening"):
        errors[CONF_TARGET_AGENT] = "target_is_self"


async def _validate_classifier(hass, data: dict[str, Any], errors: dict[str, str]) -> None:
    kind = data.get(CONF_CLASSIFIER, CLASSIFIER_NAME_ONLY)
    if kind == CLASSIFIER_NAME_ONLY:
        return
    if kind == CLASSIFIER_CLEF_CLOUD:
        account = data.get(CONF_CLASSIFIER_ACCOUNT, "").strip()
        model = data.get(CONF_CLASSIFIER_MODEL, "").strip() or CLEF_CLOUD_MODEL
        if not account:
            errors[CONF_CLASSIFIER_ACCOUNT] = "no_account"
            return
        if model not in CLEF_CLOUD_MODELS:
            errors[CONF_CLASSIFIER_MODEL] = "invalid_clef_model"
            return
        if not data.get(CONF_CLASSIFIER_API_KEY, "").strip():
            errors[CONF_CLASSIFIER_API_KEY] = "no_api_key"
            return
        data[CONF_CLASSIFIER_ACCOUNT] = account
        data[CONF_CLASSIFIER_MODEL] = model
        data[CONF_CLASSIFIER_URL] = CLEF_CLOUD_URL.format(account=account, model=model)
    if kind == CLASSIFIER_JEV:
        data.setdefault(CONF_CLASSIFIER_URL, "")
        data[CONF_CLASSIFIER_URL] = data[CONF_CLASSIFIER_URL] or JEV_URL
        data[CONF_CLASSIFIER_MODEL] = data.get(CONF_CLASSIFIER_MODEL) or JEV_MODEL
        if not data.get(CONF_CLASSIFIER_API_KEY, "").strip():
            errors[CONF_CLASSIFIER_API_KEY] = "no_api_key"
            return
    if not data.get(CONF_CLASSIFIER_URL, "").startswith(("http://", "https://")):
        errors[CONF_CLASSIFIER_URL] = "invalid_url"
        return
    if kind == CLASSIFIER_OPENAI and not data.get(CONF_CLASSIFIER_MODEL, "").strip():
        errors[CONF_CLASSIFIER_MODEL] = "no_model"
        return
    classifier = make_classifier(
        hass,
        kind,
        data.get(CONF_CLASSIFIER_URL, ""),
        data.get(CONF_CLASSIFIER_MODEL, ""),
        data.get(CONF_CLASSIFIER_API_KEY, ""),
        data.get(CONF_CLEF_FIELD, "question"),
    )
    try:
        await classifier.check()
    except Exception:  # noqa: BLE001
        errors["base"] = "cannot_connect"


def _switch_classifier(current: dict[str, Any], candidate: dict[str, Any]) -> None:
    """When the options change to a different classifier, drop the old one's endpoint
    (Jev and Clef on Cloudflare fill in their own) and use the new one's measured request
    threshold, unless that threshold was changed on the same screen."""
    kind = candidate[CONF_CLASSIFIER]
    if kind == current[CONF_CLASSIFIER]:
        return
    if kind in (CLASSIFIER_JEV, CLASSIFIER_CLEF_CLOUD):
        for key in (CONF_CLASSIFIER_URL, CONF_CLASSIFIER_MODEL):
            if candidate.get(key) == current.get(key):
                candidate[key] = ""
    defaults = CLASSIFIER_DEFAULTS.get(kind, {})
    if candidate[CONF_ADDRESS_THRESHOLD] == current[CONF_ADDRESS_THRESHOLD]:
        candidate[CONF_ADDRESS_THRESHOLD] = defaults.get(CONF_ADDRESS_THRESHOLD, DEFAULTS[CONF_ADDRESS_THRESHOLD])
    for key, value in defaults.items():
        if key != CONF_ADDRESS_THRESHOLD:
            candidate[key] = value


class SmartListeningConfigFlow(ConfigFlow, domain=DOMAIN):
    """Add a Smart Listening agent."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return SmartListeningOptionsFlow()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            _validate_basics(user_input, errors)
            if not errors:
                self._data = {**DEFAULTS, **user_input}
                if user_input[CONF_CLASSIFIER] == CLASSIFIER_NAME_ONLY:
                    return self._create()
                return await self.async_step_classifier()
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_TARGET_AGENT): _AGENT,
                        vol.Required(CONF_NAMES): _TEXT,
                        vol.Required(CONF_CLASSIFIER, default=CLASSIFIER_CLEF): _CLASSIFIER,
                    }
                ),
                user_input or {},
            ),
            errors=errors,
        )

    async def async_step_classifier(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        kind = self._data[CONF_CLASSIFIER]
        if user_input is not None:
            candidate = {**self._data, **user_input}
            await _validate_classifier(self.hass, candidate, errors)
            if not errors:
                self._data = candidate
                return self._create()
        if kind == CLASSIFIER_JEV:
            fields: dict[Any, Any] = {
                vol.Required(CONF_CLASSIFIER_API_KEY): _PASSWORD,
                vol.Required(CONF_CLASSIFIER_MODEL): _TEXT,
                vol.Required(CONF_CLASSIFIER_URL): _URL,
            }
            suggested = {CONF_CLASSIFIER_MODEL: JEV_MODEL, CONF_CLASSIFIER_URL: JEV_URL}
        elif kind == CLASSIFIER_CLEF_CLOUD:
            fields = {
                vol.Required(CONF_CLASSIFIER_ACCOUNT): _TEXT,
                vol.Required(CONF_CLASSIFIER_API_KEY): _PASSWORD,
                vol.Required(CONF_CLASSIFIER_MODEL): _TEXT,
            }
            suggested = {CONF_CLASSIFIER_MODEL: CLEF_CLOUD_MODEL}
        else:
            fields = {vol.Required(CONF_CLASSIFIER_URL): _URL}
            suggested = {}
            if kind == CLASSIFIER_CLEF:
                fields[vol.Optional(CONF_CLASSIFIER_MODEL)] = _TEXT
                suggested = {CONF_CLASSIFIER_MODEL: "clef"}
            if kind == CLASSIFIER_OPENAI:
                fields[vol.Required(CONF_CLASSIFIER_MODEL)] = _TEXT
                fields[vol.Optional(CONF_CLASSIFIER_API_KEY)] = _PASSWORD
        return self.async_show_form(
            step_id="classifier",
            data_schema=self.add_suggested_values_to_schema(vol.Schema(fields), {**suggested, **(user_input or {})}),
            description_placeholders={"classifier": kind},
            errors=errors,
        )

    def _create(self) -> ConfigFlowResult:
        self._data.update(CLASSIFIER_DEFAULTS.get(self._data[CONF_CLASSIFIER], {}))
        title = parse_names(self._data[CONF_NAMES])[0]
        self._data[CONF_NAMES] = ", ".join(parse_names(self._data[CONF_NAMES]))
        return self.async_create_entry(title=title, data={}, options=self._data)


class SmartListeningOptionsFlow(OptionsFlow):
    """Change an agent's settings (the entry reloads; no restart needed)."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        current = {**DEFAULTS, **self.config_entry.options}
        if user_input is not None:
            candidate = {**current, **user_input}
            _switch_classifier(current, candidate)
            _validate_basics(candidate, errors)
            if not errors:
                await _validate_classifier(self.hass, candidate, errors)
            if not errors:
                candidate[CONF_NAMES] = ", ".join(parse_names(candidate[CONF_NAMES]))
                return self.async_create_entry(data=candidate)
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_TARGET_AGENT): _AGENT,
                        vol.Required(CONF_NAMES): _TEXT,
                        vol.Required(CONF_CLASSIFIER): _CLASSIFIER,
                        vol.Optional(CONF_CLASSIFIER_URL): _URL,
                        vol.Optional(CONF_CLASSIFIER_ACCOUNT): _TEXT,
                        vol.Optional(CONF_CLASSIFIER_MODEL): _TEXT,
                        vol.Optional(CONF_CLASSIFIER_API_KEY): _PASSWORD,
                        vol.Required(CONF_ADDRESS_THRESHOLD): _PROBABILITY,
                        vol.Required(CONF_WAKE_WORD_THRESHOLD): _PROBABILITY,
                        vol.Required(CONF_FOLLOWUP_THRESHOLD): _PROBABILITY,
                        vol.Required(CONF_FOLLOWUP_SECONDS): _SECONDS,
                        vol.Required(CONF_DECISION_LOG): _BOOL,
                    }
                ),
                user_input or current,
            ),
            errors=errors,
        )
