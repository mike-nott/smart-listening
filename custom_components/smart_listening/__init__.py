"""The Smart Listening integration.

Voice PE devices running the Smart Listening firmware package start a request whenever
they hear speech, with no wake word. Each config entry adds a conversation agent that
decides whether that speech was meant for its assistant and, if so, passes it on.
"""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.typing import ConfigType
import voluptuous as vol

from .const import DOMAIN, SERVICE_REPLY

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS = [Platform.CONVERSATION]

REPLY_SCHEMA = vol.Schema(
    {
        vol.Required("message"): cv.string,
        vol.Optional("area"): cv.string,
        vol.Optional("satellite"): cv.entity_id,
        vol.Optional("listen"): cv.boolean,
        vol.Optional("agent"): cv.entity_id,
    }
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register smart_listening.reply: late replies from an agent, on the room's reply speaker."""

    async def _reply(call: ServiceCall) -> ServiceResponse:
        return await async_reply(hass, dict(call.data))

    hass.services.async_register(
        DOMAIN, SERVICE_REPLY, _reply, schema=REPLY_SCHEMA, supports_response=SupportsResponse.OPTIONAL
    )
    return True


async def async_reply(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Find the Voice PE (and the Smart Listening agent it uses) and say the message there."""
    agents = dict(hass.data.get(DOMAIN, {}).get("agents", {}))
    wanted = data.get("agent")
    if wanted:
        if wanted not in agents:
            raise ServiceValidationError(f"{wanted} is not a Smart Listening agent")
        agents = {wanted: agents[wanted]}
    if not agents:
        raise ServiceValidationError("No Smart Listening agent is set up")
    area, satellite = data.get("area"), data.get("satellite")
    found = [(agent, hit) for agent in agents.values() if (hit := agent.find_voice_pe(area, satellite))]
    if not found:
        where = f"area {area}" if area else satellite or "a recent voice request"
        raise ServiceValidationError(f"No Voice PE found for {where}")
    agent, (device_id, sat, _age) = min(found, key=lambda f: f[1][2])  # most recent request
    return await agent.async_late_reply(device_id, sat, data["message"], data.get("listen"), area)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a Smart Listening agent from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))
    return True


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a Smart Listening agent."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """A deleted agent leaves no repair issues behind."""
    for kind in ("agent", "classifier"):
        ir.async_delete_issue(hass, DOMAIN, f"{kind}_unavailable_{entry.entry_id}")
