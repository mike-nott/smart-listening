"""Constants for the Smart Listening integration."""

from __future__ import annotations

DOMAIN = "smart_listening"

# Config entry options
CONF_TARGET_AGENT = "target_agent"
# Tried when the target agent is missing, raises, times out or reports a server failure.
CONF_BACKUP_AGENT = "backup_agent"
# Seconds an agent may take before it counts as failed.
CONF_AGENT_TIMEOUT = "agent_timeout"
CONF_NAMES = "assistant_names"
CONF_CLASSIFIER = "classifier"
CONF_CLASSIFIER_URL = "classifier_url"
CONF_CLASSIFIER_MODEL = "classifier_model"
CONF_CLASSIFIER_API_KEY = "classifier_api_key"
CONF_CLASSIFIER_ACCOUNT = "cloudflare_account_id"
# Self-hosted Clef: the request field that carries each question's text. New entries use
# the documented "instructions"; entries without this option (made before 0.2.0) keep
# "question", which their thresholds were tuned on. Clef scores the two differently.
CONF_CLEF_FIELD = "clef_question_field"
CONF_ADDRESS_THRESHOLD = "address_threshold"
CONF_WAKE_WORD_THRESHOLD = "wake_word_threshold"
CONF_FOLLOWUP_THRESHOLD = "followup_threshold"
CONF_FOLLOWUP_SECONDS = "followup_seconds"
CONF_DECISION_LOG = "decision_log"

# Classifier backends
CLASSIFIER_CLEF = "clef"
CLASSIFIER_CLEF_CLOUD = "clef_cloud"
CLASSIFIER_JEV = "jev"
CLASSIFIER_OPENAI = "openai"
CLASSIFIER_NAME_ONLY = "name_only"
CLASSIFIERS = [CLASSIFIER_CLEF, CLASSIFIER_CLEF_CLOUD, CLASSIFIER_JEV, CLASSIFIER_OPENAI, CLASSIFIER_NAME_ONLY]
# Shown as the agent's device model.
CLASSIFIER_DISPLAY_NAMES = {
    CLASSIFIER_CLEF: "Clef",
    CLASSIFIER_CLEF_CLOUD: "Clef on Cloudflare",
    CLASSIFIER_JEV: "Jev",
    CLASSIFIER_OPENAI: "OpenAI-compatible",
    CLASSIFIER_NAME_ONLY: "Name only",
}

# TypeSafe Jev (hosted System One API). The model is pinned to a version, as TypeSafe
# advises when thresholds are tuned: the jev-latest alias moves with each release.
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.14.0"

# Clef on Cloudflare Workers AI (same System One request body, wrapped in Cloudflare's
# usual {"result": ...} reply). "clef" (27B) or "clef-flash" (9B).
CLEF_CLOUD_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
CLEF_CLOUD_MODEL = "clef"
CLEF_CLOUD_MODELS = ("clef", "clef-flash")

# Defaults that differ by classifier (measured on the 2026-11-04 test set).
CLASSIFIER_DEFAULTS = {
    # Self-hosted Clef with the documented "instructions" field, kind question first:
    # 34/38 requests, 1/28 named side talk at 0.5 (NVFP4 build of Cloudflare/clef).
    CLASSIFIER_CLEF: {"address_threshold": 0.5, CONF_CLEF_FIELD: "instructions"},
    CLASSIFIER_JEV: {"address_threshold": 0.7},
    # The same model and field as self-hosted Clef above (not measured on Workers AI itself).
    CLASSIFIER_CLEF_CLOUD: {"address_threshold": 0.5},
}

DEFAULTS = {
    CONF_BACKUP_AGENT: "",
    CONF_AGENT_TIMEOUT: 60,
    CONF_CLASSIFIER: CLASSIFIER_NAME_ONLY,
    CONF_CLASSIFIER_URL: "",
    CONF_CLASSIFIER_MODEL: "",
    CONF_CLASSIFIER_API_KEY: "",
    CONF_CLASSIFIER_ACCOUNT: "",
    # Classifier probability that the speech is a request for the assistant.
    CONF_ADDRESS_THRESHOLD: 0.8,
    # Lower bar when the device's wake word model also heard the name.
    CONF_WAKE_WORD_THRESHOLD: 0.6,
    # Classifier probability that speech after a reply continues the conversation.
    CONF_FOLLOWUP_THRESHOLD: 0.8,
    # Speech this soon after a reply is checked as a possible follow-up (0 turns it off).
    CONF_FOLLOWUP_SECONDS: 20,
    # Write every decision, with its transcript, to LOG_FILE. Off by default: it records
    # what people in the house say.
    CONF_DECISION_LOG: False,
}

CLASSIFIER_TIMEOUT = 30

# Spoken when neither the target agent nor the backup could answer an accepted request.
UNAVAILABLE_MESSAGE = "Sorry, {name} isn't available right now."
# Consecutive failures before a repair issue appears (a missing agent raises one at once).
AGENT_ISSUE_AFTER = 3
CLASSIFIER_ISSUE_AFTER = 3
# After Home Assistant starts, check that the target agent exists (agents load late).
STARTUP_CHECK_DELAY = 120

# Entities from the Voice PE firmware package, matched on the end of their unique id
# ("<mac>/0/<platform>/<entity name>"), so renamed entity ids keep working.
DEVICE_SWITCH = "/switch/Smart Listening"
DEVICE_RUN = "/binary_sensor/Smart Listening run"
DEVICE_WAKE_WORD = "/text_sensor/Smart Listening wake word"
DEVICE_THINKING = "/button/Smart Listening thinking"
# With the device switch off, a run flag older than this is treated as stale.
RUN_MAX_AGE = 30

# Fuzzy match score that counts as the assistant's name being said.
NAME_THRESHOLD = 0.75
# The classifier is only asked when the name score reaches this (or in the follow-up window).
NAME_ASK_THRESHOLD = 0.5
# Without a classifier (or when it fails), answer only when the name is in the first words.
NAME_ONLY_POSITION = 3
# Speech this similar to the last reply is the speaker being overheard.
ECHO_SIMILARITY = 0.6

# Short commands after the name that a classifier scores low but should always be answered.
SHORT_COMMANDS = (
    "stop", "cancel", "pause", "resume", "goodnight", "good night", "never mind",
    "nevermind", "quiet", "be quiet", "shut up", "louder", "quieter", "volume up",
    "volume down", "turn it up", "turn it down", "thanks", "thank you",
)

LOG_FILE = "smart_listening/decisions.jsonl"
LOG_MAX_BYTES = 20_000_000
