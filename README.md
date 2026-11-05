# Smart Listening for Home Assistant

A Home Assistant custom integration and Voice PE firmware package that removes the wake word. You talk to your assistant the way you would talk to a person in the room ("Nova, what time is it?"), with no chime and no pause. The Voice PE sends the speech it hears to speech-to-text, and Smart Listening decides which of it was meant for the assistant. Only that speech reaches your conversation agent; the rest is dropped silently.

Sibling projects: [MCP Assist](https://github.com/mike-nott/mcp-assist) and [Smart Announcements](https://github.com/mike-nott/smart-announcements).

## Key Features

- ✅ **No Wake Word, No Chime**: Say the assistant's name anywhere in a sentence, at normal speed
- ✅ **Ghost Wake Protection**: TV, radio and conversations that mention the name are judged before anything answers
- ✅ **Fuzzy Name Matching**: Transcription slips such as "Novva" still count as the name
- ✅ **Wake Word Hint**: When speech detection clips the start of a sentence, the device's wake word model tells Smart Listening the name was said
- ✅ **Natural Follow-ups**: The Voice PE listens again straight after each answer, so you can carry on without the name; each follow-up is judged too
- ✅ **Classic Wake Word Too**: Say just "Nova" and pause: it answers "How can I help?" and listens
- ✅ **Echo Check**: The assistant's own reply, heard back through the microphone, is ignored
- ✅ **Short Commands Always Work**: "Nova, stop", "Nova, louder" and similar are answered without a model call
- ✅ **Honest LEDs**: The thinking ring lights only once a request has been accepted, so the device stays dark while it ignores the TV
- ✅ **Choice of Classifier**: Clef (self-hosted or on Cloudflare), Jev (TypeSafe cloud), any OpenAI-compatible endpoint, or no model at all
- ✅ **Several Assistants**: One config entry per assistant, each with its own names and conversation agent
- ✅ **Per-Device Off Switch**: Turn Smart Listening off on any Voice PE to get the stock wake word behaviour back
- ✅ **Voice PE Timers**: "Nova, set a timer for 10 seconds" starts the Voice PE's own timer (LED countdown, ring) in about a tenth of a second, without the conversation agent, and loose wording works too
- ✅ **Backup Agent**: If the main conversation agent is down, requests go to a backup; failing that, the assistant says it isn't available instead of staying silent
- ✅ **Visible Failures**: An agent or classifier that stops answering shows under Settings → Repairs until it recovers
- ✅ **Decision Log**: Optional log of every decision, with scores, for tuning

## The Problem Smart Listening Solves

Wake words are a compromise:
- They trigger on things that only sound like the name (ghost wakes), often from the TV
- You have to say them first, wait for the chime, then speak
- A sensitive model wakes too often; a strict one misses you
- They cannot tell "Nova, turn the lights off" from "I asked Nova to turn the lights off"

Smart Listening hears the whole sentence before deciding, so it can tell the difference.

## How Smart Listening Works

1. **The Voice PE hears speech.** With the Smart Listening switch on, the device's voice activity detector starts a request whenever someone speaks. No wake word, no chime, no LEDs.
2. **Home Assistant transcribes it** with your speech-to-text engine, as for any Assist request.
3. **Smart Listening judges the transcript**, in this order:
   - Did the device's wake word model trigger during this speech? Then the name was said, even if the transcript lost it.
   - Is this the assistant's last reply, heard back? Ignore it.
   - Is it the name followed by a short command ("stop", "cancel", "thank you")? Answer it.
   - Does the transcript contain the name (fuzzy match)? If not, and it is outside the follow-up window, ignore it. No model is called.
   - Ask the classifier "Is this a request for Nova?", and within the follow-up window "Does this continue what Nova just said?"
   - Answer only if the classifier's score passes the threshold.
4. **Accepted speech** turns on the thinking LEDs and goes to your conversation agent. The reply plays as normal.
5. **Ignored speech** gets an empty reply: no sound, no LEDs, and no chat history.

Requests that do not come from speech detection (the centre button, or a wake word on a device with Smart Listening switched off) pass straight through to your conversation agent without being judged.

## Requirements

- Home Assistant 2025.8 or newer
- A Home Assistant Voice PE with its firmware built from YAML in the ESPHome Device Builder add-on (an adopted device)
- A speech-to-text engine (local Whisper or Speech-to-Phrase, Home Assistant Cloud, etc.). Every unmuted device sends all the speech it hears, so a local engine needs headroom
- A conversation agent to answer requests (MCP Assist, OpenAI, Ollama, the built-in Assist agent, etc.)
- Optional: a classifier (see [Classifiers](#classifiers))

## Installation

### Option A: HACS (Recommended)

Smart Listening is in the HACS default repositories.

1. In Home Assistant, open **HACS** and search for **Smart Listening**
2. Open it and select **Download**
3. Restart Home Assistant

[![Open your Home Assistant instance and open this repository in HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mike-nott&repository=smart-listening&category=integration)

If Smart Listening does not show up in the search, add it as a custom repository instead: in HACS, open the **⋮** menu → **Custom repositories**, enter `https://github.com/mike-nott/smart-listening` with the type **Integration**, then download it as above.

### Option B: Manual
1. Copy `custom_components/smart_listening` into your Home Assistant `config/custom_components/` folder
2. Restart Home Assistant

## Setup

### 1. Add a Smart Listening agent

Go to **Settings → Devices & services → Add integration → Smart Listening** and choose:
- **Conversation agent**: the agent that answers accepted requests
- **Assistant names**: the names the assistant answers to, separated by commas, for example `Nova, Hey Nova`. The first is the main name
- **Classifier**: how speech containing the name is judged

This creates a conversation agent called **Smart Listening - Nova**.

### 2. Create an Assist pipeline

Go to **Settings → Voice assistants → Add assistant**. Pick your speech-to-text and text-to-speech engines as usual, and set the **Conversation agent** to **Smart Listening - Nova**.

If you rename this pipeline later, re-select it on every Voice PE straight away: devices fall back to the preferred pipeline when the name they stored no longer exists.

### 3. Add the firmware package

1. Copy [`esphome/smart-listening.yaml`](esphome/smart-listening.yaml) to `/config/esphome/packages/smart-listening.yaml`
2. In the Device Builder, edit each Voice PE and add the package **after** the stock package:

   ```yaml
   packages:
     voice_pe: github://esphome/home-assistant-voice-pe/home-assistant-voice.yaml@26.9.0
     smart_listening: !include packages/smart-listening.yaml
   ```

3. Install the firmware

The package is written against stock firmware 26.9.0. It replaces some of the stock package's handlers, so check it again before moving to a newer stock version.

### 4. Turn it on

On each Voice PE's device page:
- Set **Assistant** to the pipeline from step 2
- If you have a wake word model for the assistant's name, choose it as the **Wake word**: it is used as a hint
- Turn on the **Smart Listening** switch. It starts off after every firmware install

## The Firmware Package

The package adds four entities to each Voice PE:

| Entity | Purpose |
| --- | --- |
| `switch` Smart Listening | On: requests start on speech. Off: stock wake word behaviour |
| `binary_sensor` Smart Listening run | On while a request started by speech detection is running |
| `text_sensor` Smart Listening wake word | The wake word heard during the current speech, if any |
| `button` Smart Listening thinking | Pressed by the integration to show the thinking LEDs once it accepts a request |

With the switch on:
- Speech starts a request silently, without the chime or the listening LEDs
- A wake word no longer starts a request of its own. It still stops a ringing timer or an announcement, and otherwise tells Smart Listening the name was said
- The wake word models' cutoffs drop to 0.80, since a false trigger now costs nothing: the integration still decides. The package does this for the stock models (Okay Nabu, Hey Jarvis, Hey Mycroft); if you use your own models, edit the `smart_listening_cutoffs` script
- A single press of the centre button during a speech run starts a normal request instead
- Mute, timers and announcements work as stock

## Configuration

Open **Smart Listening → Configure** on the agent to change any setting. Changes apply straight away, without a restart.

| Setting | Default | What it does |
| --- | --- | --- |
| Conversation agent | | The agent that answers accepted requests |
| Backup conversation agent | None | Answers when the main agent is missing, fails, times out or reports a server error |
| Agent timeout | 60 s | How long an agent may take to answer before it counts as failed |
| Handle Voice PE timers | On | Timer commands are handled by Smart Listening on the Voice PE that heard them (see [Timers](#timers)) |
| Timer threshold | 0.6 | How sure the classifier must be that a request is a timer command |
| Assistant names | | Names the assistant answers to, comma separated |
| Classifier | Clef | How speech containing the name is judged |
| Request threshold | Clef 0.5, Jev 0.7, others 0.8 | How sure the classifier must be that speech containing the name is a request |
| Request threshold when the wake word was heard | 0.6 | Used instead when the device's wake word model also heard the name |
| Follow-up threshold | 0.8 | How sure the classifier must be that speech after a reply continues the conversation |
| Keep listening after every answer | On | The Voice PE listens again straight after each answer; what it hears is judged as a follow-up. Off: only when the agent asks a question |
| Follow-up window | 20 s | How long after a reply speech without the name is checked as a follow-up. 0 turns follow-ups off |
| Decision log | Off | Writes every decision and its transcript to `config/smart_listening/decisions.jsonl` |

### Classifiers

The classifier is only asked about speech that contains the name (or something close to it), and about speech inside the follow-up window. Everything else is ignored without a model call.

[Clef](https://huggingface.co/Cloudflare/clef) is Cloudflare's open-weight decision model (Apache 2.0), API-compatible with TypeSafe's Jev. You can run it yourself or use it on Cloudflare Workers AI. Both use the same model and the same request, so the same default threshold.

#### Clef (self-hosted)
Download [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) from Hugging Face and serve it with SGLang, as the model card describes; the server answers at `/v1/systemone`. The full model needs a large GPU (around 55 GB of weights in BF16); community quantised builds, such as NVFP4, need less. Enter the full decision endpoint, for example `http://192.168.1.20:30000/v1/systemone`, and the model name your server uses (`Cloudflare/clef` on SGLang). Typically 70 to 100 ms per decision on a local GPU, and nothing leaves your network.

#### Clef on Cloudflare Workers AI
The same model, hosted by Cloudflare (`@cf/cloudflare/clef`), for when you do not have a GPU for it. You need your Cloudflare account ID and an API token with Workers AI permission. Choose `clef`, or `clef-flash` for the smaller, cheaper model. Cloudflare lists Clef at $0.24 per million input tokens with zero data retention. Its speed and scores on Workers AI have not been measured for Smart Listening yet; the threshold is the one measured on the self-hosted weights, and `clef-flash` may need its own.

Transcripts that contain the name, and speech inside the follow-up window, are sent to Cloudflare. If Workers AI cannot be reached, Smart Listening falls back to the name-only rule below.

#### Jev (TypeSafe, cloud)
TypeSafe's hosted System One model, for when you do not run a decision server yourself. You need an API key from TypeSafe. The model is pinned to `jev-1.14.0`, because `jev-latest` changes with each release and thresholds are tuned to one version. Around 150 to 180 ms per decision. Pricing is per input token (output is free), which comes to pennies a month for a household.

Transcripts that contain the name, and speech inside the follow-up window, are sent to TypeSafe. If Jev cannot be reached, Smart Listening falls back to the name-only rule below.

#### OpenAI-compatible endpoint
Any chat completions API: Ollama, vLLM, LM Studio, or a cloud provider. Enter the base URL (for example `http://192.168.1.20:11434/v1`), the model name and, if needed, an API key. The model is asked for a JSON answer with a confidence score. This is slower than a dedicated decision model and its scores are less well calibrated, so expect to tune the thresholds.

#### Name only
No model. Speech is answered when the name is among its first three words. There are no follow-ups. This is the simplest option, and it still ignores everything that does not start with the name.

### How the classifiers compare

Measured on 66 real transcripts from one home: 38 requests to the assistant, and 28 lines of side talk that mention the assistant by name ("did you ask Nova to set the timer?"). Speech without the name never reaches the classifier, so it is not in this set.

| Classifier | Threshold | Requests answered | Side talk answered |
| --- | --- | --- | --- |
| Jev `jev-1.14.0` | 0.7 | 37 of 38 | 1 of 28 |
| Jev `jev-1.14.0` | 0.6 | 38 of 38 | 2 of 28 |
| Clef (self-hosted, NVFP4 build) | 0.5 | 34 of 38 | 1 of 28 |

On follow-ups, Jev accepted all 7 genuine follow-ups and none of the 5 unrelated lines, at any threshold from 0.5 to 0.8.

### Tuning the thresholds

- **It answers things it should not**: raise the request threshold, or the wake word threshold if the decision log shows a `wake_hint` on those lines
- **It misses requests**: lower the request threshold a little, and check the transcripts: a request that never contains the name cannot be answered
- **Follow-ups are missed or picked up wrongly**: adjust the follow-up threshold, or shorten the window

Turn on the decision log while tuning. Each line holds the transcript, the name match, the classifier's scores and the decision.

### Timers

Home Assistant's Voice PE timers (the LED ring countdown and the ring when it ends) only start when Home Assistant's own timer command runs for that Voice PE. LLM agents usually can't do that, so Smart Listening handles timers itself:

1. An accepted request that mentions a time ("timer", "minutes", "how long", "half an hour"...) is put to the classifier as one question: start, cancel, add time, take off time, pause, resume, how long's left, a reminder, or something else. Requests without a time word go straight to the conversation agent, with no extra delay.
2. If it's a timer command with a score of at least the timer threshold, Smart Listening reads the duration ("10 seconds", "an hour and a half", "ninety seconds") and the timer's name ("the pasta timer"), runs the timer command on the Voice PE that heard it, and replies ("Timer set for 10 seconds.").
3. Anything else goes to the conversation agent as usual: reminders with a message ("remind me in 15 minutes to take the washing out"), questions that only mention a time, low scores, or a duration it can't read.

Loose wording works: "time the eggs, 6 minutes", "count down three minutes", "give the timer another two minutes", "how long until the pasta's done?". Named timers, "cancel all the timers", and pressing the Voice PE's button and asking all work too. To stop a ringing timer, say the assistant's name or press the button, as usual.

Measured on 60 labelled requests (40 timer commands, 5 reminders, 15 that only mention a time):

| Classifier | Time per check | Right | Wrong timer action at 0.6 | Timer commands left to the agent |
| --- | --- | --- | --- | --- |
| Clef (self-hosted) | 68 ms (median) | 53/60 | 0 | 1 |
| Jev `jev-1.14.0` | 186 ms (median) | 58/60 | 0 | 2 |

Timers need a classifier (not "name only") and work in English only. Requests in other languages go to the agent.

### When the conversation agent is down

If the agent that answers requests can't be reached, Smart Listening does not leave you in silence:

1. **The backup agent answers**, if one is set. Use one that does not depend on the same server, for example a local model if your main agent is in the cloud.
2. **Otherwise the assistant says** "Sorry, Nova isn't available right now."

An agent counts as failed when it is missing (for example, its integration has not loaded), raises an error, takes longer than the agent timeout, or replies with an `unknown` error, which is how LLM agents such as MCP Assist report a server failure. Other errors, such as "there is no device called that", are real answers: they are spoken as they are, and the request is not sent again.

While the main agent keeps failing, **Settings → Repairs** shows "Nova: \<agent\> isn't answering", and Home Assistant's log has one warning. Both clear themselves the next time the agent answers. The same happens when the classifier stops answering: speech is judged by the name position until it recovers.

The decision log records which agent answered (`answered_by`) and why the others failed (`agent_error`, `backup_error`).

## Privacy

Smart Listening changes what your Voice PEs send off the device:
- **Every unmuted Voice PE transcribes all the speech it hears**, not just requests. Transcripts come from your speech-to-text engine: use a local one if that matters to you, and mute devices in rooms where you do not want this
- **Transcripts that contain the name**, and speech shortly after a reply, go to the classifier. With self-hosted Clef or a local model they stay on your network; with Clef on Cloudflare, Jev or a cloud endpoint they do not
- **Ignored speech is not stored** in Home Assistant: no reply and no chat history
- **The decision log stores transcripts** of everything judged. It is off by default; delete `config/smart_listening/decisions.jsonl` when you have finished tuning

## Troubleshooting

### Nothing answers
- Is the **Smart Listening** switch on? It turns off after every firmware install
- Is the device's **Assistant** set to the pipeline that uses the Smart Listening agent? If the pipeline was renamed, re-select it
- Turn on the decision log and speak. No new line means the request never reached Smart Listening (check the pipeline); a line with `"answer": false` shows how it was judged

### A timer request went to the agent instead
Read the decision log line: the `timer` field shows the classifier's choice (`action`), its score (`p`) and any `error` (for example "no duration"). Lower the timer threshold a little if real timer commands score just below it, or say the duration in plain numbers.

### The assistant says it isn't available
The main agent and the backup (if any) both failed. Check **Settings → Repairs**, and the `agent_error` and `backup_error` fields in the decision log. A missing agent usually means its integration failed to load: for MCP Assist, check that the server it connects to is running.

### The name is missing from transcripts
Speech detection can clip the first word, so "Nova, what time is it?" arrives as "what time is it?". Use a wake word model for the name: it then supplies the name as a hint. If your speech-to-text engine misspells the name in the same way each time, add the misspelling as an extra name.

### It answers the TV
Read the decision log for those lines. If the classifier scored them above the request threshold, raise it. If they have a `wake_hint`, raise the wake word threshold.

### Button requests are being judged
They should not be: button requests pass straight through. If they are judged, the device's run flag is stuck. Reboot the device, and please open an issue with the decision log lines.

### Turning it off
Switch **Smart Listening** off on a device for the stock wake word behaviour. To remove it entirely, remove the package from the device YAML, reinstall the firmware, and set the device's assistant back to your usual pipeline.

## Development

Tests use Home Assistant's own helpers:

```bash
python -m venv .venv && . .venv/bin/activate
pip install homeassistant hassil gazetteer-matcher pytest pytest-asyncio
pytest tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mike-nott/smart-listening/issues)
- **Home Assistant Community**: [Community Forum](https://community.home-assistant.io/)
