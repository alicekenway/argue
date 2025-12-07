"""Demo multi-agent argument simulator for DashScope's Qwen API."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple

from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout

# OpenAI-compatible endpoint plus prompt/config tuning knobs.
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
# Directory that stores one JSON file per language (e.g., english, chinese, ...).
CHARACTER_CONFIG_DIR = Path(__file__).with_name("character_profiles")
DEFAULT_SPEAKER_GUIDANCE = (
    "Stay in first-person dialogue only and avoid stage directions. "
    "Reply with one or two sentences at most."
)
MAX_CONTEXT_MESSAGES = 12
SPEAKER_CONTEXT_MESSAGES = 8
LLM_RETRY_DELAY = 1.5
GOD_MAX_RETRIES = 4
SPEAKER_MAX_RETRIES = 3
SPEAKER_MAX_TOKENS = 300
SPEAKER_SLEEP_MIN = 0.3
SPEAKER_SLEEP_MAX = 1.0


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a float between provided bounds."""
    return max(minimum, min(maximum, value))


def estimate_topic_relevance(text: str, topic: str) -> float:
    """Tiny heuristic for topic relevance based on overlapping keywords."""
    topic_keywords = [word for word in topic.lower().split() if len(word) > 3]
    if not topic_keywords:
        return 0.5
    text_lower = text.lower()
    overlap = sum(1 for word in topic_keywords if word in text_lower)
    return clamp(overlap / len(topic_keywords) if topic_keywords else 0.5, 0.1, 1.0)


def extract_json_object(raw_text: str) -> Dict:
    """Attempt to locate JSON within an LLM response."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start : end + 1])
        raise


TALK_LEVEL_LABELS = {
    1: "almost silent",
    2: "reserved",
    3: "even-tempered",
    4: "talkative",
    5: "dominates conversations",
}

AGGRESSION_LEVEL_LABELS = {
    1: "highly diplomatic",
    2: "cautiously assertive",
    3: "direct but fair",
    4: "confrontational",
    5: "relentless attacker",
}


def level_to_ratio(level: int) -> float:
    """Convert a 1-5 scale into 0.0-1.0 ratio."""
    return clamp((level - 1) / 4) if level else 0.5


def level_label(level: int, labels: Dict[int, str], fallback: str) -> str:
    """Map numeric level to textual descriptor."""
    return labels.get(level, fallback)


@dataclass
class CharacterConfig:
    """Configuration bundle describing a debate character."""

    name: str
    persona: str
    background_topics: List[str]
    talkativeness: float
    aggressiveness: float
    speaking_style: str = ""
    goals: List[str] = field(default_factory=list)
    talk_level: int = 3
    talk_summary: str = "even-tempered"
    aggressiveness_level: int = 3
    aggressiveness_summary: str = "direct but fair"


@dataclass
class GodPromptConfig:
    """Holds configurable prompt scaffolding for the God orchestrator."""

    overview: str
    rules: List[str]
    response_format: str


@dataclass
class EnvironmentConfig:
    """Global environment controls applied to every speaker."""

    speaker_guidance: str = DEFAULT_SPEAKER_GUIDANCE


@dataclass
class LanguageProfileDescriptor:
    """Represents a single language pack on disk."""

    code: str
    label: str
    path: Path


@dataclass
class EmotionState:
    """Tracks the active emotion for a character."""

    character_name: str
    emotion: str
    intensity: float


@dataclass
class Message:
    """Represents a single utterance within the conversation."""

    speaker: str
    text: str
    emotion: Optional[str] = None
    topic_relevance: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    """Holds the evolving transcript and convenience helpers."""

    topic: str
    messages: List[Message] = field(default_factory=list)
    emotions: Dict[str, EmotionState] = field(default_factory=dict)

    def append_message(self, message: Message) -> None:
        """Add a message while normalising metadata."""
        message.topic_relevance = clamp(message.topic_relevance)
        self.messages.append(message)

    def recent_messages(self, limit: int) -> List[Message]:
        """Return the latest N messages for context windows."""
        return self.messages[-limit:]

    def format_recent_messages(self, limit: int) -> str:
        """Create a textual snapshot suitable for prompting."""
        if not self.messages:
            return "No conversation yet."
        snippets = []
        for msg in self.recent_messages(limit):
            emo = f" ({msg.emotion})" if msg.emotion else ""
            snippets.append(f"{msg.speaker}{emo}: {msg.text}")
        return "\n".join(snippets)

    def turns_since_character_spoke(self, name: str) -> int:
        """Count turns since the given character last spoke."""
        for idx, msg in enumerate(reversed(self.messages), 1):
            if msg.speaker == name:
                return idx - 1
        return len(self.messages)


def load_character_profiles(
    config_path: Path,
) -> Tuple[GodPromptConfig, EnvironmentConfig, List[CharacterConfig]]:
    """Load God + character prompt definitions from a JSON file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Character config not found: {config_path}")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse character config: {exc}") from exc

    god_raw = payload.get("god")
    if not god_raw:
        raise ValueError("Character config is missing the 'god' section.")

    characters_raw = payload.get("characters", [])
    if not characters_raw:
        raise ValueError("Character config requires at least one character entry.")

    god_prompt = GodPromptConfig(
        overview=god_raw.get("overview", "").strip(),
        rules=[rule.strip() for rule in god_raw.get("rules", []) if rule.strip()],
        response_format=god_raw.get("response_format", "").strip(),
    )

    environment_raw = payload.get("environment", {})
    speaker_guidance = environment_raw.get("speaker_guidance", DEFAULT_SPEAKER_GUIDANCE)
    environment = EnvironmentConfig(
        speaker_guidance=speaker_guidance.strip() or DEFAULT_SPEAKER_GUIDANCE
    )

    characters: List[CharacterConfig] = []
    for entry in characters_raw:
        try:
            raw_talk = entry.get("talkativeness_level", entry.get("talkativeness", 3))
            raw_aggr = entry.get("aggressiveness_level", entry.get("aggressiveness", 3))

            if isinstance(raw_talk, str) and raw_talk.isdigit():
                raw_talk = int(raw_talk)
            if isinstance(raw_aggr, str) and raw_aggr.isdigit():
                raw_aggr = int(raw_aggr)

            talk_level = int(raw_talk) if isinstance(raw_talk, int) else max(
                1, min(5, int(round(float(raw_talk) * 4 + 1)))
            )
            aggr_level = int(raw_aggr) if isinstance(raw_aggr, int) else max(
                1, min(5, int(round(float(raw_aggr) * 4 + 1)))
            )

            characters.append(
                CharacterConfig(
                    name=entry["name"],
                    persona=entry["persona"],
                    background_topics=entry.get("background_topics", []),
                    talkativeness=level_to_ratio(talk_level),
                    aggressiveness=level_to_ratio(aggr_level),
                    speaking_style=entry.get("speaking_style", ""),
                    goals=entry.get("goals", []),
                    talk_level=talk_level,
                    talk_summary=level_label(
                        talk_level, TALK_LEVEL_LABELS, "balanced participation"
                    ),
                    aggressiveness_level=aggr_level,
                    aggressiveness_summary=level_label(
                        aggr_level, AGGRESSION_LEVEL_LABELS, "balanced assertiveness"
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Character entry missing field: {exc}") from exc

    return god_prompt, environment, characters


def discover_language_profiles(
    config_dir: Path,
) -> Dict[str, LanguageProfileDescriptor]:
    """Scan the config directory and build a language -> descriptor map."""

    if not config_dir.exists() or not config_dir.is_dir():
        raise FileNotFoundError(
            f"Language profile directory not found: {config_dir}"
        )

    descriptors: Dict[str, LanguageProfileDescriptor] = {}
    for json_file in sorted(config_dir.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[Profiles] Skipping {json_file.name}: invalid JSON ({exc}).")
            continue

        language_code = (
            str(raw.get("language_code") or raw.get("language") or json_file.stem)
            .strip()
            .lower()
        )
        language_label = str(raw.get("language_label") or language_code).strip()

        if not language_code:
            print(
                f"[Profiles] Skipping {json_file.name}: missing 'language_code'."
            )
            continue

        descriptors[language_code] = LanguageProfileDescriptor(
            code=language_code,
            label=language_label,
            path=json_file,
        )

    if not descriptors:
        raise ValueError(
            f"No usable language profile files found under {config_dir}."
        )

    return descriptors


def prompt_for_language_choice(
    descriptors: Dict[str, LanguageProfileDescriptor],
) -> LanguageProfileDescriptor:
    """Prompt the user to pick a language, defaulting to English if available."""

    ordered = sorted(descriptors.values(), key=lambda item: item.code)
    default_code = "en" if "en" in descriptors else ordered[0].code

    options_display = ", ".join(
        f"{descriptor.code} ({descriptor.label})" for descriptor in ordered
    )
    prompt = (
        f"Select language [{default_code}] "
        f"(available: {options_display}): "
    )

    while True:
        choice = input(prompt).strip().lower()
        if not choice:
            return descriptors[default_code]
        if choice in descriptors:
            return descriptors[choice]
        print(
            f"Unknown language '{choice}'. Please choose one of: {options_display}."
        )


class LLMClient:
    """Resilient wrapper around the OpenAI-compatible chat endpoint."""

    def __init__(self, client: OpenAI, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ) -> str:
        """Call the chat endpoint with retries on transient failures."""
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except (APIConnectionError, Timeout) as exc:
                last_error = exc
                print(f"[LLMClient] Connection error (attempt {attempt}): {exc}")
            except RateLimitError as exc:
                last_error = exc
                print(f"[LLMClient] Rate limit (attempt {attempt}): {exc}")
            except APIError as exc:
                last_error = exc
                print(f"[LLMClient] API error (attempt {attempt}): {exc}")
                if not getattr(exc, "should_retry", False):
                    break
            time.sleep(LLM_RETRY_DELAY * attempt)
        raise RuntimeError("LLM chat call failed") from last_error

    def call_god(self, system_prompt: str, user_prompt: str) -> str:
        """Helper for God engine with tighter temperature."""
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=600,
            max_retries=GOD_MAX_RETRIES,
        )

    def call_speaker(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = SPEAKER_MAX_TOKENS,
    ) -> str:
        """Helper for speaker generation with more diversity."""
        return self.chat(
            messages,
            temperature=0.8,
            max_tokens=max_tokens,
            max_retries=SPEAKER_MAX_RETRIES,
        )


class GodEngine:
    """LLM-powered orchestrator that selects emotions and speakers each round."""

    def __init__(
        self,
        llm_client: LLMClient,
        god_prompt: GodPromptConfig,
        characters: List[CharacterConfig],
    ):
        self.llm = llm_client
        self.god_prompt = god_prompt
        self.characters = {char.name: char for char in characters}

    def decide_next_speakers_and_emotions(
        self,
        conversation: ConversationState,
        pending_user_messages: List[Message],
    ) -> Tuple[List[str], Dict[str, EmotionState]]:
        """Ask the God model which characters speak next."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(conversation, pending_user_messages)
        try:
            response_text = self.llm.call_god(system_prompt, user_prompt)
            parsed = extract_json_object(response_text)
            return self._parsed_decision(parsed, conversation)
        except Exception as exc:
            print(f"[GodEngine] Failed to parse LLM output, falling back: {exc}")
            return self._fallback_decision(conversation)

    def _build_system_prompt(self) -> str:
        """Compose a persona-rich system prompt assembled from JSON config."""

        lines: List[str] = []

        if self.god_prompt.overview:
            lines.append(self.god_prompt.overview)

        if self.god_prompt.rules:
            lines.append("Rules:")
            lines.extend(f"* {rule}" for rule in self.god_prompt.rules)

        lines.append("Character reference:")
        for char in self.characters.values():
            background = ", ".join(char.background_topics) or "general knowledge"
            goals = "; ".join(char.goals) if char.goals else "respond decisively"
            style = char.speaking_style or "no specific style"
            lines.append(
                f"- {char.name}: persona={char.persona}. style={style}. goals={goals}. "
                f"background={background}. talk level={char.talk_summary}, "
                f"aggression level={char.aggressiveness_summary}."
            )

        response_hint = (
            self.god_prompt.response_format
            or "Respond with JSON: {\"emotions\": {\"Name\": {\"emotion\": str, \"intensity\": float}},"
            ' "speakers": ["Name"]}.'
        )
        lines.append(response_hint)

        return "\n".join(lines)

    def _build_user_prompt(
        self,
        conversation: ConversationState,
        pending_user_messages: List[Message],
    ) -> str:
        """Provide topic, transcript, and new user input to the God model."""
        user_section = (
            "\n".join(f"- {msg.text}" for msg in pending_user_messages) or "none"
        )
        transcript = conversation.format_recent_messages(MAX_CONTEXT_MESSAGES)
        idle_info = ", ".join(
            f"{name}: {conversation.turns_since_character_spoke(name)} turns silence"
            for name in self.characters
        )
        response_hint = (
            self.god_prompt.response_format
            or "Return JSON describing per-character emotions and the next speakers."
        )
        return (
            f"Topic: {conversation.topic}\n"
            f"Recent conversation:\n{transcript}\n\n"
            f"New user messages since last decision:\n{user_section}\n\n"
            f"{response_hint}\n"
            "Characters may ignore both the user and their fellow debaters; encourage follow-ups when remarks are overlooked.\n"
            "Ensure at least one speaker and prefer 1-2 per round. "
            "Idle tracker: "
            f"{idle_info}. Characters increasingly want to speak after 2+ silent turns."
        )

    def _parsed_decision(
        self,
        parsed: Dict,
        conversation: ConversationState,
    ) -> Tuple[List[str], Dict[str, EmotionState]]:
        """Convert raw JSON into structured states."""
        speakers = [
            name
            for name in parsed.get("speakers", [])
            if isinstance(name, str) and name in self.characters
        ]
        emotions: Dict[str, EmotionState] = {}
        raw_emotions = parsed.get("emotions", {})
        if not speakers:
            speakers = self._fallback_speakers(conversation)
        for name, char in self.characters.items():
            info = raw_emotions.get(name, {}) if isinstance(raw_emotions, dict) else {}
            emotion_label = info.get("emotion") if isinstance(info, dict) else None
            intensity = float(info.get("intensity", 0.5)) if isinstance(info, dict) else 0.5
            if not emotion_label:
                previous = conversation.emotions.get(name)
                emotion_label = previous.emotion if previous else "neutral"
            emotions[name] = EmotionState(
                character_name=name,
                emotion=emotion_label,
                intensity=clamp(intensity),
            )
        return speakers, emotions

    def _fallback_speakers(self, conversation: ConversationState) -> List[str]:
        """Choose speakers heuristically if God output is unusable."""
        sorted_chars = sorted(
            self.characters.values(),
            key=lambda char: (
                conversation.turns_since_character_spoke(char.name),
                char.talkativeness + char.aggressiveness,
            ),
            reverse=True,
        )
        return [char.name for char in sorted_chars[:2]]

    def _fallback_decision(
        self,
        conversation: ConversationState,
    ) -> Tuple[List[str], Dict[str, EmotionState]]:
        """Fallback: neutral emotions plus heuristic speakers."""
        speakers = self._fallback_speakers(conversation)
        emotions = {
            name: conversation.emotions.get(
                name,
                EmotionState(character_name=name, emotion="neutral", intensity=0.4),
            )
            for name in self.characters
        }
        return speakers, emotions


class SpeakerEngine:
    """Generates a single character's utterance."""

    def __init__(
        self,
        llm_client: LLMClient,
        character: CharacterConfig,
        environment: EnvironmentConfig,
    ):
        self.llm = llm_client
        self.character = character
        self.environment = environment

    def generate_utterance(
        self,
        conversation: ConversationState,
        current_emotion: EmotionState,
    ) -> str:
        """Request an in-character line from the LLM."""
        goals_text = (
            "; ".join(self.character.goals) if self.character.goals else "advance your viewpoint"
        )
        background = ", ".join(self.character.background_topics) or "general knowledge"
        speaking_style = self.character.speaking_style or "direct and candid"
        system_sections = [
            self.environment.speaker_guidance.strip(),
            (
                f"You are {self.character.name}, {self.character.persona}. "
                f"Speaking style: {speaking_style}. Goals: {goals_text}. "
                f"Background knowledge: {background}. "
                f"Talkative level: {self.character.talk_summary}. "
                f"Aggressiveness level: {self.character.aggressiveness_summary}."
            ),
        ]
        system_prompt = "\n".join(section for section in system_sections if section)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in conversation.recent_messages(SPEAKER_CONTEXT_MESSAGES):
            role = "assistant" if msg.speaker == self.character.name else "user"
            speaker_label = msg.speaker
            messages.append(
                {
                    "role": role,
                    "content": f"{speaker_label}: {msg.text}",
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"It is now {self.character.name}'s turn to speak about {conversation.topic}. "
                    f"Current emotion: {current_emotion.emotion} "
                    f"(intensity {current_emotion.intensity:.2f}). "
                    "Respond directly to the ongoing discussion."
                ),
            }
        )

        return self.llm.call_speaker(messages)


class UserInputManager:
    """Background thread ingesting free-form user inputs."""

    def __init__(self):
        self.messages: Queue[Message] = Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Launch the listener thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for completion."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        """Collect console input while the simulation runs."""
        while not self._stop_event.is_set():
            try:
                text = input("> You: ").strip()
            except EOFError:
                break
            if not text:
                continue
            self.messages.put(
                Message(
                    speaker="User",
                    text=text,
                    emotion=None,
                )
            )

    def drain_messages(self, topic: str) -> List[Message]:
        """Grab all pending user messages for the main loop."""
        drained: List[Message] = []
        while True:
            try:
                message = self.messages.get_nowait()
            except Empty:
                break
            else:
                message.topic_relevance = estimate_topic_relevance(message.text, topic)
                drained.append(message)
        return drained


def print_message(message: Message) -> None:
    """Render a conversation line to the console."""
    emotion = message.emotion or "neutral"
    print(f"[{message.speaker}][{emotion}]: {message.text}")


def run_simulation() -> None:
    """Main entry point: gather inputs, run rounds, and coordinate threads."""

    try:
        language_profiles = discover_language_profiles(CHARACTER_CONFIG_DIR)
    except (OSError, ValueError) as exc:
        print(f"Unable to locate language profiles: {exc}")
        return

    selected_profile = prompt_for_language_choice(language_profiles)
    try:
        god_prompt, environment, characters = load_character_profiles(selected_profile.path)
    except (OSError, ValueError) as exc:
        print(
            "Unable to load selected language profile "
            f"({selected_profile.label} - {selected_profile.path.name}): {exc}"
        )
        return

    topic = input("Enter a debate topic: ").strip()
    if not topic:
        print("Topic is required. Exiting.")
        return
    try:
        rounds = int(input("Enter number of rounds: ").strip())
    except ValueError:
        print("Invalid number of rounds. Exiting.")
        return
    if rounds <= 0:
        print("Number of rounds must be positive. Exiting.")
        return

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Missing DASHSCOPE_API_KEY environment variable.")
        return

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    llm_client = LLMClient(client)

    god_engine = GodEngine(llm_client, god_prompt, characters)
    speaker_engines = {
        char.name: SpeakerEngine(llm_client, char, environment) for char in characters
    }
    conversation = ConversationState(topic=topic)
    for char in characters:
        conversation.emotions[char.name] = EmotionState(
            character_name=char.name,
            emotion="neutral",
            intensity=0.3,
        )

    input_manager = UserInputManager()
    input_manager.start()

    print("\n=== Argument Simulator ===")
    print(
        f"Language profile: {selected_profile.label} "
        f"({selected_profile.code}) from {selected_profile.path.name}"
    )
    print(f"Topic: {topic}")
    for char in characters:
        focus = ", ".join(char.background_topics) or "general knowledge"
        style = char.speaking_style or "freeform"
        print(
            f"- {char.name}: {char.persona} | style={style} | focus={focus} | "
            f"talk={char.talk_summary}, aggression={char.aggressiveness_summary}"
        )
    print("Type in the prompt at any time to interject as 'User'.\n")

    try:
        for round_index in range(1, rounds + 1):
            print(f"\n--- Round {round_index} ---")
            pending_user_messages = input_manager.drain_messages(topic)
            for user_msg in pending_user_messages:
                conversation.append_message(user_msg)
                print_message(user_msg)

            speakers, emotions = god_engine.decide_next_speakers_and_emotions(
                conversation, pending_user_messages
            )
            conversation.emotions.update(emotions)

            for speaker in speakers:
                engine = speaker_engines.get(speaker)
                if not engine:
                    continue
                emotion_state = conversation.emotions.get(
                    speaker,
                    EmotionState(character_name=speaker, emotion="neutral", intensity=0.4),
                )
                try:
                    text = engine.generate_utterance(conversation, emotion_state)
                except Exception as exc:
                    print(f"[SpeakerEngine] Failed to generate for {speaker}: {exc}")
                    continue
                message = Message(
                    speaker=speaker,
                    text=text,
                    emotion=f"{emotion_state.emotion} ({emotion_state.intensity:.2f})",
                    topic_relevance=estimate_topic_relevance(text, topic),
                )
                conversation.append_message(message)
                print_message(message)
                time.sleep(random.uniform(SPEAKER_SLEEP_MIN, SPEAKER_SLEEP_MAX))
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        input_manager.stop()
        print("\nConversation ended.")


if __name__ == "__main__":
    run_simulation()

