Here’s a prompt you can paste into Cursor so it generates the demo program.
(I’ll write it as if you are talking directly to Cursor.)

---

**PROMPT FOR CURSOR**

You are an expert Python developer.
Build me a **demo multi-agent “argument simulator”** using the OpenAI-compatible Qwen API on DashScope.

I want a **single Python program** (one file is OK) that does the following:

---

### 1. Tech & API

* **Language**: Python 3
* **Run mode**: console / terminal app (no GUI)
* **LLM API**: use the OpenAI-compatible client on DashScope:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Example:
completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
)
print(completion.model_dump_json())
```

* Reuse this style for all LLM calls.
* Use type hints and clear function boundaries.
* Add comments for all public functions and main classes.

---

### 2. High-level behaviour

We simulate:

* 3 **speaker characters** (fixed, e.g. Alice, Bob, Carol)
* 1 **“God” controller** (not speaking in the chat, only planning)
* 1 **User** (me) who can type messages in **another thread** while the simulation is running

Flow:

1. User starts the program.

2. Program asks:

   * for a **topic** string (e.g. “Should AI replace human teachers?”)
   * for **number of rounds** (integer, number of turns the simulation should run)

3. After that, the 3 speakers start to talk to each other around this topic, for the requested number of rounds.

4. On each round:

   * The **God** LLM call:

     * Reads the conversation so far.
     * Reads each character’s personality + background.
     * Reads any new **user input** (if user typed something since last round).
     * Decides:

       * Each character’s **current emotion**.
       * Each character’s **“willingness to speak”** score.
       * Which speaker(s) will talk in this round.
   * Then, for each chosen speaker, we make a **speaker LLM call** to generate that character’s line, using their persona, background, current emotion, and the conversation history.
   * The line is printed like:
     `[Alice][angry]: I think you're completely wrong about that.`
   * The message is appended to the conversation history.

5. At any time while the simulation is running:

   * In another thread, the user can type their own lines (e.g. “I think Bob is stupid”).
   * These messages are inserted into the conversation as `speaker="User"`.
   * The next time the God LLM is called, it will see those lines and may cause characters to react.

The key feeling: **they do not always answer each other directly**, and **who speaks next depends on disagreement and personality**, not pure randomness.

---

### 3. Program structure

Create clear, well-structured code with the following components (classes/functions).
You can keep everything in one file, but organise it logically.

#### 3.1 Data models

Use `dataclasses` where appropriate.

* `CharacterConfig`

  * `name: str`
  * `persona: str`  (free text describing personality & speaking style)
  * `background_topics: list[str]`  (topics this character is knowledgeable about)
  * `talkativeness: float`  (0–1)
  * `aggressiveness: float`  (0–1)

* `EmotionState`

  * `character_name: str`
  * `emotion: str`  (e.g. "calm", "angry", "mocking", "excited")
  * `intensity: float` (0–1)

* `Message`

  * `speaker: str`  (e.g. "Alice", "Bob", "Carol", "User")
  * `text: str`
  * `emotion: str | None`
  * `topic_relevance: float` (0–1; how related to the main topic)
  * `timestamp: float` (or just an incrementing index)

* `ConversationState`

  * `topic: str`
  * `messages: list[Message]`
  * `emotions: dict[str, EmotionState]` (current emotional state per character)
  * helper methods to append messages and fetch recent context.

---

#### 3.2 LLM client wrapper

Create a small wrapper around `client.chat.completions.create`:

* `class LLMClient:`

  * `def __init__(self, client: OpenAI, model: str = "qwen-plus")`
  * `def chat(self, messages: list[dict]) -> str:`

    * Takes OpenAI chat-format messages.
    * Returns the assistant text content.
  * Optionally, have specialized methods:

    * `call_god(...)`
    * `call_speaker(...)`

Use **robust error handling**:

* Retry a few times on HTTP/connection errors.
* Print/log simple error messages.

---

#### 3.3 God engine

`class GodEngine:`

* Holds:

  * reference to `LLMClient`
  * list of `CharacterConfig`s
* Main method:

```python
def decide_next_speakers_and_emotions(
    self,
    conversation: ConversationState,
    pending_user_messages: list[Message]
) -> tuple[list[str], dict[str, EmotionState]]:
    ...
```

Responsibilities:

1. Compose a **system prompt** describing:

   * The role of the God:

     * It never speaks as a character.
     * It only returns JSON with:

       * `emotions`: mapping from character name → emotion + intensity
       * `speakers`: list of character names who will speak this round
   * The personalities and backgrounds of all characters.
   * The rule:

     * If the last messages strongly **disagree** with someone’s core background/stance, that character has a **higher probability to speak** (they like to argue).
     * If the conversation is about things not in their background, they are **less likely** to speak.
     * A character who hasn’t spoken for a while becomes more willing to speak.
     * User messages should be treated as coming from a distinct person; characters may support or attack the user.
2. Compose a **user message** to the LLM that includes:

   * The **topic**.
   * A **short summary** of the last N messages (or just the raw text if easier).
   * A list of new user messages since the last round.
   * A reminder of each character’s background and personality.
3. Ask the LLM to respond with **strict JSON**, e.g.:

```json
{
  "emotions": {
    "Alice": {"emotion": "concerned", "intensity": 0.7},
    "Bob":   {"emotion": "angry", "intensity": 0.9},
    "Carol": {"emotion": "amused", "intensity": 0.6}
  },
  "speakers": ["Bob", "Carol"]
}
```

4. Parse this JSON into:

   * `EmotionState` dict
   * list of speaker names for this round

This “God” layer is where you model the **willingness to talk**.

---

#### 3.4 Speaker engine

`class SpeakerEngine:`

* Holds:

  * reference to `LLMClient`
  * `CharacterConfig`
* Method:

```python
def generate_utterance(
    self,
    conversation: ConversationState,
    current_emotion: EmotionState
) -> str:
    ...
```

Prompting rules:

* `system` message:

  * Explains this is, for example, Alice.
  * Includes persona, background topics.
  * Includes current emotion and intensity.
  * Explains:

    * “You may respond to the last speaker, or push your own viewpoint.”
    * “You speak in 1–2 sentences, no narration, only your line.”
    * “If the recent messages are mostly unrelated to your background and interests, you may stay more vague or change the topic back to what you care about.”
* `user` message:

  * Contains:

    * Topic
    * Truncated conversation history
    * Current emotion state
    * Possibly a short instruction like:
      “Now it is Alice’s turn. Speak one line.”

Return the text line for that speaker.

---

#### 3.5 User input thread

Implement a simple **thread-safe queue** or list to collect user messages:

* `class UserInputManager:`

  * Uses `threading.Thread`.
  * In the thread’s loop:

    * `input("> You: ")`
    * Wrap into a `Message(speaker="User", text=..., ...)`.
    * Push into a `queue.Queue` or internal list.

The main simulation loop will:

* At each round, pull **all** pending user messages from this queue.
* Append them into `ConversationState`.
* Pass them to `GodEngine.decide_next_speakers_and_emotions`.

---

#### 3.6 Main simulation loop

Create a `run_simulation()` function and a `if __name__ == "__main__":` block.

Steps:

1. Read topic from input.

2. Read number of rounds.

3. Create:

   * `OpenAI` client
   * `LLMClient`
   * `CharacterConfig` list for 3 characters (give them distinct personas & backgrounds)
   * `GodEngine`
   * `SpeakerEngine` instances (or a dictionary name → engine)
   * `ConversationState`
   * `UserInputManager` (start its thread)

4. Print an initial description of the 3 characters and the topic.

5. Loop for `round_index` in `range(num_rounds)`:

   * Collect pending user messages from `UserInputManager`, append to conversation.
   * Call `GodEngine.decide_next_speakers_and_emotions(...)`.
   * Update `conversation.emotions` with the returned emotions.
   * For each speaker in the returned list:

     * Use the corresponding `SpeakerEngine` to generate a line.
     * Create `Message` with speaker name, text, emotion, etc.
     * Append to conversation.
     * Print to console, including emotion label:
       `[Carol][amused]: ...`
   * Optionally sleep a small random delay (e.g. 0.3–1.0s) between speakers for realism.

6. After the loop, stop the user input thread gracefully (e.g. with a flag) and exit.

---

### 4. Character design

Define three sample characters in code (you can adjust text yourself):

* **Alice**:

  * Persona: idealistic, cooperative, cares about fairness.
  * Background topics: `"education"`, `"social equality"`, `"technology and ethics"`.
* **Bob**:

  * Persona: cynical, aggressive, likes to attack weak arguments.
  * Background topics: `"economy"`, `"power struggles"`, `"history of conflicts"`.
* **Carol**:

  * Persona: sarcastic observer, enjoys making ironic comments.
  * Background topics: `"psychology"`, `"media"`, `"online culture"`.

The **God** should favour:

* If the current subtopic matches a character’s background topics, they are more likely to talk.
* If a message strongly opposes their usual outlook, they are more likely to argue.
* If the topic drifts far away from their background, their willingness may drop, or they only make side comments.

---

### 5. Non-functional requirements

* Code should run as-is (assuming `openai` and environment variables are set).
* Handle API errors gracefully; don’t crash the whole app on one failed call.
* Clear console output formatting: easy to see who is speaking and what their emotion is.
* Put all configuration (like number of messages to keep in context, max tokens, etc.) in small constants at the top for easy tuning.

Generate the full Python source code according to this specification.
