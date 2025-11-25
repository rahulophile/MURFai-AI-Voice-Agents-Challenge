import logging
import os
import json
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

CONTENT_PATH = "shared-data/day4_tutor_content.json"


def _load_tutor_content() -> List[dict]:
    """Load tutor concepts from the JSON content file."""
    if not os.path.exists(CONTENT_PATH):
        logger.warning("Tutor content file not found at %s", CONTENT_PATH)
        return []

    try:
        with open(CONTENT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load tutor content: %s", e)
        return []

    if not isinstance(data, list):
        logger.error("Tutor content must be a list of concept objects.")
        return []

    return data


class Assistant(Agent):
    def __init__(self) -> None:
        self.content: List[dict] = _load_tutor_content()

        # Build a short overview of concepts for the system prompt
        if self.content:
            concepts_overview = "\n".join(
                f"- {c.get('id', '')}: {c.get('title', '')}"
                for c in self.content
            )
        else:
            concepts_overview = "No concepts loaded. Make sure the JSON file exists."

        super().__init__(
            instructions=f"""
You are an Active Recall Coach in a "Teach-the-Tutor" experience.

Your job:
- Help the user learn programming concepts using three modes: learn, quiz, and teach_back.
- Encourage the user to explain concepts back to you.
- Keep your tone simple, encouraging, and clear.
- You are not a general chatbot. Stay focused on helping the user learn the concepts in the course content.

Available learning modes:
- learn: you explain the concept in simple language.
- quiz: you ask the user questions about the concept.
- teach_back: you ask the user to explain the concept back, then you give short qualitative feedback.

Voices (conceptual mapping):
- learn mode uses Murf Falcon voice "Matthew".
- quiz mode uses Murf Falcon voice "Alicia".
- teach_back mode uses Murf Falcon voice "Ken".

You do not control the audio directly, but you should behave as if these voices are used for the three modes.

Course content:
You have a small set of concepts loaded from a JSON file.
Each concept has:
- id
- title
- summary
- sample_question

Concepts available:
{concepts_overview}

Core behavior:
1. When the conversation starts, greet the user briefly and ask:
   - which mode they want to use first (learn, quiz, or teach_back),
   - and which concept they want (e.g. by id or title).
2. When the user chooses a mode and concept, call the `set_tutor_mode` tool to select them.
3. After `set_tutor_mode` returns, follow these rules:

   - learn mode:
     - Use the concept summary to explain the idea in simple terms.
     - Give 1–2 short examples if helpful.
     - Keep it concise.

   - quiz mode:
     - Use the sample_question as a starting point.
     - Ask the question and wait for the user's answer.
     - Then, give brief, encouraging feedback based on their answer.
     - You can ask 1–2 follow-up questions if time allows.

   - teach_back mode:
     - Ask the user to explain the concept in their own words.
     - After they respond, highlight what they did well and one thing they could add or clarify.
     - Keep feedback short and friendly.

4. The user can switch modes at any time by saying things like:
   - "Switch to quiz mode."
   - "Can we do teach back for loops?"
   When that happens, call `set_tutor_mode` again with the new mode (and concept if they specify it).

Important:
- Stay focused on the given concepts. If the user asks about something completely unrelated, gently bring them back.
- Keep your answers short and spoken-friendly. Avoid long paragraphs.
- Do not mention tools, JSON files, or internal state.
""",
        )

    def _find_concept(self, concept_id_or_title: Optional[str]) -> Optional[dict]:
        """Find a concept by id or title (case-insensitive)."""
        if not self.content:
            return None

        if not concept_id_or_title:
            # Default to the first concept
            return self.content[0]

        key = concept_id_or_title.strip().lower()

        # Try id match
        for c in self.content:
            if str(c.get("id", "")).lower() == key:
                return c

        # Try title match
        for c in self.content:
            if str(c.get("title", "")).lower() == key:
                return c

        # Fallback: first concept
        return self.content[0]

    @function_tool
    async def set_tutor_mode(
        self,
        context: RunContext,
        mode: str,
        concept: Optional[str] = None,
    ) -> str:
        """
        Select the current learning mode and concept for the tutor.

        Args:
            mode: One of "learn", "quiz", or "teach_back".
            concept: A concept id or title (e.g., "variables" or "loops").
        """

        mode_normalized = (mode or "").strip().lower()
        if mode_normalized not in ("learn", "quiz", "teach_back"):
            return (
                "Invalid mode. Please choose one of: learn, quiz, or teach_back. "
                "Keep the user's request in mind and ask them again."
            )

        selected_concept = self._find_concept(concept)
        if not selected_concept:
            return (
                "No tutor content is available. Ask the user to check back later or "
                "to ensure the content file is configured."
            )

        # Prepare a compact description for the LLM to use
        concept_id = selected_concept.get("id", "")
        title = selected_concept.get("title", "")
        summary = selected_concept.get("summary", "")
        sample_q = selected_concept.get("sample_question", "")

        # Note: we are not mutating any shared session state here.
        # The LLM should use this returned text to continue the conversation.
        return (
            f"Tutor mode set to '{mode_normalized}' for concept '{concept_id}' "
            f"({title}). "
            f"Summary: {summary} "
            f"Sample question: {sample_q}. "
            "In learn mode, explain using the summary. "
            "In quiz mode, use the sample question (and simple follow-ups). "
            "In teach_back mode, ask the user to explain the concept in their own words, "
            "then give friendly, brief feedback."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
