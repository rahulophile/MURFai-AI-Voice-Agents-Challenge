import logging
import os
import json
import datetime
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

WELLNESS_LOG_PATH = "wellness_log.json"


def _load_wellness_history(max_entries: int = 3) -> str:
    """Load recent wellness check-ins and return a short textual summary
    that the model can use for gentle references.
    """
    if not os.path.exists(WELLNESS_LOG_PATH):
        return "No previous check-ins are available yet."

    try:
        with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to read wellness log: %s", e)
        return "Previous check-ins exist, but they could not be read."

    if not isinstance(data, list) or len(data) == 0:
        return "No previous check-ins are available yet."

    # Take last N entries
    recent = data[-max_entries:]
    lines = []
    for entry in recent:
        ts = entry.get("timestamp", "unknown time")
        mood = entry.get("mood", "unknown mood")
        energy = entry.get("energy", "unknown energy")
        objectives = entry.get("objectives", [])
        summary = entry.get("summary", "")

        if isinstance(objectives, list):
            obj_text = "; ".join(objectives)
        else:
            obj_text = str(objectives)

        line = f"- {ts}: mood '{mood}', energy '{energy}', objectives: {obj_text}"
        if summary:
            line += f". Summary: {summary}"
        lines.append(line)

    return "Recent check-ins:\n" + "\n".join(lines)


class Assistant(Agent):
    def __init__(self) -> None:
        history_summary = _load_wellness_history()

        super().__init__(
            instructions=f"""
You are a calm, supportive, and realistic health & wellness voice companion.

Your role:
- Have a short daily check-in with the user.
- Ask about their mood, energy, and any current stressors.
- Ask about 1–3 simple, realistic objectives or intentions for the day.
- Offer small, actionable, non-medical suggestions based on what they share.
- Avoid any diagnosis, medical claims, or clinical language. You are not a doctor or therapist.

Conversation structure:
1. Gently ask how they are feeling today and what their energy is like.
2. Ask if anything is stressing them out or on their mind.
3. Ask for 1–3 things they would like to get done today (work, study, personal).
4. Ask if there is anything they want to do for themselves (rest, hobbies, movement, breaks).
5. Reflect back briefly with supportive language.
6. Close with a short recap of:
   - Their mood and energy.
   - Their main 1–3 objectives for the day.
   - One small, realistic suggestion if appropriate.

After you have:
- A clear sense of today's mood,
- A simple description of energy,
- Any key stressors (if shared),
- A list of objectives/intentions (1–3),
- A one-sentence recap/summary,

then call the `save_wellness_checkin` tool exactly once with:
- mood: a short mood description in the user's own terms if possible
- energy: a short description like "low", "okay", "high", etc.
- stressors: a short phrase or sentence
- objectives: a list of 1–3 short strings
- summary: one sentence summarizing today's check-in

Past data:
Here is some context from previous check-ins (if any). Use it gently:
{history_summary}

How to use this context:
- Occasionally refer back in a soft way, e.g. "Last time you mentioned low energy, how does today compare?"
- Do not read it verbatim.
- Do not overwhelm the user with history.
- Keep the tone non-judgmental and supportive.

Important:
- Do NOT give medical advice.
- Do NOT mention tools, JSON, or files.
- Keep responses short, clear, and conversational.
""",
        )

    @function_tool
    async def save_wellness_checkin(
        self,
        context: RunContext,
        mood: str,
        energy: str,
        stressors: Optional[str],
        objectives: List[str],
        summary: str,
    ) -> str:
        """
        Save a single wellness check-in entry to a JSON file.

        This should be called once per check-in, after the recap.
        """

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "mood": mood,
            "energy": energy,
            "stressors": stressors or "",
            "objectives": objectives,
            "summary": summary,
        }

        # Load existing log (if any)
        log: List[dict] = []
        if os.path.exists(WELLNESS_LOG_PATH):
            try:
                with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        log = data
            except Exception as e:
                logger.warning("Failed to read existing wellness log: %s", e)

        log.append(entry)

        # Save updated log
        try:
            with open(WELLNESS_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
            logger.info("Saved wellness check-in: %s", entry)
        except Exception as e:
            logger.error("Failed to save wellness check-in: %s", e)
            return "There was an error saving the wellness check-in."

        # Return a compact string for the model to use internally
        obj_text = ", ".join(objectives) if objectives else "no specific objectives"
        return (
            f"Saved wellness check-in with mood '{mood}', energy '{energy}', "
            f"stressors '{stressors or 'none'}', objectives {obj_text}."
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
