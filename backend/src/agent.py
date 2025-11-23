import logging
import os
import json
import datetime
from typing import List

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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly coffee shop barista for a specialty coffee brand called "Pallavi's Roastery".

Your only job is to help the customer place a coffee order by asking clear, simple questions.
You must maintain an internal coffee order with the following fields:

- drinkType: type of drink (e.g., latte, cappuccino, americano, cold brew)
- size: small, medium, or large
- milk: e.g., regular, skim, soy, almond, oat
- extras: a list of extras (e.g., extra shot, caramel syrup, vanilla, whipped cream)
- name: the customer's name

Behavior rules:
- Start by greeting the user as a barista and asking what they would like.
- If some fields are missing or unclear, ask polite follow-up questions.
- Ask only one or two things at a time so it's easy to answer.
- Keep the conversation short, natural, and to the point.
- Once you are confident that all fields (drinkType, size, milk, extras, name) are filled,
  call the `finalize_order` tool exactly once with the values you collected.
- After the tool returns, clearly summarize the final order to the user and thank them.

Never talk about tools, JSON, files, or internal state directly to the user.
Stay in-character as a barista at all times.
""",
        )

    @function_tool
    async def finalize_order(
        self,
        context: RunContext,
        drinkType: str,
        size: str,
        milk: str,
        extras: List[str],
        name: str,
    ) -> str:
        """
        Finalize the coffee order and save it to a JSON file.

        Use this tool only when the order is complete and all fields are known.
        """

        order = {
            "drinkType": drinkType,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name,
        }

        # Ensure the orders directory exists (relative to backend working directory)
        os.makedirs("orders", exist_ok=True)

        # Create a timestamped filename
        ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"order-{ts}.json"
        path = os.path.join("orders", filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2)

        logger.info("Saved order to %s: %s", path, order)

        extras_text = ", ".join(extras) if extras else "no extras"

        # This string is for the model to read and then confirm back to the user
        return (
            f"Order saved: {size} {drinkType} with {milk} milk, "
            f"{extras_text}, for {name}. File name: {filename}."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Deepgram, Gemini, Murf, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
