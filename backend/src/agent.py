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

FAQ_PATH = "shared-data/day5_company_faq.json"
LEADS_DIR = "leads"


def _load_company_faq() -> dict:
    """Load company info + FAQ from JSON file."""
    if not os.path.exists(FAQ_PATH):
        logger.warning("FAQ file not found at %s", FAQ_PATH)
        return {}

    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load FAQ content: %s", e)
        return {}

    if not isinstance(data, dict):
        logger.error("FAQ content must be a JSON object (dict) with company, description, faqs.")
        return {}

    # Normalize FAQ list
    faqs = data.get("faqs", [])
    if not isinstance(faqs, list):
        data["faqs"] = []
    return data


class Assistant(Agent):
    def __init__(self) -> None:
        self.company_data = _load_company_faq()
        company_name = self.company_data.get("company", "the company")
        description = self.company_data.get("description", "")
        faqs = self.company_data.get("faqs", [])

        # Short FAQ overview for system prompt
        if faqs:
            faq_overview_lines: List[str] = []
            for f in faqs:
                q = f.get("q", "")
                a = f.get("a", "")
                if len(a) > 180:
                    a_preview = a[:177] + "..."
                else:
                    a_preview = a
                faq_overview_lines.append(f"- Q: {q}\n  A: {a_preview}")
            faq_overview = "\n".join(faq_overview_lines)
        else:
            faq_overview = "No FAQ entries loaded."

        super().__init__(
            instructions=f"""
You are a voice-based Sales Development Representative (SDR) for the company "{company_name}".

Company description (for your context):
{description}

Your job:
- Greet visitors warmly and professionally.
- Ask what brought them here and what they are working on.
- Keep the conversation focused on understanding their needs and whether {company_name} is a good fit.
- Answer questions about the product, who it is for, and pricing using the provided FAQ content.
- Politely collect key lead details and summarize them at the end of the call.

Very important:
- When the user asks about the product, pricing, or who it is for, you MUST call the `lookup_faq` tool with their question.
- Use ONLY information from the FAQ answer returned by the tool. Do NOT invent or guess extra details.
- If the FAQ tool indicates that information is not available, be honest and say that you don't have that detail.

FAQ overview (for your reference only):
{faq_overview}

Lead collection:
Over the course of the conversation, you should naturally ask for:
- Name
- Company
- Email
- Role
- Use case (what they want to use this for)
- Team size
- Timeline (now / soon / later)

Do this gradually, not as an interrogation. Fit questions into the flow of conversation.

End-of-call behavior:
- When the user signals the conversation is ending (e.g., says "that's all", "I'm done", "thanks"), you should:
  1. Call the `save_lead` tool ONCE with the best values you have for all lead fields. Use "unknown" for anything that was not provided.
  2. After the tool returns, give a short spoken summary of:
     - Who they are (name, role, company),
     - What they are interested in (use case),
     - Their approximate timeline (now / soon / later).
  3. Thank them for their time.

Tone:
- Warm, concise, and clearly focused on being helpful.
- Avoid jargon unless the user is clearly technical.
- Do NOT mention tools, JSON files, or internal implementation details to the user.
- Keep your responses short and suitable for speech.
""",
        )

    @function_tool
    async def lookup_faq(self, context: RunContext, question: str) -> str:
        """
        Look up an answer in the company FAQ for a user question.

        The model should call this whenever the user asks about product/company/pricing.
        """

        data = self.company_data or {}
        faqs = data.get("faqs", [])
        if not faqs:
            return (
                "No FAQ data is available. You should tell the user that you don't have "
                "the detailed information right now, but can share a high-level description."
            )

        q_lower = (question or "").lower()
        best_match = None
        best_score = 0

        # Very simple keyword-based matching: count overlapping words
        q_words = [w for w in q_lower.split() if len(w) > 2]

        for entry in faqs:
            fq = str(entry.get("q", "")).lower()
            fa = str(entry.get("a", ""))
            text = fq + " " + fa.lower()
            score = 0
            for w in q_words:
                if w in text:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = entry

        if not best_match or best_score == 0:
            return (
                "I could not find a specific FAQ answer for this question. "
                "You should respond honestly that you don't have that exact detail, "
                "and if possible, give a short high-level answer based on the company description."
            )

        answer = best_match.get("a", "")
        faq_q = best_match.get("q", "")
        return (
            f"FAQ match found. The closest question was: '{faq_q}'. "
            f"The answer is: {answer}"
        )

    @function_tool
    async def save_lead(
        self,
        context: RunContext,
        name: str,
        company: str,
        email: str,
        role: str,
        use_case: str,
        team_size: str,
        timeline: str,
    ) -> str:
        """
        Save the collected lead information to a JSON file.

        The model should call this ONCE at the end of the call, after the user
        indicates they are done. If any field is unknown, pass 'unknown'.
        """

        lead = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "name": name or "unknown",
            "company": company or "unknown",
            "email": email or "unknown",
            "role": role or "unknown",
            "use_case": use_case or "unknown",
            "team_size": team_size or "unknown",
            "timeline": timeline or "unknown",
        }

        os.makedirs(LEADS_DIR, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"lead-{ts}.json"
        path = os.path.join(LEADS_DIR, filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(lead, f, indent=2)
            logger.info("Saved lead to %s: %s", path, lead)
        except Exception as e:
            logger.error("Failed to save lead: %s", e)
            return "There was an error saving the lead information."

        # Return a compact summary string for the model
        return (
            f"Lead saved as {filename}. "
            f"Name: {lead['name']}, Company: {lead['company']}, "
            f"Email: {lead['email']}, Role: {lead['role']}, "
            f"Use case: {lead['use_case']}, Team size: {lead['team_size']}, "
            f"Timeline: {lead['timeline']}."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        # STT: user's speech -> text
        stt=deepgram.STT(model="nova-3"),
        # LLM: brain of the SDR
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # TTS: voice of the SDR (Murf Falcon - Matthew)
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # Turn detection & VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Allow partial preemptive generation
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

    # Start SDR session
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
