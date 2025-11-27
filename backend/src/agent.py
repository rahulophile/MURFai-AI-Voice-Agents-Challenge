import logging
import os
import json
import datetime
from typing import List, Optional, Tuple

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

FRAUD_DB_PATH = "fraud_cases.json"


def _load_fraud_db() -> List[dict]:
    """Load fraud cases from a JSON 'database' file."""
    if not os.path.exists(FRAUD_DB_PATH):
        logger.warning("Fraud DB file not found at %s", FRAUD_DB_PATH)
        return []

    try:
        with open(FRAUD_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load fraud DB: %s", e)
        return []

    if isinstance(data, dict):
        # Allow a single object DB, normalize to list
        return [data]
    if isinstance(data, list):
        return data

    logger.error("Fraud DB must be a list or object, got %s", type(data))
    return []


def _save_fraud_db(cases: List[dict]) -> None:
    """Save fraud cases list back to JSON DB."""
    try:
        with open(FRAUD_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
        logger.info("Fraud DB updated with %d case(s).", len(cases))
    except Exception as e:
        logger.error("Failed to save fraud DB: %s", e)


def _find_case_by_username(
    cases: List[dict], user_name: str
) -> Tuple[Optional[dict], Optional[int]]:
    """Find a case by userName (case-insensitive). Returns (case, index)."""
    if not user_name:
        return None, None

    key = user_name.strip().lower()
    for idx, case in enumerate(cases):
        if str(case.get("userName", "")).strip().lower() == key:
            return case, idx
    return None, None


class Assistant(Agent):
    def __init__(self) -> None:
        cases = _load_fraud_db()
        if cases:
            example_case = cases[0]
            example_name = example_case.get("userName", "SampleUser")
            example_card = example_case.get("cardEnding", "1234")
        else:
            example_name = "SampleUser"
            example_card = "1234"

        super().__init__(
            instructions=f"""
You are a calm, professional fraud alert representative for a fictional bank called "SecureTrust Bank".

Your job:
- Handle a single fraud alert call/session at a time.
- Use ONLY fake demo data loaded from the fraud database by tools.
- Never ask for full card numbers, PINs, passwords, OTPs, or any sensitive credentials.

High-level call flow:
1. Greet the customer and introduce yourself clearly as SecureTrust Bank fraud monitoring.
2. Explain that you are contacting them about a suspicious transaction on their card.
3. Ask for their first name to look up the case.
4. Call the `load_fraud_case` tool exactly once after the user shares their name.
   - If the tool says no case is found, politely say you couldn't locate their record and end the call.
5. If a case is found:
   - Use the returned details to:
     - Mention the masked card ending (e.g. "card ending in 4242"),
     - Mention the merchant, amount, time, and location.
   - Ask the security question provided in the tool result (for example, "What is your favorite color?").
6. After the user answers the security question:
   - Call the `verify_security_answer` tool with the user's name and their answer.
   - If verification FAILED:
     - Tell the user that you cannot complete the verification and cannot discuss account details.
     - Call `update_fraud_status` with status "verification_failed" and a short outcome note.
     - End the call politely.
   - If verification PASSED:
     - Clearly read out the suspicious transaction details from the case.
     - Ask: "Did you make this transaction? Please answer yes or no."

7. When the user answers about the transaction:
   - If they say it WAS them:
     - Treat the case as safe.
     - Call `update_fraud_status` with status "confirmed_safe" and a short note such as
       "Customer confirmed the transaction as legitimate."
     - Tell them the transaction is marked as safe and no further action is taken.
   - If they say it was NOT them:
     - Treat the case as fraud.
     - Call `update_fraud_status` with status "confirmed_fraud" and a short note such as
       "Customer denied the transaction. Card blocked and dispute initiated (demo)."
     - Tell them you are blocking the card and raising a mock dispute (clearly mention that this is a demo).

8. End the call with a short summary of:
   - The final decision: safe, fraud, or verification failed.
   - Any mock actions taken (e.g., "card blocked in this demo scenario").

Important safety rules:
- NEVER ask for full card numbers, CVV, PIN, passwords, or OTP.
- Verification must ONLY use the security question from the database.
- Make it clear this is a demo/sandbox if appropriate.
- Keep responses short, clear, and suitable for spoken conversation.
- Do not mention tools, JSON, or internal implementation details.

Example fake customer context (for your understanding only, do not read verbatim):
- Example user name: {example_name}
- Example card ending: {example_card}
""",
        )

    @function_tool
    async def load_fraud_case(self, context: RunContext, user_name: str) -> str:
        """
        Load a fraud case for the given user name.

        The model should call this once after the user shares their name.
        """

        cases = _load_fraud_db()
        case, _ = _find_case_by_username(cases, user_name)

        if not case:
            return (
                "No fraud case was found for this user name. "
                "You should politely tell the user that you could not find a matching record "
                "and end the call."
            )

        # Build a compact description for the LLM to use
        userName = case.get("userName", "Unknown")
        cardEnding = case.get("cardEnding", "Unknown")
        amount = case.get("transactionAmount", "Unknown amount")
        merchant = case.get("transactionName", "Unknown merchant")
        category = case.get("transactionCategory", "Unknown category")
        time = case.get("transactionTime", "Unknown time")
        location = case.get("transactionLocation", "Unknown location")
        securityQuestion = case.get("securityQuestion", "Unknown question")
        status = case.get("status", "pending_review")

        return (
            f"A fraud case was found for user '{userName}'. "
            f"Masked card ending: {cardEnding}. "
            f"Suspicious transaction: {amount} at {merchant} "
            f"in category {category}, around {time} in {location}. "
            f"Security question to ask the user: {securityQuestion}. "
            f"Current status is '{status}'. "
            "Use this information to carefully describe the transaction in your own words, "
            "ask ONLY the security question provided, and then verify their answer "
            "using the verify_security_answer tool."
        )

    @function_tool
    async def verify_security_answer(
        self,
        context: RunContext,
        user_name: str,
        provided_answer: str,
    ) -> str:
        """
        Verify the user's security answer against the stored answer.

        Returns a short string describing whether verification passed or failed.
        """

        cases = _load_fraud_db()
        case, _ = _find_case_by_username(cases, user_name)

        if not case:
            return (
                "No fraud case found for this user name while verifying. "
                "You should tell the user you cannot verify their identity and end the call."
            )

        expected = str(case.get("securityAnswer", "")).strip().lower()
        given = (provided_answer or "").strip().lower()

        if not expected:
            return (
                "No security answer is stored for this case. "
                "You should say that you cannot perform verification and end the call."
            )

        if expected == given:
            return (
                "Verification PASSED. You may now proceed to describe the suspicious "
                "transaction in detail and ask if the user made it."
            )

        return (
            "Verification FAILED. You must tell the user that you cannot verify their identity "
            "and cannot proceed with account details. "
            "Then you should call update_fraud_status with status 'verification_failed' and end the call."
        )

    @function_tool
    async def update_fraud_status(
        self,
        context: RunContext,
        user_name: str,
        status: str,
        outcome_note: str,
    ) -> str:
        """
        Update the fraud case status for the given user.

        Valid statuses (for this demo): 'confirmed_safe', 'confirmed_fraud', 'verification_failed'.
        Any other status will be stored as-is but should not be used normally.
        """

        cases = _load_fraud_db()
        case, idx = _find_case_by_username(cases, user_name)

        if case is None or idx is None:
            return (
                "No fraud case found to update for this user name. "
                "You should verbally acknowledge that you could not update the record."
            )

        # Update fields
        status_clean = status.strip().lower() if status else "unknown"
        case["status"] = status_clean
        case["outcomeNote"] = outcome_note or ""

        # Add/update a lastUpdated timestamp
        case["lastUpdated"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        cases[idx] = case
        _save_fraud_db(cases)

        logger.info("Fraud case for user '%s' updated to status '%s'", user_name, status_clean)

        return (
            f"Fraud case updated. Final status: {status_clean}. "
            f"Outcome note: {case['outcomeNote']}."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        # STT: speech to text
        stt=deepgram.STT(model="nova-3"),
        # LLM: brain of the fraud agent
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # TTS: Murf Falcon voice
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # Turn detection & VAD
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
