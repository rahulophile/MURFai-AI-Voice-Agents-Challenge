import logging
import os
import json
import datetime
from typing import List, Dict, Optional, Tuple

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

CATALOG_PATH = "shared-data/day7_catalog.json"
RECIPES_PATH = "shared-data/day7_recipes.json"
ORDERS_DIR = "orders"


def _load_catalog() -> List[dict]:
    """Load the food & grocery catalog from JSON."""
    if not os.path.exists(CATALOG_PATH):
        logger.warning("Catalog file not found at %s", CATALOG_PATH)
        return []

    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load catalog: %s", e)
        return []

    if not isinstance(data, list):
        logger.error("Catalog must be a list of items.")
        return []

    # Basic cleanup
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        price = item.get("price", None)
        if not name or price is None:
            continue
        cleaned.append(item)
    return cleaned


def _load_recipes() -> Dict[str, List[dict]]:
    """Load recipes mapping (dish -> list of {item_name, quantity})."""
    if not os.path.exists(RECIPES_PATH):
        logger.warning("Recipes file not found at %s", RECIPES_PATH)
        return {}

    try:
        with open(RECIPES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load recipes: %s", e)
        return {}

    if not isinstance(data, dict):
        logger.error("Recipes file must be a JSON object mapping dish -> items.")
        return {}

    return data


def _find_item_by_name(catalog: List[dict], name_or_id: str) -> Optional[dict]:
    """Find an item in the catalog by name or id (very simple fuzzy match)."""
    if not name_or_id:
        return None

    key = name_or_id.strip().lower()
    best = None
    best_score = 0

    for item in catalog:
        name = str(item.get("name", "")).lower()
        item_id = str(item.get("id", "")).lower()
        tags = " ".join(str(t).lower() for t in item.get("tags", []))

        text = f"{name} {item_id} {tags}"

        score = 0
        for w in key.split():
            if w and w in text:
                score += 1

        if score > best_score:
            best_score = score
            best = item

    return best


class Assistant(Agent):
    def __init__(self) -> None:
        self.catalog: List[dict] = _load_catalog()
        self.recipes: Dict[str, List[dict]] = _load_recipes()
        # Simple in-memory cart for the current session
        # Each entry: {"item_id", "name", "unit_price", "quantity"}
        self.cart: List[dict] = []

        # Build a short overview of the catalog
        if self.catalog:
            catalog_overview_lines: List[str] = []
            for item in self.catalog[:15]:
                nm = item.get("name", "")
                cat = item.get("category", "")
                price = item.get("price", "")
                catalog_overview_lines.append(f"- {nm} ({cat}) – {price}")
            catalog_overview = "\n".join(catalog_overview_lines)
        else:
            catalog_overview = "No items loaded. Make sure the catalog JSON exists."

        super().__init__(
            instructions=f"""
You are a friendly, focused **Food & Grocery Ordering Assistant** for a fictional store called "QuickBasket".

Your job:
- Help users order groceries, snacks, and simple meal ingredients.
- Understand what items they want, quantities, and "ingredients for X" style requests.
- Keep track of their cart and, when they are done, place the order and save it using tools.

You have access to:
- A catalog of items (loaded from JSON).
- A small set of "recipes" that map dish names (like "peanut butter sandwich") to multiple items.

High-level behavior:
1. When the conversation starts:
   - Greet the user warmly as QuickBasket's ordering assistant.
   - Briefly explain that you can add items, update the cart, and even add ingredients for simple dishes.

2. Use the tools to manage the cart:
   - When the user asks for a specific item (e.g., "add 2 breads" or "I want 1 litre of milk"):
     - Call `add_to_cart` with the item name and quantity.
   - When they want to remove or change items:
     - Use `remove_from_cart` or `update_cart_item`.
   - When they ask "what's in my cart?" or similar:
     - Call `list_cart` and then read out a short summary of the cart contents.

3. Ingredients for X:
   - If the user says something like:
     - "I need ingredients for a peanut butter sandwich."
     - "Get me what I need for pasta for two."
   - Call `add_recipe_to_cart` with the dish name (like "peanut butter sandwich" or "pasta for two").
   - Then clearly confirm what was added (e.g., "I added bread and peanut butter for your sandwich.").

4. Placing the order:
   - When the user says things like:
     - "That's all."
     - "Place my order."
     - "I'm done."
   - First, call `list_cart` if needed and give a short spoken summary and total.
   - Then ask for any simple customer info you want to capture (for example, their name or area).
   - Finally, call `place_order` with the cart and basic customer info.
   - After `place_order` returns, tell the user that the order has been placed in this demo and mention the order total.

Important details:
- Keep your responses short and spoken-friendly.
- Always confirm changes to the cart so the user feels in control.
- If an item or recipe can't be found, explain that briefly and suggest alternatives.
- DO NOT mention tools, JSON files, or internal implementation details to the user.

Catalog snapshot (for your reference only, do not read this verbatim):
{catalog_overview}
""",
        )

    # --- Internal helpers for cart management (not tools themselves) ---

    def _add_item_to_cart(self, item: dict, quantity: int) -> None:
        if quantity <= 0:
            return
        item_id = item.get("id")
        name = item.get("name")
        price = float(item.get("price", 0))

        for entry in self.cart:
            if entry["item_id"] == item_id:
                entry["quantity"] += quantity
                return

        self.cart.append(
            {
                "item_id": item_id,
                "name": name,
                "unit_price": price,
                "quantity": quantity,
            }
        )

    def _remove_item_from_cart(self, item_name_or_id: str) -> bool:
        key = (item_name_or_id or "").strip().lower()
        if not key:
            return False

        for idx, entry in enumerate(self.cart):
            if (
                key in str(entry["name"]).lower()
                or key == str(entry["item_id"]).lower()
            ):
                self.cart.pop(idx)
                return True
        return False

    def _update_cart_item(self, item_name_or_id: str, quantity: int) -> bool:
        key = (item_name_or_id or "").strip().lower()
        if not key:
            return False

        for entry in self.cart:
            if (
                key in str(entry["name"]).lower()
                or key == str(entry["item_id"]).lower()
            ):
                if quantity <= 0:
                    self.cart.remove(entry)
                else:
                    entry["quantity"] = quantity
                return True
        return False

    def _cart_summary(self) -> Tuple[str, float]:
        """Return (text summary, total_price)."""
        if not self.cart:
            return "Your cart is currently empty.", 0.0

        lines = []
        total = 0.0
        for entry in self.cart:
            name = entry["name"]
            qty = entry["quantity"]
            unit = entry["unit_price"]
            line_total = qty * unit
            total += line_total
            lines.append(f"{qty} x {name} (each {unit}) = {line_total}")

        summary = "Items in cart:\n" + "\n".join(lines) + f"\nTotal: {total}"
        return summary, total

    # --- Tools exposed to the LLM ---

    @function_tool
    async def add_to_cart(
        self,
        context: RunContext,
        item_name: str,
        quantity: int = 1,
    ) -> str:
        """
        Add a specific catalog item to the cart.

        Args:
            item_name: Name or description of the item (e.g., "bread", "Amul milk").
            quantity: How many units to add (default 1).
        """
        if not self.catalog:
            return "No catalog items are available. You should tell the user you can't add items right now."

        if quantity <= 0:
            quantity = 1

        item = _find_item_by_name(self.catalog, item_name)
        if not item:
            return (
                f"No matching item found in the catalog for '{item_name}'. "
                "You should apologize briefly and ask the user to rephrase or choose another item."
            )

        self._add_item_to_cart(item, quantity)

        return (
            f"Added {quantity} x '{item.get('name')}' to the cart. "
            "Confirm this to the user in a friendly way."
        )

    @function_tool
    async def remove_from_cart(
        self,
        context: RunContext,
        item_name: str,
    ) -> str:
        """
        Remove an item from the cart by name or id.
        """
        if not self.cart:
            return "The cart is empty; there is nothing to remove. Tell the user the cart is empty."

        removed = self._remove_item_from_cart(item_name)
        if not removed:
            return (
                f"No matching item found in the cart for '{item_name}'. "
                "Tell the user you couldn't find that item in their cart."
            )

        return (
            f"Removed '{item_name}' from the cart. "
            "Confirm to the user that the item has been removed."
        )

    @function_tool
    async def update_cart_item(
        self,
        context: RunContext,
        item_name: str,
        quantity: int,
    ) -> str:
        """
        Update the quantity of an item in the cart.
        If quantity <= 0, the item will be removed.
        """
        if not self.cart:
            return "The cart is currently empty. There is nothing to update."

        updated = self._update_cart_item(item_name, quantity)
        if not updated:
            return (
                f"No matching item found in the cart for '{item_name}'. "
                "Tell the user you couldn't find that item in their cart."
            )

        if quantity <= 0:
            return (
                f"Quantity updated to {quantity}, so the item is removed from the cart. "
                "Tell the user the item has been removed."
            )

        return (
            f"Updated quantity of '{item_name}' to {quantity}. "
            "Confirm this change to the user."
        )

    @function_tool
    async def list_cart(self, context: RunContext) -> str:
        """
        Return a summary of the current cart contents and total.
        """
        summary, total = self._cart_summary()
        return (
            f"{summary} "
            f"The numeric total is {total}. "
            "Read this back briefly to the user in a natural way."
        )

    @function_tool
    async def add_recipe_to_cart(
        self,
        context: RunContext,
        recipe_name: str,
        servings: int = 1,
    ) -> str:
        """
        Add multiple items for a dish (ingredients for X) based on recipe mapping.

        Args:
            recipe_name: Name of the dish (e.g., 'peanut butter sandwich', 'pasta for two').
            servings: Optional multiplier for quantities (default 1).
        """
        if not self.recipes:
            return (
                "No recipes are configured. You should tell the user that you can't "
                "automatically add ingredients for dishes right now."
            )

        key = (recipe_name or "").strip().lower()
        # Simple lookup by lowercased key
        matched_key = None
        for rname in self.recipes.keys():
            if key == rname.lower():
                matched_key = rname
                break

        if not matched_key:
            return (
                f"No recipe found for '{recipe_name}'. "
                "Tell the user you don't have a preset for that dish but can help add items individually."
            )

        recipe_items = self.recipes[matched_key]
        if not isinstance(recipe_items, list):
            return (
                "The recipe format is invalid. You should apologize and fall back to normal item ordering."
            )

        if servings <= 0:
            servings = 1

        added_items: List[str] = []
        for entry in recipe_items:
            if not isinstance(entry, dict):
                continue
            item_name = entry.get("item_name")
            base_qty = int(entry.get("quantity", 1))
            qty = max(1, base_qty * servings)
            item = _find_item_by_name(self.catalog, item_name)
            if not item:
                continue
            self._add_item_to_cart(item, qty)
            added_items.append(f"{qty} x {item.get('name')}")

        if not added_items:
            return (
                "The recipe did not match any items in the catalog. "
                "Tell the user you weren't able to add ingredients automatically."
            )

        added_text = ", ".join(added_items)
        return (
            f"For recipe '{matched_key}', added these items to the cart: {added_text}. "
            "Explain this to the user in a friendly way."
        )

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        customer_name: str = "",
        address: str = "",
    ) -> str:
        """
        Place the current order: save it to a JSON file and clear the cart.

        Args:
            customer_name: Optional name to store with the order.
            address: Optional address or location text.
        """
        if not self.cart:
            return (
                "The cart is empty. You should tell the user that there is nothing to place as an order."
            )

        summary, total = self._cart_summary()

        order = {
            "order_id": f"order_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "customer_name": customer_name or "unknown",
            "address": address or "unknown",
            "items": [
                {
                    "name": e["name"],
                    "quantity": e["quantity"],
                    "unit_price": e["unit_price"],
                    "line_total": e["quantity"] * e["unit_price"],
                }
                for e in self.cart
            ],
            "total": total,
        }

        os.makedirs(ORDERS_DIR, exist_ok=True)
        filename = f"{order['order_id']}.json"
        path = os.path.join(ORDERS_DIR, filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(order, f, indent=2)
            logger.info("Saved order to %s: %s", path, order)
        except Exception as e:
            logger.error("Failed to save order: %s", e)
            return (
                "There was an error saving the order. You should apologize "
                "and tell the user this is only a demo."
            )

        # Clear cart after successful order placement
        self.cart = []

        return (
            f"Order placed successfully with ID {order['order_id']}. "
            f"Total amount was {total}. "
            "Tell the user that their order has been placed in this demo and thank them."
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
