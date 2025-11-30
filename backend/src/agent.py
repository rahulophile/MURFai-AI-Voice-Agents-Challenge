import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("ecommerce_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Simple Product Catalog (Dr Abhishek Shop)
# -------------------------

CATALOG: List[Dict] = [
    {
        "id": "mug-001",
        "name": "Stoneware Chai Mug",
        "description": "Hand-glazed ceramic mug perfect for masala chai.",
        "price": 299,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
        "sizes": [],
    },
    {
        "id": "mug-002",
        "name": "Insulated Travel Mug",
        "description": "Keeps chai warm on your way to work.",
        "price": 599,
        "currency": "INR",
        "category": "mug",
        "color": "white",
        "sizes": [],
    },
    {
        "id": "tee-001",
        "name": "Dr Abhishek Tee (Cotton)",
        "description": "Comfort-fit cotton t-shirt with subtle logo.",
        "price": 799,
        "currency": "INR",
        "category": "tshirt",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tee-002",
        "name": "Casual Cotton Tee",
        "description": "Everyday cotton t-shirt, breathable and soft.",
        "price": 299,
        "currency": "INR",
        "category": "tshirt",
        "color": "white",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tee-003",
        "name": "Graphic Tee",
        "description": "Printed graphic t-shirt with vibrant design.",
        "price": 499,
        "currency": "INR",
        "category": "tshirt",
        "color": "navy",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "hoodie-001",
        "name": "Cozy Hoodie",
        "description": "Warm pullover hoodie, fleece-lined.",
        "price": 1499,
        "currency": "INR",
        "category": "hoodie",
        "color": "grey",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "hoodie-002",
        "name": "Black Zip Hoodie",
        "description": "Lightweight zip-up hoodie, black.",
        "price": 1299,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": ["S", "M", "L"],
    },
    {
        "id": "rain-001",
        "name": "Light Raincoat",
        "description": "Waterproof light raincoat, packable.",
        "price": 1299,
        "currency": "INR",
        "category": "raincoat",
        "color": "yellow",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "rain-002",
        "name": "Heavy Duty Raincoat",
        "description": "Heavy-duty rainproof coat for monsoon.",
        "price": 2499,
        "currency": "INR",
        "category": "raincoat",
        "color": "navy",
        "sizes": ["L", "XL"],
    },
    {
        "id": "phone-001",
        "name": "Redmi Note (Entry)",
        "description": "Affordable Redmi smartphone with solid features.",
        "price": 12000,
        "currency": "INR",
        "category": "mobile",
        "color": "blue",
        "sizes": [],
    },
    {
        "id": "phone-002",
        "name": "Oppo A-Series",
        "description": "Stylish Oppo phone with good camera.",
        "price": 18000,
        "currency": "INR",
        "category": "mobile",
        "color": "green",
        "sizes": [],
    },
    {
        "id": "phone-003",
        "name": "Samsung M-Series",
        "description": "Mid-range Samsung phone for everyday use.",
        "price": 25000,
        "currency": "INR",
        "category": "mobile",
        "color": "black",
        "sizes": [],
    },
    {
        "id": "phone-004",
        "name": "iPhone (Standard)",
        "description": "Apple iPhone model example.",
        "price": 50000,
        "currency": "INR",
        "category": "mobile",
        "color": "white",
        "sizes": [],
    },
]

ORDERS_FILE = "orders.json"

if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w") as f:
        json.dump([], f)

# -------------------------
# Session Userdata
# -------------------------

@dataclass
class Userdata:
    customer_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    cart: List[Dict] = field(default_factory=list)   # [{product_id, quantity, attrs}]
    orders: List[Dict] = field(default_factory=list) # placed orders in session
    history: List[Dict] = field(default_factory=list)

# -------------------------
# Merchant-layer helpers
# -------------------------

def _load_all_orders() -> List[Dict]:
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_order(order: Dict):
    orders = _load_all_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)


def list_products(filters: Optional[Dict] = None) -> List[Dict]:
    """
    Naive filtering by category, max_price, color, etc.
    Supports category synonyms like 'phones' -> 'mobile', 'tees' -> 'tshirt'.
    """
    filters = filters or {}
    results = []

    query = filters.get("q")
    category = filters.get("category")
    max_price = filters.get("max_price")
    color = filters.get("color")

    # normalize category
    if category:
        cat = category.lower()
        if cat in ("phone", "phones", "mobile", "mobiles", "mobile phone"):
            category = "mobile"
        elif cat in ("tee", "tees", "tshirt", "t-shirts"):
            category = "tshirt"
        else:
            category = cat

    for p in CATALOG:
        ok = True

        if category:
            pcat = p.get("category", "").lower()
            if category not in pcat and pcat not in category:
                ok = False

        if max_price is not None:
            try:
                if p.get("price", 0) > int(max_price):
                    ok = False
            except Exception:
                pass

        if color:
            if p.get("color", "").lower() != color.lower():
                ok = False

        if query:
            q = query.lower()
            if q not in p.get("name", "").lower() and q not in p.get("description", "").lower():
                ok = False

        if ok:
            results.append(p)

    return results


def find_product_by_ref(ref_text: str, candidates: Optional[List[Dict]] = None) -> Optional[Dict]:
    """
    Resolve things like:
    - 'second hoodie'
    - 'black hoodie'
    - explicit id like 'mug-001'
    """
    ref = (ref_text or "").lower().strip()
    cand = candidates if candidates is not None else CATALOG

    # ordinals
    ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3}
    for word, idx in ordinals.items():
        if word in ref and idx < len(cand):
            return cand[idx]

    # direct id
    for p in cand:
        if p["id"].lower() == ref:
            return p

    # color + category
    for p in cand:
        c = p.get("color", "").lower()
        cat = p.get("category", "").lower()
        if c and cat and c in ref and cat in ref:
            return p

    # name substring
    for p in cand:
        name = p["name"].lower()
        if all(tok in name for tok in ref.split() if len(tok) > 2):
            return p

    # numeric index (e.g. '2' => second)
    for tok in ref.split():
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(cand):
                return cand[idx]

    return None


def create_order_object(line_items: List[Dict], currency: str = "INR") -> Dict:
    """
    line_items: [{product_id, quantity, attrs}]
    -> returns order dict and saves it to orders.json
    """
    items = []
    total = 0

    for li in line_items:
        pid = li.get("product_id")
        qty = int(li.get("quantity", 1))
        prod = next((p for p in CATALOG if p["id"] == pid), None)
        if not prod:
            raise ValueError(f"Product {pid} not found")
        line_total = prod["price"] * qty
        total += line_total
        items.append(
            {
                "product_id": pid,
                "name": prod["name"],
                "unit_price": prod["price"],
                "quantity": qty,
                "line_total": line_total,
                "attrs": li.get("attrs", {}),
            }
        )

    order = {
        "id": f"order-{str(uuid.uuid4())[:8]}",
        "items": items,
        "total": total,
        "currency": currency,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    _save_order(order)
    return order


def get_most_recent_order() -> Optional[Dict]:
    all_orders = _load_all_orders()
    if not all_orders:
        return None
    return all_orders[-1]

# -------------------------
# Tools exposed to LLM
# -------------------------

@function_tool
async def show_catalog(
    ctx: RunContext[Userdata],
    q: Annotated[Optional[str], Field(description="Search query (optional)", default=None)] = None,
    category: Annotated[Optional[str], Field(description="Category (optional)", default=None)] = None,
    max_price: Annotated[Optional[int], Field(description="Maximum price (optional)", default=None)] = None,
    color: Annotated[Optional[str], Field(description="Color (optional)", default=None)] = None,
) -> str:
    """
    Return a spoken summary of matching products (name, price, id).
    """
    filters = {"q": q, "category": category, "max_price": max_price, "color": color}
    prods = list_products({k: v for k, v in filters.items() if v is not None})

    if not prods:
        return "I couldn’t find anything matching that. You can try another category, price range, or color."

    lines = [
        f"Here are the top {min(6, len(prods))} items I found at Dr Abhishek Shop:"
    ]
    for idx, p in enumerate(prods[:6], start=1):
        size_info = (
            f" available sizes: {', '.join(p['sizes'])}"
            if p.get("sizes")
            else ""
        )
        lines.append(
            f"{idx}. {p['name']} — {p['price']} {p['currency']} (id: {p['id']}){size_info}"
        )
    lines.append(
        "You can say things like: 'I want the second hoodie in size M' or 'add mug-001 to my cart, quantity two'."
    )
    return "\n".join(lines)


@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_ref: Annotated[str, Field(description="Product id, name, or reference like 'second hoodie'")],
    quantity: Annotated[int, Field(description="Quantity", default=1)] = 1,
    size: Annotated[Optional[str], Field(description="Size (if applicable)", default=None)] = None,
) -> str:
    """
    Add a resolved product to the session cart.
    """
    userdata = ctx.userdata
    prod = find_product_by_ref(product_ref, CATALOG)
    if not prod:
        return (
            "I couldn’t figure out which product you meant. "
            "Try saying the item id like 'mug-001' or ask me to show the catalog again."
        )

    userdata.cart.append(
        {
            "product_id": prod["id"],
            "quantity": int(quantity),
            "attrs": {"size": size} if size else {},
        }
    )
    userdata.history.append(
        {
            "time": datetime.utcnow().isoformat() + "Z",
            "action": "add_to_cart",
            "product_id": prod["id"],
            "quantity": int(quantity),
        }
    )
    return f"Added {quantity} x {prod['name']} to your cart. What would you like to do next?"


@function_tool
async def show_cart(
    ctx: RunContext[Userdata],
) -> str:
    """
    Show current cart items and total.
    """
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is currently empty. You can say 'show me hoodies under 1500' or 'show mugs'."

    lines = ["Here’s what’s in your cart:"]
    total = 0

    for li in userdata.cart:
        p = next((x for x in CATALOG if x["id"] == li["product_id"]), None)
        if not p:
            continue
        qty = li.get("quantity", 1)
        line_total = p["price"] * qty
        total += line_total
        sz = li.get("attrs", {}).get("size")
        sz_text = f", size {sz}" if sz else ""
        lines.append(
            f"- {p['name']} x {qty}{sz_text}: {line_total} {p['currency']}"
        )

    lines.append(f"Cart total: {total} INR.")
    lines.append(
        "You can say 'place my order' to checkout or 'clear my cart' if you want to start over."
    )
    return "\n".join(lines)


@function_tool
async def clear_cart(
    ctx: RunContext[Userdata],
) -> str:
    """
    Clear the current cart.
    """
    userdata = ctx.userdata
    userdata.cart = []
    userdata.history.append(
        {"time": datetime.utcnow().isoformat() + "Z", "action": "clear_cart"}
    )
    return "I’ve cleared your cart. What would you like to browse next?"


@function_tool
async def place_order(
    ctx: RunContext[Userdata],
    confirm: Annotated[
        bool, Field(description="Whether the user has confirmed placing the order", default=True)
    ] = True,
) -> str:
    """
    Convert the current cart into an order and persist it.
    """
    userdata = ctx.userdata

    if not userdata.cart:
        return "Your cart is empty, so there’s nothing to place yet. Would you like to add something first?"

    if not confirm:
        return "Okay, I won’t place the order yet. You can review your cart or add more items."

    line_items = []
    for li in userdata.cart:
        line_items.append(
            {
                "product_id": li["product_id"],
                "quantity": li.get("quantity", 1),
                "attrs": li.get("attrs", {}),
            }
        )

    order = create_order_object(line_items)
    userdata.orders.append(order)
    userdata.history.append(
        {
            "time": datetime.utcnow().isoformat() + "Z",
            "action": "place_order",
            "order_id": order["id"],
        }
    )
    userdata.cart = []

    return (
        f"Your order is placed. Order ID {order['id']} with total {order['total']} {order['currency']}. "
        "You can say 'what did I just buy' to hear the summary again."
    )


@function_tool
async def last_order(
    ctx: RunContext[Userdata],
) -> str:
    """
    Read back the most recent order.
    """
    ord_data = get_most_recent_order()
    if not ord_data:
        return "You don’t have any past orders yet in this demo."

    lines = [f"Your most recent order is {ord_data['id']} created at {ord_data['created_at']}."]
    for it in ord_data["items"]:
        lines.append(
            f"- {it['name']} x {it['quantity']}: {it['line_total']} {ord_data['currency']}"
        )
    lines.append(f"Total: {ord_data['total']} {ord_data['currency']}.")
    return "\n".join(lines)

# -------------------------
# Agent class
# -------------------------

class GameMasterAgent(Agent):
    def __init__(self) -> None:
        instructions = """
You are 'Ramu Kaka', the friendly voice-based shopkeeper for Dr Abhishek Shop.

Role:
- Help the user browse products, filter by category/price/color, and understand options.
- Add items to the cart, show the cart, clear it, and place orders.
- Answer questions like "What did I just buy?" using the last_order tool.

Behavior:
- Always speak like a real Indian shopkeeper: warm, polite, and to the point.
- Keep responses short and easy to understand for voice.
- When listing products, include index, name, price, and id.
- When the user sounds ready to buy, confirm details clearly before placing the order.

Tools:
- Use show_catalog to explore products.
- Use add_to_cart, show_cart, clear_cart, place_order, and last_order to manage orders.
Do not talk about tools or JSON; just behave like a natural shopkeeper.
"""
        super().__init__(
            instructions=instructions,
            tools=[show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order],
        )

# -------------------------
# Prewarm & entrypoint
# -------------------------

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.warning(f"VAD prewarm failed: {e}")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("🛍️ Starting E-commerce Agent (Ramu Kaka)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
