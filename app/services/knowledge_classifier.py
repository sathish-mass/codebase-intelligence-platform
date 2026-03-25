from typing import Dict, List, Optional

from app.services.knowledge_schema import normalize_tag_list


# -----------------------------------------------------------------------------
# Seed signals only
# -----------------------------------------------------------------------------
# These are NOT the final limits of the product.
# They are only initial hints so the classifier can start working.
#
# Later the platform can learn/discover additional system identifiers from:
# - uploaded file names
# - folder names
# - imports
# - class names
# - repeated client names
# - documentation titles
# - API spec names
# -----------------------------------------------------------------------------
SYSTEM_SEED_KEYWORDS = {
    "xts": ["xts", "xtsconnect", "xtsclient"],
    "dhan": ["dhan", "dhanhq"],
    "tradehull": ["tradehull"],
    "kite": ["kite", "kiteconnect", "zerodha"],
    "upstox": ["upstox"],
    "fyers": ["fyers"],
    "angel": ["angel", "angelone", "smartapi"],
    "groww": ["groww"],
    "sharekhan": ["sharekhan"],
    "5paisa": ["5paisa"],
    "iifl": ["iifl"],
}


ROLE_KEYWORDS = {
    "order": ["order", "place_order", "order_placement"],
    "placement": ["placement", "place"],
    "modify": ["modify", "modification", "update_order"],
    "cancel": ["cancel", "cancel_order"],
    "stoploss": ["stoploss", "sl", "stop_loss"],
    "target": ["target", "take_profit", "tp"],
    "quantity": ["quantity", "qty", "freeze_quantity"],
    "auth": ["auth", "login", "token", "session"],
    "wrapper": ["wrapper", "adapter", "dispatcher"],
    "helper": ["helper", "util", "utility", "support_library"],
    "validation": ["validate", "validation"],
    "api": ["api", "endpoint", "route", "router"],
    "config": ["config", "settings"],
    "websocket": ["websocket", "socket", "stream"],
    "margin": ["margin"],
    "position": ["position", "holdings"],
}


STYLE_KEYWORDS = {
    "platform_style": ["tradehull"],
    "system_style": ["broker", "order_placement", "place_order"],
    "helper_style": ["helper", "utility", "support_library"],
    "service_style": ["service"],
    "route_style": ["router", "endpoint", "route"],
    "multi_system_style": ["multi_broker", "wrapper", "adapter", "dispatcher"],
}


def collect_system_tags(file_path: str, file_name: str, content: str) -> List[str]:
    """
    Infer system/entity tags from multiple text signals.

    Important:
    - This is a seed-based classifier, not a hard product limit.
    - Tags can represent brokers, vendors, platforms, libraries, or internal systems.
    """
    haystack = f"{file_path}\n{file_name}\n{content}".lower()
    tags = []

    for system_name, keywords in SYSTEM_SEED_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            tags.append(system_name)

    return sorted(set(tags))


def collect_role_tags(file_name: str, symbol_name: Optional[str], content: str) -> List[str]:
    haystack = f"{file_name}\n{symbol_name or ''}\n{content}".lower()
    tags = []

    for role_name, keywords in ROLE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            tags.append(role_name)

    return sorted(set(tags))


def collect_style_tags(file_name: str, symbol_name: Optional[str], content: str) -> List[str]:
    haystack = f"{file_name}\n{symbol_name or ''}\n{content}".lower()
    tags = []

    for style_name, keywords in STYLE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            tags.append(style_name)

    return sorted(set(tags))


def infer_content_kind(source_type: str, file_name: str, content: str) -> str:
    name = (file_name or "").lower()

    if source_type == "api_spec":
        return "api_spec"

    if source_type == "documentation":
        return "doc"

    if source_type == "config":
        return "config"

    if "test" in name:
        return "test"

    if "example" in name or "sample" in name:
        return "example"

    if source_type == "code":
        return "code"

    return "unknown"


def infer_scope_kind(
    system_tags: List[str],
    role_tags: List[str],
    style_tags: List[str],
    file_name: str,
    symbol_name: Optional[str],
    content: str,
) -> str:
    """
    Decide how broad/specific this knowledge item is.

    Possible meanings:
    - broker_specific / system_specific: focused on one concrete external system
    - shared: common helper/utility/platform style used across systems
    - wrapper: bridges multiple systems or dispatches between them
    - abstraction: interface/base/abstract layer
    - hybrid: mixed-purpose and not cleanly classifiable
    - unknown: not enough signal yet
    """
    haystack = f"{file_name}\n{symbol_name or ''}\n{content}".lower()

    if "wrapper" in role_tags or "multi_system_style" in style_tags:
        return "wrapper"

    if any(word in haystack for word in ["base", "interface", "abstract"]):
        return "abstraction"

    if len(system_tags) >= 2:
        if "helper" in role_tags:
            return "shared"
        return "hybrid"

    if len(system_tags) == 1:
        system = system_tags[0]
        if system not in {"tradehull"}:
            return "broker_specific"

    if "tradehull" in system_tags or "helper" in role_tags:
        return "shared"

    return "unknown"


def infer_project_tags(system_tags: List[str], scope_kind: str, file_name: str) -> List[str]:
    tags = list(system_tags)
    name = (file_name or "").lower()

    if scope_kind == "shared":
        tags.append("shared-lib")

    if scope_kind == "wrapper":
        tags.append("wrapper")

    if "support_library" in name:
        tags.append("support-library")

    return sorted(set(tags))


def classify_knowledge_item(
    *,
    file_path: str,
    file_name: str,
    source_type: str,
    content: str,
    symbol_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Infer classification metadata using multi-signal rules.

    This is the first working classifier layer.
    It is intentionally generic and extensible.

    Returns:
    - content_kind
    - scope_kind
    - system_tags
    - role_tags
    - style_tags
    - project_tags

    Tag fields are returned as comma-separated strings for metadata compatibility.
    """
    system_tags = collect_system_tags(file_path, file_name, content)
    role_tags = collect_role_tags(file_name, symbol_name, content)
    style_tags = collect_style_tags(file_name, symbol_name, content)

    content_kind = infer_content_kind(source_type, file_name, content)
    scope_kind = infer_scope_kind(
        system_tags=system_tags,
        role_tags=role_tags,
        style_tags=style_tags,
        file_name=file_name,
        symbol_name=symbol_name,
        content=content,
    )
    project_tags = infer_project_tags(system_tags, scope_kind, file_name)

    return {
        "content_kind": content_kind,
        "scope_kind": scope_kind,
        "system_tags": normalize_tag_list(system_tags),
        "role_tags": normalize_tag_list(role_tags),
        "style_tags": normalize_tag_list(style_tags),
        "project_tags": normalize_tag_list(project_tags),
    }