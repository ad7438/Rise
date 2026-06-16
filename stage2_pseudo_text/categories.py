"""Category definitions and label helpers for Stage 2 pseudo text."""

from __future__ import annotations

from typing import Dict


CATEGORY_SPECS: Dict[str, Dict[str, object]] = {
    "human": {"zh": "\u4eba\u7269", "en": "person", "aliases": ["human", "person", "people", "man", "woman", "girl", "boy", "soldier", "portrait", "figure", "human figure"]},
    "bird": {"zh": "\u9e1f", "en": "bird", "aliases": ["owl", "duck", "eagle", "parrot", "avian", "bittern", "bat"]},
    "mammal": {"zh": "\u54fa\u4e73\u7c7b\u52a8\u7269", "en": "mammal", "aliases": ["rodent", "cat", "dog", "deer", "bear", "leopard", "tiger", "monkey", "cheetah", "lion"]},
    "reptile_amphibian": {"zh": "\u722c\u884c\u6216\u4e24\u6816\u7c7b\u52a8\u7269", "en": "reptile or amphibian", "aliases": ["frog", "toad", "amphibian", "lizard", "gecko", "chameleon", "iguana", "snake", "crocodile", "alligator", "reptile"]},
    "aquatic_animal": {"zh": "\u6c34\u751f\u52a8\u7269", "en": "aquatic animal", "aliases": ["fish", "batfish", "pipefish", "seahorse", "leafy sea dragon", "leafy_sea_dragon", "cephalopod", "octopus", "squid", "cuttlefish", "nautilus", "crab", "shrimp", "prawn", "lobster", "pagurian", "hermit crab", "hermit_crab", "crocodilefish", "ghostpipefish"]},
    "arthropod": {"zh": "\u8282\u80a2\u52a8\u7269", "en": "arthropod", "aliases": ["insect", "bug", "beetle", "moth", "butterfly", "cicada", "katydid", "bee", "mantis", "grasshopper", "fly", "spider", "arachnid", "centipede", "dragonfly", "ant", "caterpillar", "stickinsect", "stick insect"]},
    "plant": {"zh": "\u690d\u7269", "en": "plant", "aliases": ["leaf", "flower", "tree", "branch", "grass", "moss", "twig", "bark", "vine", "vegetation"]},
    "manmade_object": {"zh": "\u4eba\u9020\u7269\u4f53", "en": "man-made object", "aliases": ["artifact", "object", "tool", "weapon", "vehicle", "statue", "sculpture", "painting", "poster", "mannequin", "decoration", "building", "fabric", "cloth", "indoor"]},
    "other_animal": {"zh": "\u5176\u4ed6\u52a8\u7269", "en": "animal", "aliases": ["animal", "creature"]},
    "other_non_animal": {"zh": "\u5176\u4ed6\u975e\u52a8\u7269\u76ee\u6807", "en": "non-animal object", "aliases": ["non_animal", "non-animal", "background object", "object_like_target"]},
    "unknown": {"zh": "\u672a\u77e5\u76ee\u6807", "en": "unknown target", "aliases": ["uncertain", "unsure", "none", "unknown", "not sure", "cannot tell"]},
}

DEFAULT_CATEGORY_KEYS = tuple(CATEGORY_SPECS.keys())

_ALIAS_TO_KEY = {}
for _key, _spec in CATEGORY_SPECS.items():
    _ALIAS_TO_KEY[_key] = _key
    _ALIAS_TO_KEY[_spec["zh"]] = _key
    for _alias in _spec.get("aliases", []):
        _ALIAS_TO_KEY[_alias] = _key

SIZE_KEY_TO_ZH = {
    "small": "\u5c0f\u578b",
    "medium": "\u4e2d\u7b49\u5927\u5c0f",
    "large": "\u5927\u578b",
}

LOCATION_KEY_TO_ZH = {
    "top_left": "\u5de6\u4e0a\u89d2",
    "top_center": "\u4e0a\u65b9",
    "top_right": "\u53f3\u4e0a\u89d2",
    "middle_left": "\u5de6\u4fa7",
    "middle_center": "\u4e2d\u592e",
    "middle_right": "\u53f3\u4fa7",
    "bottom_left": "\u5de6\u4e0b\u89d2",
    "bottom_center": "\u4e0b\u65b9",
    "bottom_right": "\u53f3\u4e0b\u89d2",
}

LOCATION_KEY_TO_EN = {
    "top_left": "top-left",
    "top_center": "top-center",
    "top_right": "top-right",
    "middle_left": "middle-left",
    "middle_center": "center",
    "middle_right": "middle-right",
    "bottom_left": "bottom-left",
    "bottom_center": "bottom-center",
    "bottom_right": "bottom-right",
}

SIZE_KEY_TO_EN = {
    "small": "small",
    "medium": "medium",
    "large": "large",
}

ANIMAL_CATEGORY_KEYS = {
    "bird",
    "mammal",
    "reptile_amphibian",
    "aquatic_animal",
    "arthropod",
    "other_animal",
}


def normalize_category_key(value: str | None) -> str:
    """Normalize free-form category text into one supported category key."""

    if not value:
        return "unknown"
    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    if normalized in _ALIAS_TO_KEY:
        return _ALIAS_TO_KEY[normalized]
    normalized_flat = normalized.replace("_", "")
    for alias, key in _ALIAS_TO_KEY.items():
        alias_flat = alias.lower().replace("_", "").replace("-", "").replace(" ", "")
        if normalized_flat == alias_flat:
            return key
    return "unknown"


def category_label_zh(category_key: str) -> str:
    return str(CATEGORY_SPECS.get(category_key, CATEGORY_SPECS["unknown"])["zh"])


def category_label_en(category_key: str) -> str:
    return str(CATEGORY_SPECS.get(category_key, CATEGORY_SPECS["unknown"])["en"])


def category_subject_zh(category_key: str) -> str:
    if category_key == "human":
        return "\u4e00\u540d"
    if category_key == "plant":
        return "\u4e00\u682a"
    if category_key in ANIMAL_CATEGORY_KEYS:
        return "\u4e00\u53ea"
    return "\u4e00\u4e2a"
