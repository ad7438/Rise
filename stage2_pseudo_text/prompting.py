"""Prompt construction and pseudo-text templating for Stage 2."""

from __future__ import annotations

from .categories import (
    DEFAULT_CATEGORY_KEYS,
    LOCATION_KEY_TO_EN,
    SIZE_KEY_TO_EN,
    category_label_en,
    category_label_zh,
    category_subject_zh,
)


def build_category_prompt(location_key: str, size_key: str, category_keys=DEFAULT_CATEGORY_KEYS) -> str:
    category_list = ", ".join(category_keys)
    location_hint = LOCATION_KEY_TO_EN[location_key]
    size_hint = SIZE_KEY_TO_EN[size_key]
    return (
        "You are given three views of the same highlighted camouflaged target: a full-image overlay, a context crop, and a tight crop. "
        "Focus only on the highlighted target, not the full scene. "
        f"The object is roughly located at {location_hint} and its scale is {size_hint}. "
        "The target may be an animal, a person, a plant, or a man-made/artificial object. "
        "Do not force an animal category for humans, statues, paintings, plants, or other non-animal targets. "
        "Category guidance: "
        "'bird' covers birds and bats in flight images; "
        "'mammal' covers furry or hoofed mammals; "
        "'reptile_amphibian' covers frogs, toads, lizards, snakes, chameleons, crocodiles; "
        "'aquatic_animal' covers fish, seahorses, octopus, squid, crab, shrimp and similar marine animals; "
        "'arthropod' covers insects, spiders and similar small segmented animals; "
        "'plant' covers leaves, flowers, branches, bark and vegetation; "
        "'manmade_object' covers statues, paintings, fabric, weapons, tools and other artificial objects. "
        f"Choose one coarse category key from: {category_list}. "
        "If the target clearly belongs to one broad category above, choose that broad category instead of 'unknown'. "
        "If the target is visible but does not fit any animal category, use a matching non-animal key. "
        "If the target is too ambiguous, use 'unknown'. "
        "The evidence field must be a short descriptive phrase about the target's texture, outline, color, or similarity to the nearby background. "
        "Do not just repeat the category name. "
        "Return strict JSON with this schema only: "
        "{\"category\":\"human\",\"category_confidence\":0.72,\"evidence\":\"skin-toned human silhouette blending into mural background\"}."
    )


def compose_pseudo_text(
    category_key: str,
    location_label_zh: str,
    size_label_zh: str,
    category_confidence: float,
    uncertain_threshold: float = 0.5,
) -> str:
    category_zh = category_label_zh(category_key)
    use_uncertain = category_key == "unknown" or category_confidence < uncertain_threshold
    prefix = "\u7591\u4f3c" if use_uncertain and category_key != "unknown" else ""
    subject = category_subject_zh(category_key)
    return f"{subject}\u4f4d\u4e8e{location_label_zh}\u7684{size_label_zh}{prefix}{category_zh}\u3002"


def compose_clip_text(
    category_key: str,
    location_key: str,
    size_key: str,
    category_confidence: float,
    uncertain_threshold: float = 0.5,
) -> str:
    category_en = category_label_en(category_key)
    location_en = LOCATION_KEY_TO_EN[location_key]
    size_en = SIZE_KEY_TO_EN[size_key]
    maybe_prefix = "possibly " if category_key == "unknown" or category_confidence < uncertain_threshold else ""
    article = "an" if category_en[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"A camouflaged {size_en} target, {maybe_prefix}{article} {category_en}, located near the {location_en} of the image."


def compose_training_text(
    category_key: str,
    location_key: str,
    size_key: str,
    category_confidence: float,
    evidence: str,
    uncertain_threshold: float = 0.5,
) -> str:
    category_en = category_label_en(category_key)
    location_en = LOCATION_KEY_TO_EN[location_key]
    size_en = SIZE_KEY_TO_EN[size_key]
    maybe_prefix = "possibly " if category_key == "unknown" or category_confidence < uncertain_threshold else ""
    article = "an" if category_en[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    evidence = " ".join(str(evidence or "").strip().split())
    sentence = f"A camouflaged {size_en} target, {maybe_prefix}{article} {category_en}, located near the {location_en} of the image"
    if evidence:
        sentence += f", with {evidence}"
    return sentence.rstrip(" ,.") + "."
