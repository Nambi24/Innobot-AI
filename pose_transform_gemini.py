#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Editorial Pose Transformation Pipeline
======================================

1. Process all images in input/ in parallel
2. Interactive modes:
   - upscale: Nano Banana 2 only (fixed upscale prompt, no LLM) @ 4K → output_upscale_pose/*_1.jpg
   - pose: Gemini 3 Flash → weird pose prompt, then Nano Banana 2 edit @ 2K → output_upscale_pose/*_2.jpg
   - upscale_pose_bg: ask donor folder → transfer BG+pose from THAT folder (not input),
       then weird pose on _1 — both saved in output_upscale_pose_bg as _1 + _2 @ 4K
   - both: like Model_change scenario — per input produce two images @ 4K
       _1 = upscale + BG/pose transfer (donor folder)
       _2 = weird pose change on _1 (mode 2, outfit/bg locked from _1)
   - upscale_pose: per input produce two images @ 4K in the SAME folder as modes 1+2
       _1 = upscale only (mode 1)
       _2 = weird pose change on _1 (mode 2, outfit/bg locked from _1)

Cross-image transfer (upscale_pose_bg / both image_1):
  BG + pose donors come from a user-provided donor folder (not the subject input folder).
  Each donor file is used at most once across the batch when enough images exist.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import replicate
from replicate.exceptions import ModelError
from replicate.helpers import transform_output
from PIL import Image as PILImage

def get_replicate_api_key() -> Optional[str]:
    """Load Replicate API token from environment (never hardcode)."""
    token = (
        os.getenv("REPLICATE_API_TOKEN")
        or os.getenv("REPLICATE_API_KEY")
        or ""
    ).strip()
    return token or None


REPLICATE_API_KEY = get_replicate_api_key()

PIPELINE_MODES = ("upscale", "pose", "upscale_pose_bg", "both", "upscale_pose")
DEFAULT_MODE = "pose"
REPLICATE_RATE_LIMIT = 40
REPLICATE_RATE_PERIOD_SECONDS = 60.0
REPLICATE_MAX_RETRIES = 15
REPLICATE_MAX_CONCURRENT = 40
REPLICATE_RETRY_BACKOFF_BASE = 2.0
REPLICATE_POLL_INTERVAL = 2.0
REPLICATE_HTTP_CONNECT_TIMEOUT = 30.0
REPLICATE_HTTP_READ_TIMEOUT = 120.0
REPLICATE_HTTP_WRITE_TIMEOUT = 300.0
REPLICATE_HTTP_POOL_TIMEOUT = 30.0
REPLICATE_DOWNLOAD_READ_TIMEOUT = 300.0
DEFAULT_UPSCALE_WORKERS = 40
DEFAULT_MAX_WORKERS = 40
DESCRIPTIONS_JSON = "pose_descriptions.json"
POSE_BG_DESCRIPTIONS_JSON = "pose_bg_descriptions.json"
BOTH_DESCRIPTIONS_JSON = "both_descriptions.json"
UPSCALE_POSE_DESCRIPTIONS_JSON = "upscale_pose_descriptions.json"

CAMERA_PROFILE: Dict[str, Any] = {
    "make": "NIKON CORPORATION",
    "model": "NIKON D7500",
    "max_aperture": (41, 10),
    "metering_mode": 5,
}

EXPOSURE_TIMES = ((1, 125), (1, 160), (1, 200), (1, 250), (1, 320), (1, 400))
FNUMBERS = ((40, 10), (45, 10), (50, 10), (56, 10), (63, 10))
ISO_VALUES = (200, 250, 320, 400, 500, 640, 800)
FOCAL_LENGTHS = ((240, 10), (280, 10), (350, 10), (500, 10), (700, 10))
FOCAL_35MM = (36, 42, 52, 75, 105)
FLASH_VALUES = (16, 24)

TIME_OF_DAY_HOUR_RANGES = {
    "late_night": (0, 4),
    "night": (20, 23),
    "dawn": (5, 7),
    "morning": (7, 10),
    "midday": (11, 14),
    "afternoon": (14, 17),
    "evening": (17, 20),
    "indoor": (10, 16),
}

NIGHT_PROMPT_PATTERNS = (
    r"\bat night\b",
    r"\bnighttime\b",
    r"\bnight\b",
    r"\bmidnight\b",
    r"\bafter dark\b",
    r"\bdark (street|alley|road|lane|corner|outdoor)\b",
    r"\bstreet\s*lamp\b",
    r"\bstreetlight\b",
    r"\bmoonlight\b",
    r"\bunder the stars\b",
)

EVENING_PROMPT_PATTERNS = (
    r"\bevening\b",
    r"\bdusk\b",
    r"\bsunset\b",
    r"\btwilight\b",
    r"\bgolden hour\b",
)

DAWN_PROMPT_PATTERNS = (
    r"\bdawn\b",
    r"\bsunrise\b",
    r"\bearly morning\b",
)

FOREIGN_BACKGROUND_PATTERNS = (
    r"\beuropean\b",
    r"\blondon\b",
    r"\bparis\b",
    r"\bitaly\b",
    r"\bitalian\b",
    r"\bfrench\b",
    r"\bbritish\b",
    r"\buk\b",
    r"\bnew\s*york\b",
    r"\bmanhattan\b",
    r"\bcobblestone\b",
    r"\bvictorian\b",
    r"\bgeorgian\b",
    r"\btourist\s+street\b",
    r"\bwestern\s+(street|cafe|alley)\b",
    r"\bheritage\s+shopfront\b",
)

UPSCALE_PROMPT = (
    "Upscale and enhance ONLY the actual photograph to 4K. "
    "If the input is a phone screenshot, gallery preview, or share-sheet capture with black letterbox bars, "
    "status bar, navigation bar, time/battery icons, 'Screenshot saved' toast, notification banners, "
    "or any phone UI overlay: ignore ALL of that chrome completely. "
    "Fill the entire output frame with only the real photo content — the person and their real environment. "
    "Do not keep black bars, phone UI, status icons, watermarks from the screenshot chrome, or empty padding. "
    "Output a clean full-bleed photorealistic image at 2:3 portrait aspect ratio. "
    "super high resolution, open pores, freckles, visible skin texture, "
    "natural face imperfections, sharpness — photorealistic, not AI-smooth"
)

SYSTEM_PROMPT_POSE_BG = """You are directing a CROSS-IMAGE POSE/BG TRANSFER edit with 4K upscale quality.

You receive THREE images with fixed roles:
- IMAGE 1 (SUBJECT): person identity to KEEP — face, skin, hair, markings, body proportions, photo realism
- IMAGE 2 (BACKGROUND DONOR): copy this REAL environment as the new background — surfaces, architecture, depth, props, lighting character of that place
- IMAGE 3 (POSE DONOR): copy this REAL body pose — legs, hips, torso, shoulders, arms, hands, head orientation

Your job: put the SUBJECT person into the BACKGROUND DONOR's scene, matching the POSE DONOR's body configuration, in a NEW outfit that fits the borrowed scene. Do NOT invent a generic kitchen/courtyard/terrace — use IMAGE 2's actual place. Do NOT invent a generic candid pose — use IMAGE 3's actual pose (adapted so it fits IMAGE 2's surfaces).

IDENTITY LOCK (from IMAGE 1 only):
- Same person: face structure, skin tone, hair, markings/bindi if present, body proportions
- Preserve IMAGE 1 photo realism: color grading feel, tonal contrast, texture, natural grain — NOT AI polish

BACKGROUND TRANSFER (from IMAGE 2 — NON-NEGOTIABLE):
- Describe and lock the EXACT environment visible in IMAGE 2
- Architecture, materials, furniture, plants, floor, walls, depth, ambient light of IMAGE 2
- Do NOT replace it with a different invented location
- Do NOT keep IMAGE 1's background
- EMPTY BACKGROUND PEOPLE LOCK — CRITICAL: the final image must contain ONLY the subject person from IMAGE 1.
  If IMAGE 2 has other people, faces, crowds, bystanders, pedestrians, or anyone else — REMOVE them completely.
  Keep only empty environment: architecture, props, plants, vehicles without occupants if needed.
  No extra humans, no silhouettes, no distant figures, no reflections of other people.

POSE TRANSFER (from IMAGE 3 — NON-NEGOTIABLE):
- Match IMAGE 3's full-body pose STRUCTURE as closely as anatomy allows — same sit/stand/lean/crouch intent, limb angles, torso twist, head direction, contact points
- Adapt contact points to IMAGE 2's real surfaces (if IMAGE 3 sits on a chair and IMAGE 2 has a bench/step, use that)
- Do NOT keep IMAGE 1's pose
- Do NOT invent a different "candid walk / lean on counter" default

CROSS-GENDER / BODY ADAPTATION — CRITICAL:
- Always keep IMAGE 1's gender presentation, body shape, and proportions — NEVER copy the pose donor's sexed anatomy
- If IMAGE 1 and IMAGE 3 differ in gender presentation (man↔woman or similar), transfer only the pose geometry (angles, contacts, silhouette intent), then REGENERATE the body correctly for the SUBJECT:
  - Correct shoulder/hip width, chest/torso shape, limb proportions for IMAGE 1
  - Natural stance and weight distribution for that person's body
  - Outfit drapes on IMAGE 1's body — never force donor body silhouette under subject clothes
- Do NOT masculinize a woman or feminize a man
- Do NOT blend faces or bodies between subject and pose donor
- Result must look like IMAGE 1's person holding that pose, not IMAGE 3's person with a face swap

OUTFIT:
- Completely different from IMAGE 1 — everyday clothing that fits IMAGE 2's scene and the person
- Do not copy clothing from IMAGE 2 or IMAGE 3 unless it clearly fits; invent a fitting everyday outfit

INDIAN / REALISM CONTEXT:
- Prefer staying in the cultural world of the subject; if IMAGE 2 is clearly foreign, still transfer it faithfully but keep the person identity from IMAGE 1
- Output must look like a real photo, not AI/CGI/beauty-filter polish

Return ONLY JSON:
{
  "subject_summary": "who is in IMAGE 1 — age, gender presentation, key identity traits to preserve exactly",
  "input_realism_lock": "what to preserve from IMAGE 1 photo quality — color grading, contrast, grain, sharpness, lighting character",
  "context_lock": "cultural/everyday context — stay coherent with subject + borrowed scene",
  "identity_lock": ["face structure", "skin tone", "hair", "bindi/markings if present", "body proportions"],
  "bg_donor_reading": "exact environment from IMAGE 2 — surfaces, objects, architecture, depth, light (note any people to REMOVE)",
  "pose_donor_reading": "exact full-body pose from IMAGE 3 — legs, hips, torso, arms, head, contacts",
  "gender_pose_adapt": "same-gender or cross-gender — how pose geometry was adapted to IMAGE 1's body without copying donor anatomy",
  "new_outfit": "exact new everyday outfit — must differ from IMAGE 1, fits IMAGE 2 scene",
  "new_background": "IMAGE 2 environment with ZERO other people — empty scene, subject only",
  "no_extra_people": "confirm background has no bystanders/crowds/other faces — only IMAGE 1 subject",
  "lighting_description": "lighting from IMAGE 2 scene, still compatible with IMAGE 1 photo family",
  "camera_description": "candid phone/camera framing — full body visible, same realism as IMAGE 1",
  "current_pose": "one line — IMAGE 1 pose being replaced",
  "new_pose": "one line — IMAGE 3 pose geometry on IMAGE 1's correctly gendered body in IMAGE 2 scene",
  "full_body_breakdown": "legs, hips, torso, shoulders, arms, hands, head — contacts with IMAGE 2 surfaces; proportions match IMAGE 1",
  "shot_description": "one line, 5-10 words — e.g. 'Blue kurta, donor courtyard, seated pose'",
  "prompt": "full image-generation prompt for cross-image pose/bg transfer + 4K upscale quality"
}

HOW TO THINK:
Step 1: Lock identity + realism from IMAGE 1.
Step 2: Read IMAGE 2 background in concrete detail — this IS the new background. Strip out any other people.
Step 3: Read IMAGE 3 pose in concrete full-body detail — transfer pose GEOMETRY only.
Step 4: If cross-gender (or different body type), adapt that geometry onto IMAGE 1's real body — correct proportions, no donor anatomy bleed.
Step 5: Invent a NEW outfit that fits IMAGE 2 and the person (not IMAGE 1's clothes).
Step 6: Adapt contacts onto IMAGE 2 surfaces without inventing a different pose.
Step 7: Full person visible. Empty background (no other people). 4K detail. Not AI-looking.

FRAMING:
- Full person completely visible — no cropped head, limbs, hands, or feet. Leave safe margin.
- ONLY the subject in frame — no crowds, bystanders, extra faces, distant people, or reflections of other people.

Structure the "prompt" field as one flowing paragraph:
[IMAGE 2 background with specific materials, EMPTY of other people] + [IMAGE 1 person identity] + [new outfit] + [IMAGE 3 pose adapted to IMAGE 2] + [IMAGE 2 lighting + IMAGE 1 photo family] + [candid camera] + [4K: open pores, freckles, skin texture, fabric detail] + [negative: other people in background, bystanders, crowds, extra faces, inventing different bg, inventing different pose, same outfit as IMAGE 1, cropped limbs, AI look, CGI, plastic skin, beauty filter, HDR]

shot_description must be 5-10 words, plain English.

No markdown. JSON only."""

SYSTEM_PROMPT_POSE_BG_INVENT = """You are directing a POSE/BG CHANGE edit with 4K upscale quality (single-image fallback — no donor photos available).

Keep the SAME person identity from the reference. Invent a NEW outfit, NEW natural candid pose, and NEW everyday background that fits the person's cultural world. Prefer concrete unique locations — avoid repeating the same kitchen/courtyard/terrace default.

NO OTHER PEOPLE: background must be empty of bystanders, crowds, extra faces, or distant figures — only the subject.

Return ONLY JSON with keys:
subject_summary, input_realism_lock, context_lock, identity_lock, new_outfit, new_background,
lighting_description, camera_description, current_pose, new_pose, full_body_breakdown,
no_extra_people, shot_description (5-10 words), prompt.

No markdown. JSON only."""

SYSTEM_PROMPT = """You are directing an ACCEPTABLY WEIRD portrait pose edit in a locked scene. Your ONLY job: replace the original pose with an unusual but believable, safe, scene-specific FULL-BODY configuration — same person, outfit, background, lighting, camera.

CRITICAL: Previous outputs became TOO EXTREME, then TOO NORMAL. Do NOT create stunt, fall, climbing, hanging, railing-perching, or physically alarming poses. Also do NOT create normal stretch, workout, prayer, thinking, leaning, tourist prop, Instagram wall-art, pointing-at-decor, or casual portrait poses. The target is controlled oddness: clearly weird, still safe.

You receive ONE reference image. First READ the scene like a director: what objects, surfaces, plants, furniture, architecture, and space exist? Then invent an unusual full-body pose that uses THIS exact environment in a quirky but acceptable way — while staying fully safe, grounded, and logically possible.

ACCEPTABLE WEIRD: awkward, playful, committed full-body poses a real person could hold for a photo without injury, fear, embarrassment, or falling. No stunts, no danger, no "about to die" energy, no public-disturbance energy.

WEIRDNESS TARGET — NON-NEGOTIABLE:
- Internally invent FIVE different pose ideas. Reject anything normal, fitness-like, devotional/prayer-like, fashion-like, tourist-photo-like, pointing-at-prop, or stunt-like. Output the strangest remaining safe option.
- weirdness_score in JSON should be 7 or 8 only. If it is 9-10, reduce intensity. If it is below 7, make it more unusual.
- The pose must pass the REAL PHOTO TEST: someone could actually take this as a quirky portrait without anyone looking endangered.
- The pose must pass the SAFETY READ TEST: viewer immediately understands the subject is stable and supported.
- The pose must pass the ACCEPTABILITY TEST: odd enough that it could not be mistaken for normal stretching, praying, exercising, leaning, pointing at decor, tourist posing, or casual posing; not so odd that it looks like a stunt, injury, or emergency.

DEFAULT TO STRONG BUT SAFE WEIRD. Mild, "slightly different" poses are too simple, but extreme body-hanging, wall-climbing, railing-perching, or stunt-like poses are too much.
If the new pose could still pass as a normal passport-style portrait, it is TOO SIMPLE.
If the new pose is just a small variation of the reference (same basic squat/stand/sit shape), it is TOO SIMPLE — redo completely.
If weird_action sounds too extreme ("hanging", "climbing over", "balancing on rail", "perching on edge", "mid-fall") — UNSAFE. Redo milder.
Use acceptable verbs like fold, crouch sideways, sit backward, brace with elbow, tuck chin, twist torso, hug object awkwardly, press shoulder, look under, peek around, cross arms unevenly, kneel asymmetrically, step sideways. Avoid pointing as the main action.
Change the silhouette clearly from the reference — but keep it plausible, stable, and photo-friendly.

IMPORTANT LIMB RULE: hands and legs CAN do weird things, but keep them moderate. Raised knees, reaching, grabbing, hooking, bracing, and dangling arms are allowed when the whole body changes. Avoid high kicks, wall-bracing splits, rail-balancing, hanging, or martial-arts-like shapes.

This is NOT "reach farther" or "touch the same thing with a longer arm." That is banned.
This is NOT "lean a little", "shift weight", "one hand on hip", "slight head tilt", or "casual sit/stand variant."
This IS a clear pose swap: legs, hips, torso, shoulders, arms, and head all repositioned into one awkward, scene-driven action.
The viewer should think "that's an odd pose" — not "that person is in danger."

Return ONLY JSON:
{
  "subject_summary": "who is in the image — age, gender presentation, key identity traits to preserve exactly",
  "scene_reading": "what you see in the background — list specific props, surfaces, plants, architecture, floor/wall features the body can interact with",
  "environment_element": "the ONE specific thing in the scene driving the weird action (tree trunk, doorframe, bench, railing, step, pillar, table edge, etc.)",
  "weird_action": "verb phrase — acceptably weird action with that element (e.g. 'crouching sideways under the wall art with crossed uneven arms', 'turning torso away while peeking under the decoration', 'folding chest-first onto a chair back while hands hang') — avoid pointing at decor, tourist wall-photo poses, and dangerous verbs like hang, climb over, perch on edge, mid-fall",
  "weirdness_score": "7 or 8 only — rate how unusual but acceptable the pose is; if 9-10, reduce intensity; if below 7, make it more interesting",
  "outfit_lock": "exact outfit description — must stay identical",
  "identity_lock": ["face structure", "skin tone", "hair", "bindi/markings if present", "body proportions"],
  "background_lock": "exact background description to preserve — every visible element unchanged",
  "lighting_lock": "exact lighting in the reference",
  "camera_lock": "exact camera framing — angle, distance, crop, perspective",
  "current_pose": "one line — stiff/boring pose to completely replace",
  "pose_category": "short label — use moderate verbs: fold, crouch, sideways sit, awkward reach, brace, tuck, twist, shoulder press, peek, kneel — NOT hang, perch, climb, fall",
  "new_pose": "one line — the new weird pose as a complete body configuration",
  "full_body_breakdown": "describe legs, hips, torso, shoulders, arms, hands, head — how each part is positioned for the weird action; specify what touches floor/wall/object for EACH limb",
  "shot_description": "one line, 5-10 words only — what pose change happened (e.g. 'Sideways doorway crouch, arms crossed awkwardly')",
  "prompt": "full image-generation prompt for pose-only transformation"
}

MANDATORY POSE RULES (ALL must be true — if any fails, revise the pose):
1. CLEAR CHANGE: pose must visibly differ from the reference through body level, torso angle, arm placement, leg placement, and head direction; a normal stretch, pointing pose, tourist prop pose, or fitness pose does NOT count.
2. MODERATE LIMBS: legs and hands may be raised, hooked, grabbed, braced, crossed, or stretched, but avoid high kicks, splits, hanging limbs over ledges, or one-leg balancing.
3. ASYMMETRY: create an uneven, quirky shape through torso angle, shoulder level, arm placement, head direction, leg placement, and environmental contact. At least THREE of these must change strongly, not just one arm.
4. CONTACT / BALANCE: subject must look stable. At least two body parts should clearly touch or use floor, wall, furniture, railing, tree, or another stable object.
5. ENVIRONMENT USE: at least one real object/surface in the scene should influence the pose, but do not simply point at it, present it, or pose under it like a tourist.
6. BODY AXIS: rotate torso/hips enough to differ from reference, usually 35-60°, not a tiny turn and not a contortion.
7. NO EXTREME GEOMETRY: banned silhouettes include — wall-split, railing perch, hanging over wall, martial-arts kick, dancer extension, mid-fall, body draped over dangerous edge.

HOW TO THINK — SCENE → ACCEPTABLY WEIRD PERSON → FULL BODY POSE:
Step 1: Inventory the scene (tree, door, steps, chair, wall, gate, plant pot, window ledge, etc.)
Step 2: Ask: "What would a quirky person do here?" — unusual but believable, uses the object in a slightly wrong way
Step 3: Build the ENTIRE body around that action — not just an arm reaching
Step 4: Compare to reference — if it is too similar, add one more body-level or torso-angle change
Step 5: Pick the middle of your three internal candidates: not normal, not extreme

SCENE-DRIVEN EXAMPLES — ACCEPTABLY WEIRD BUT SAFE (adapt to what's actually visible — never copy blindly):
- Tree / plant / pole → crouching beside it with one shoulder pressed to trunk, both hands placed at mismatched heights, head turned away, legs crossed awkwardly but stable
- Doorframe / pillar → sideways doorway crouch with shoulder on one side, one hand high and one hand low on the frame, knees angled differently; or seated low in doorway with torso twisted and arms placed oddly
- Bench / chair / step → sitting backward on the chair with chest leaning onto the backrest and arms hanging; sitting sideways across the seat with elbows at mismatched heights; or crouching beside it with one elbow and one hand placed awkwardly on the seat
- Wall / wall art / neon wings / decor → do NOT point at the art or pose like wings. Instead crouch sideways below it with one shoulder near the wall and arms crossed unevenly; sit backward under it with torso twisted away and head looking down; or half-crouch with forehead near the wall and feet placed oddly
- Railing / fence (near drops, dams, balconies) → crouch safely on the walkway side, one hand gripping railing low, opposite hand on knee, head turned away, knees bent unevenly; NEVER climb, perch, lean out, hang over, or put body weight over the drop
- Table / counter edge → crouched beside it with chin or forearm lightly on edge, legs uneven and hands at mismatched heights; or standing close with torso folded sideways and both hands hanging safely
- Stairs / floor → seated across one or two steps with knees uneven and torso twisted away; crouched sideways on a step with one hand on wall and the other on knee; or sitting on floor with one leg extended, one bent, head tilted down
- Open floor → asymmetric low crouch with arms crossed unevenly, sideways seated pose with torso twisted, uneven kneel with hands on different surfaces, or folded-forward seated pose — strange but still something a person could comfortably hold

MODEL FAILURE MODES — YOU KEEP OUTPUTTING THESE; STOP:
- Repeating the same leg-up / knee-up / high-kick shape on every image instead of using the actual scene (BAD)
- Squat/crouch near object with head or hand resting on it (lazy default — BANNED)
- Casual lean on wall/railing/door with one foot forward (catalog pose — BANNED)
- Standing with hand on hip, arms crossed, or one arm raised (portrait pose — BANNED)
- Gentle sit with knees together or polite cross-legged (BANNED)
- One-arm reach while body stays upright (arm-only change — BANNED; reach is allowed only when hips, torso, legs, and other arm also change)
- "Quirky" head tilt with same leg placement as reference (BANNED)
- Same standing/sitting height as reference with minor limb tweak (BANNED)
- Fashion/editorial "dynamic" stance — weight on back foot, front foot pointed (BANNED)
- Crouching facing camera with elbows on knees (stock photo — BANNED)
- "Interacting playfully" with prop — touching, holding, pointing (BANNED unless entire body is rearranged around it)
- Repeating the same arm motif across images, especially hidden-arm placement (BANNED)
- Tourist / Instagram prop pose: seated under wall art, one arm pointing up, legs spread normally, smiling/cool posture (BANNED)
- Angel-wing wall-art pose: sitting or standing centered under wings, pointing at crown/wings, pretending the wings belong to the subject (BANNED)
- Any pose where the main idea is pointing at a visible decoration/sign/object (BANNED)
- Normal stretch/workout pose: one foot on wall, hands clasped, torso leaning forward (BANNED)
- Prayer/thinking pose: clasped hands near wall or face, head bowed, one knee raised (BANNED)
- Standard lunge, calf stretch, hamstring stretch, yoga warmup, or gym pose (BANNED)
- Railing/wall stunt poses, wall-splits, hanging torsos, perched bodies, mid-fall energy, or anything that looks unsafe (TOO EXTREME)

TOO-SIMPLE POSES — REJECT AND REDO (these fail the assignment):
- Any pose where standing/sitting height matches the reference
- Any pose describable in under 8 words without sounding unusual
- Any pose a yoga instructor or model would demonstrate comfortably
- Any pose a person would do for stretching, exercise, prayer, or resting
- Any pose a person would do for a normal tourist photo, Instagram wall photo, or prop interaction
- Any pose that could appear on a clothing website

TOO-EXTREME POSES — REDUCE INTENSITY (these also fail):
- Body appears to hang over a wall, railing, ledge, balcony, or rooftop edge
- Subject appears stuck, injured, falling, climbing, or trapped
- Pose depends on a high kick, split, martial arts move, dance extension, or one-leg wall brace
- Body weight appears supported by a railing, window, door, or unstable object

SAFETY GUARDRAILS (MANDATORY — weird poses must pass ALL of these):
- ALL body weight stays on stable, safe surfaces: floor, walkway, step tread, bench seat, ground — never suspended over empty space
- If railings, ledges, balconies, dams, cliffs, water, or drops are visible: subject stays on the SAFE side of the barrier; no leaning out, no dangling legs over the edge, no torso hanging over the rail
- Pose must be logically possible: a healthy adult could hold it briefly for a photo without falling, slipping, or injury
- No fall/collapse/death/stunt energy: no mid-fall, no slipping, no losing balance off a ledge, no "about to go over"
- No climbing over barriers, no sitting on top of railings over a drop, no perching on unsafe ledges
- Feet (or seated hips) must maintain clear stable contact with a solid surface unless lying fully on a flat safe surface

ACCEPTABLE WEIRD PERSON ENERGY:
- Quirky, awkward, playful, and committed — but still believable as a safe portrait
- The body uses the environment lightly or moderately: shoulder to wall, hand on railing, seated sideways, crouched near object, torso folded over chair, head peeking around edge
- Silhouette must look like a different pose, not the same standing/sitting geometry with a hand moved
- Center of mass can shift, but subject must look stable and calm
- Think: an odd photo direction, not a prank, stunt, injury, or emergency

FULL-BODY REQUIREMENTS (ALL MANDATORY):
1. Legs: clearly new placement — crossed, one bent, one extended, small raised knee, uneven kneel, sideways sit, feet at odd but stable angles — NOT same stance
2. Hips + spine: clear angle change vs reference — twisted, folded, crouched, side-sat, or tilted, but not contorted
3. Torso: bent, angled, folded, or turned — not upright mirror of original
4. Arms: both involved in varied balance, grip, cross, hang, or brace placements — never the same repeated motif
5. Head: tucked, craned, dropped, tilted, peeking, or lightly pressed against a surface — within natural neck range
6. At least TWO body parts must interact with the environment, and the interaction must look safe
7. Height level should change when possible: standing → crouch, sit, kneel, or angled stand; sitting → sideways sit, low crouch, folded sit, or awkward stand
8. Body axis should rotate 35-60° vs reference — enough to differ, not a contortion

ANATOMY GUARDRAILS (weird but healthy body):
- NO twisted head/neck past natural range, NO backwards-bent wrists, NO twisted/knotted legs, NO feet pointing wrong way
- NO contortion, crab-bridge, folded spine, disability/injury/deformity mimicry
- Joints look normal — weirdness is in the POSE and ACTION, not broken anatomy

ABSOLUTELY BANNED (these are TOO SIMPLE or UNSAFE — never output these):
- Arm-only changes: reaching farther, pointing, touching a prop with one hand while body stays the same
- Subtle tweaks: head tilt only, weight shift, one foot slightly forward, hand on hip, crossed arms, "dynamic" model stand
- Polite variants: casual lean on wall, gentle sit, soft knee bend, relaxed standing with smile
- Cute, catalog, passport-photo, influencer, editorial fashion, natural candid everyday poses
- Karate kick form, martial arts guard, spiderman crouch, superhero landing
- Upside-down / full inversion
- Poses where only the arms or head moved but legs/hips stayed in the same basic standing/sitting shape
- DEATH SETUPS: leaning over railings/ledges toward drops, dangling over edges, body weight on wrong side of a barrier, hanging off balconies/cliffs/dams, sitting on railing over water/void, stunt/fall/slip poses, anything implying imminent falling

WEIRDNESS SELF-TEST (must pass ALL before output):
- weirdness_score is 7 or 8? → if below 7, add interest; if above 8, reduce intensity
- Is the main visual idea only "one leg is up" while the rest of the body stayed normal? → TOO SIMPLE, redo with full-body weirdness
- Is the main visual idea only "one arm points at decor" while the rest of the body is normal? → TOO SIMPLE, redo with full-body weirdness
- Could this pose appear in a normal family photo, tourist photo, or Instagram wall-art pose? → TOO SIMPLE, add clear awkward torso, head, arm, and leg changes
- Could this pose appear on a fashion website? → probably TOO SIMPLE, add awkwardness
- Is the body silhouette basically the same as reference? → TOO SIMPLE, add a torso/level/limb change
- Does the pose look like a stunt, fall, injury, or emergency? → TOO EXTREME, reduce intensity
- Does it involve climbing, perching, hanging, or leaning body weight over a railing/wall/ledge? → UNSAFE, redo
- Does it look like a minor variation of the reference pose? → TOO SIMPLE, add a torso/level/limb change
- Could this pose cause a fall or look like a death/stunt setup? → UNSAFE, redo with feet on safe ground
- Is all body weight clearly on stable safe surfaces? → must be YES

MUST CHANGE:
- Clear full-body pose driven by a scene-specific quirky action. Same room, same person, acceptably weird body geometry.

MUST PRESERVE:
- Same person, face, skin, hair, markings, healthy proportions
- Same outfit, background, lighting, camera framing
- Photorealistic skin and fabric

Structure the "prompt" field as one flowing paragraph:
[Background lock] + [acceptably weird but SAFE action with environment element, stable body support visible] + [full body breakdown — legs, hips, torso, arms, head, what touches the scene] + [same camera + lighting] + [identity + outfit lock] + [negative: subtle tweak, arm-only reach, pointing at decor, tourist prop pose, Instagram wall-art pose, same stance, catalog, influencer, editorial, twisted anatomy, martial arts, dancer kick, hanging over wall, railing perch, climbing, leaning over drop, dangling over edge, death setup, fall pose]

shot_description must be 5-10 words, plain English, describing the acceptably weird pose change only.

No markdown. JSON only."""

SYSTEM_PROMPT_POSE_TRANSFER = """You are directing a CROSS-IMAGE POSE TRANSFER edit in a locked scene.

You receive TWO images with fixed roles:
- IMAGE 1 (SUBJECT): keep this person, outfit, background, lighting, and camera framing
- IMAGE 2 (POSE DONOR): copy this REAL full-body pose onto the subject

Your ONLY job: replace IMAGE 1's pose with IMAGE 2's pose, adapted so contact points use IMAGE 1's real environment (floor, wall, furniture, railing, steps, etc.). Do NOT invent a different weird pose. Do NOT keep IMAGE 1's pose.

IDENTITY / SCENE LOCK (IMAGE 1):
- Same person: face, skin, hair, markings, body proportions
- Same outfit, background, lighting, camera framing
- Photorealistic — not AI polish

POSE TRANSFER (IMAGE 2 — NON-NEGOTIABLE):
- Match IMAGE 2's pose STRUCTURE — legs/hips/torso/shoulders/arms/head angles and contact intent
- Adapt contacts onto IMAGE 1 surfaces (if IMAGE 2 sits and IMAGE 1 has a chair/step/floor, use that)
- Pose must clearly differ from IMAGE 1's original pose
- Keep the pose safe and stable — no hanging over drops, climbing, mid-fall

CROSS-GENDER / BODY ADAPTATION — CRITICAL:
- Keep IMAGE 1's gender presentation and body proportions always
- If IMAGE 1 and IMAGE 2 differ in gender presentation, transfer pose geometry only, then regenerate the body correctly for IMAGE 1
- Do NOT masculinize a woman or feminize a man; do NOT copy donor chest/hip/shoulder shape
- Result = IMAGE 1's person in IMAGE 2's pose — never a face-swap onto the donor body

Return ONLY JSON:
{
  "subject_summary": "who is in IMAGE 1 — identity traits to preserve",
  "scene_reading": "IMAGE 1 background — props/surfaces the transferred pose can contact",
  "pose_donor_reading": "exact full-body pose from IMAGE 2",
  "gender_pose_adapt": "same-gender or cross-gender — how pose geometry was adapted to IMAGE 1's body",
  "environment_element": "the IMAGE 1 surface/object used for contact adaptation",
  "weird_action": "short phrase for the transferred pose adapted into IMAGE 1",
  "weirdness_score": "7 or 8 — how different from IMAGE 1 original pose",
  "outfit_lock": "exact IMAGE 1 outfit — must stay identical",
  "identity_lock": ["face structure", "skin tone", "hair", "bindi/markings if present", "body proportions"],
  "background_lock": "exact IMAGE 1 background — unchanged",
  "lighting_lock": "exact IMAGE 1 lighting",
  "camera_lock": "exact IMAGE 1 camera framing",
  "current_pose": "one line — IMAGE 1 pose being replaced",
  "pose_category": "transfer",
  "new_pose": "one line — IMAGE 2 pose geometry on IMAGE 1's correctly gendered body",
  "full_body_breakdown": "legs, hips, torso, shoulders, arms, hands, head — contacts with IMAGE 1 surfaces; proportions match IMAGE 1",
  "shot_description": "one line, 5-10 words — e.g. 'Donor seated twist on courtyard steps'",
  "prompt": "full image-generation prompt for pose-only transfer"
}

HOW TO THINK:
Step 1: Lock person/outfit/bg/lighting/camera from IMAGE 1.
Step 2: Read IMAGE 2 pose geometry in full-body detail.
Step 3: Map that pose onto IMAGE 1 environment contacts, adapting for IMAGE 1's gender/body if cross-gender.
Step 4: Ensure clear difference from IMAGE 1 original pose; anatomically clean for IMAGE 1; full body visible.

Structure the "prompt" field as one flowing paragraph:
[IMAGE 1 background lock] + [IMAGE 2 pose adapted to IMAGE 1 surfaces] + [full body breakdown] + [same camera + lighting] + [identity + outfit lock] + [negative: inventing different pose, keeping IMAGE 1 pose, changing outfit/bg, cropped limbs, AI look]

shot_description must be 5-10 words, plain English.

No markdown. JSON only."""


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def build_donor_pairs(
    subject_paths: List[str],
    donor_paths: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Pair each subject with BG + pose donors from a separate donor pool.

    - Never uses a subject file as its own donor (resolved-path match).
    - Prefers unique donors across the whole batch (each donor used at most once).
    - For each subject, bg_source != pose_source when possible.
    """
    subjects = list(subject_paths)
    pairs: Dict[str, Dict[str, Optional[str]]] = {p: {"bg_source": None, "pose_source": None} for p in subjects}
    if not subjects:
        return pairs

    subject_keys = {_norm_path(p) for p in subjects}
    raw_donors = list(donor_paths) if donor_paths is not None else list(subjects)

    # Unique donors by resolved path; exclude anything that is also a subject file
    seen: set = set()
    donors: List[str] = []
    for path in raw_donors:
        key = _norm_path(path)
        if key in subject_keys or key in seen:
            continue
        seen.add(key)
        donors.append(path)

    # If donor pool empty (e.g. same folder / all overlap), fall back to other subjects
    if not donors and len(subjects) >= 2:
        donors = list(subjects)

    if len(donors) < 2:
        return pairs

    random.shuffle(donors)
    pool = list(donors)
    reused = False

    def take_donor(exclude: Optional[set] = None) -> Optional[str]:
        nonlocal pool, reused
        exclude = exclude or set()
        for i, candidate in enumerate(pool):
            if _norm_path(candidate) in exclude:
                continue
            return pool.pop(i)
        # Pool exhausted — reshuffle full donor list for limited reuse
        reused = True
        pool = [d for d in donors if _norm_path(d) not in exclude]
        random.shuffle(pool)
        if not pool:
            pool = list(donors)
            random.shuffle(pool)
        if not pool:
            return None
        return pool.pop(0)

    for subject in subjects:
        subject_key = _norm_path(subject)
        # When falling back to subject-as-donor pool, never pick self
        bg_src = take_donor(exclude={subject_key})
        pose_exclude = {subject_key}
        if bg_src:
            pose_exclude.add(_norm_path(bg_src))
        pose_src = take_donor(exclude=pose_exclude)
        pairs[subject] = {"bg_source": bg_src, "pose_source": pose_src}

    if reused:
        print(
            "WARNING: Not enough unique donor images for the batch — "
            "some donor files were reused. Add more images to the donor folder to avoid reuse."
        )
    return pairs


def select_donor_folder(input_folder: Path, script_dir: Path) -> Path:
    """Ask for a donor folder that is not the subject input folder."""
    env_value = os.getenv("DONOR_FOLDER", "").strip().strip('"')
    if env_value:
        donor = Path(env_value).expanduser()
        if not donor.is_absolute():
            donor = (script_dir / donor).resolve()
        else:
            donor = donor.resolve()
        if donor.is_dir() and _norm_path(str(donor)) != _norm_path(str(input_folder.resolve())):
            return donor
        print(f"DONOR_FOLDER invalid or same as input ({donor}) — asking interactively.")

    if not sys.stdin.isatty():
        raise ValueError(
            "Donor folder required for pose/bg transfer. "
            "Set DONOR_FOLDER to a folder different from input."
        )

    input_resolved = input_folder.resolve()
    while True:
        print("\nDonor folder for BG + pose references")
        print(f"  (must differ from input: {input_resolved})")
        raw = input("Donor path: ").strip().strip('"')
        if not raw:
            print("Path required.")
            continue
        donor = Path(raw).expanduser()
        if not donor.is_absolute():
            donor = (script_dir / donor).resolve()
        else:
            donor = donor.resolve()
        if not donor.is_dir():
            print(f"Not a folder: {donor}")
            continue
        if _norm_path(str(donor)) == _norm_path(str(input_resolved)):
            print("Donor folder cannot be the same as the input folder. Pick another path.")
            continue
        return donor


def pil_image_to_replicate_file(image: PILImage.Image, name: str = "reference.png") -> BytesIO:
    """Convert PIL image to a Replicate-uploadable PNG file object."""
    buffer = BytesIO()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


def build_replicate_client(api_token: str) -> replicate.Client:
    """
    Replicate client tuned for this pipeline.

    - Long write timeout for file uploads (Replicate Files API)
    - Polling mode (wait=False) used at call sites for long image jobs
    """
    timeout = httpx.Timeout(
        connect=REPLICATE_HTTP_CONNECT_TIMEOUT,
        read=REPLICATE_HTTP_READ_TIMEOUT,
        write=REPLICATE_HTTP_WRITE_TIMEOUT,
        pool=REPLICATE_HTTP_POOL_TIMEOUT,
    )
    client = replicate.Client(api_token=api_token, timeout=timeout)
    client.poll_interval = REPLICATE_POLL_INTERVAL
    return client


class RollingRateLimiter:
    """Thread-safe rolling window limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._timestamps = [
                ts for ts in self._timestamps if now - ts < self.period_seconds
            ]
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._timestamps[0]) + 0.5
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                self._timestamps = [
                    ts for ts in self._timestamps if now - ts < self.period_seconds
                ]
            self._timestamps.append(time.monotonic())


def api_retry_after_seconds(exc: Exception) -> float:
    """Parse rate-limit hint, e.g. 'try again in 12s'."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return 15.0


def is_retryable_replicate_error(exc: Exception) -> bool:
    """True for transient network/server errors worth retrying."""
    error_text = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    retryable_fragments = (
        "server disconnected",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection error",
        "broken pipe",
        "timeout",
        "timed out",
        "temporary failure",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "429",
        "rate",
        "thrott",
        "502",
        "503",
        "504",
        "remote end closed",
        "unexpected eof",
        "unexpected end",
        "incomplete read",
        "network",
        "socket",
        "ssl",
        "read timed out",
        "write timed out",
        "pool timeout",
    )
    if any(fragment in error_text for fragment in retryable_fragments):
        return True
    if any(fragment in exc_name for fragment in ("timeout", "connection", "remote", "protocol")):
        return True
    return False


def is_timeout_error(exc: Exception) -> bool:
    exc_name = type(exc).__name__.lower()
    error_text = str(exc).lower()
    return "timeout" in exc_name or "timed out" in error_text


def retry_backoff_seconds(attempt: int, exc: Optional[Exception] = None) -> float:
    """Exponential backoff with jitter; honor rate-limit hints when present."""
    if exc is not None:
        error_text = str(exc).lower()
        if "rate" in error_text or "429" in error_text or "thrott" in error_text:
            return api_retry_after_seconds(exc)
        if is_timeout_error(exc):
            return min(30.0, 4.0 * attempt + random.uniform(0.5, 2.0))
    return min(90.0, REPLICATE_RETRY_BACKOFF_BASE ** attempt + random.uniform(0.5, 2.0))


def reset_input_payload_files(input_payload: Dict) -> None:
    """Rewind file-like objects in a Replicate input payload before retry."""
    for value in input_payload.values():
        if isinstance(value, list):
            for item in value:
                if hasattr(item, "seek"):
                    item.seek(0)
        elif hasattr(value, "seek"):
            value.seek(0)


def replicate_output_to_text(output: Any) -> str:
    """Normalize Replicate text output (string or iterator) to one string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        return "".join(str(chunk) for chunk in output)
    try:
        return "".join(str(chunk) for chunk in output)
    except TypeError:
        return str(output)


def infer_time_of_day_from_prompt(prompt: str) -> Optional[str]:
    """Infer lighting period from generation prompt wording."""
    text = prompt.lower()
    if any(re.search(pattern, text) for pattern in NIGHT_PROMPT_PATTERNS):
        return "night"
    if any(re.search(pattern, text) for pattern in EVENING_PROMPT_PATTERNS):
        return "evening"
    if any(re.search(pattern, text) for pattern in DAWN_PROMPT_PATTERNS):
        return "dawn"
    if "morning" in text:
        return "morning"
    if "afternoon" in text:
        return "afternoon"
    if "midday" in text or "noon" in text:
        return "midday"
    if "indoor" in text or "fluorescent" in text or "tube light" in text:
        return "indoor"
    return None


def infer_time_of_day_from_image(image: PILImage.Image) -> str:
    """Infer lighting period from overall image brightness."""
    gray = image.convert("L").resize((128, 128))
    brightness = sum(gray.getdata()) / (128 * 128)

    if brightness < 48:
        return "night"
    if brightness < 72:
        return "evening"
    if brightness < 95:
        return "dawn"
    if brightness < 125:
        return "indoor"
    if brightness < 155:
        return "afternoon"
    return "midday"


def resolve_time_of_day(prompt: str, image: PILImage.Image) -> str:
    """Match EXIF time to the image — image brightness wins for dark scenes."""
    image_tod = infer_time_of_day_from_image(image)
    prompt_tod = infer_time_of_day_from_prompt(prompt)

    dark_periods = {"night", "late_night", "evening", "dawn"}
    bright_periods = {"midday", "afternoon", "morning"}

    if image_tod in dark_periods:
        if image_tod == "dawn" and prompt_tod in {"evening", "night"}:
            return prompt_tod
        return image_tod

    if prompt_tod in dark_periods:
        return prompt_tod

    if prompt_tod in bright_periods:
        return prompt_tod

    return image_tod


def pick_capture_datetime(seed: str, time_of_day: str) -> datetime:
    """Pick a stable random capture time within the last 7 days."""
    rng = random.Random(seed)
    days_ago = rng.randint(0, 6)
    hour_lo, hour_hi = TIME_OF_DAY_HOUR_RANGES.get(time_of_day, (9, 17))
    hour = rng.randint(hour_lo, hour_hi)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    capture_day = datetime.now() - timedelta(days=days_ago)
    return capture_day.replace(hour=hour, minute=minute, second=second, microsecond=0)


def pick_shooting_settings(seed: str, time_of_day: str) -> Dict[str, Any]:
    """Pick stable per-image shooting settings matched to lighting conditions."""
    rng = random.Random(f"{seed}:shooting")
    focal_index = rng.randrange(len(FOCAL_LENGTHS))

    if time_of_day in {"night", "late_night"}:
        iso = rng.choice((800, 1000, 1250, 1600, 2000, 2500, 3200))
        exposure = rng.choice(((1, 30), (1, 40), (1, 50), (1, 60), (1, 80), (1, 100)))
        fnumber = rng.choice(((28, 10), (32, 10), (35, 10), (40, 10)))
        flash = rng.choice((24, 25))
    elif time_of_day == "evening":
        iso = rng.choice((400, 500, 640, 800, 1000))
        exposure = rng.choice(((1, 60), (1, 80), (1, 100), (1, 125), (1, 160)))
        fnumber = rng.choice(((35, 10), (40, 10), (45, 10), (50, 10)))
        flash = rng.choice((16, 24))
    elif time_of_day == "dawn":
        iso = rng.choice((320, 400, 500, 640))
        exposure = rng.choice(((1, 100), (1, 125), (1, 160), (1, 200)))
        fnumber = rng.choice(((40, 10), (45, 10), (50, 10), (56, 10)))
        flash = 16
    elif time_of_day == "indoor":
        iso = rng.choice((400, 500, 640, 800))
        exposure = rng.choice(((1, 80), (1, 100), (1, 125), (1, 160), (1, 200)))
        fnumber = rng.choice(((40, 10), (45, 10), (50, 10), (56, 10)))
        flash = rng.choice((16, 24))
    else:
        iso = rng.choice(ISO_VALUES)
        exposure = rng.choice(EXPOSURE_TIMES)
        fnumber = rng.choice(FNUMBERS)
        flash = rng.choice(FLASH_VALUES)

    return {
        "exposure": exposure,
        "fnumber": fnumber,
        "iso": iso,
        "focal_length": FOCAL_LENGTHS[focal_index],
        "focal_35mm": FOCAL_35MM[focal_index],
        "flash": flash,
    }


def build_jpeg_exif(seed: str, image: PILImage.Image, prompt: str = "") -> PILImage.Exif:
    """Build camera-style EXIF metadata matched to image lighting."""
    time_of_day = resolve_time_of_day(prompt, image)
    captured_at = pick_capture_datetime(seed, time_of_day)
    settings = pick_shooting_settings(seed, time_of_day)
    timestamp = captured_at.strftime("%Y:%m:%d %H:%M:%S")
    profile = CAMERA_PROFILE

    exif = PILImage.Exif()
    exif[0x010F] = profile["make"]
    exif[0x0110] = profile["model"]
    exif[0x0132] = timestamp

    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9003] = timestamp
    exif_ifd[0x9004] = timestamp
    exif_ifd[0x829A] = settings["exposure"]
    exif_ifd[0x829D] = settings["fnumber"]
    exif_ifd[0x8827] = settings["iso"]
    exif_ifd[0x920A] = settings["focal_length"]
    exif_ifd[0xA405] = settings["focal_35mm"]
    exif_ifd[0x9205] = profile["max_aperture"]
    exif_ifd[0x9207] = profile["metering_mode"]
    exif_ifd[0x9209] = settings["flash"]
    exif_ifd[0x9204] = (0, 1)
    return exif


def is_foreign_background_prompt(prompt_data: Dict[str, Any]) -> bool:
    """True when pose/bg output drifts into foreign/European tourist backgrounds."""
    text = " ".join(
        str(prompt_data.get(key, ""))
        for key in ("new_background", "prompt", "shot_description", "context_lock")
    ).lower()
    return any(re.search(pattern, text) for pattern in FOREIGN_BACKGROUND_PATTERNS)


class EditorialPoseTransformer:
    """Pose / upscale / upscale+pose-bg modes via Gemini 3 Flash + Nano Banana 2."""

    MODES = PIPELINE_MODES
    PROMPT_MODEL = "google/gemini-3-flash"
    IMAGE_MODEL = "google/nano-banana-2"
    PROMPT_TEMPERATURE = 2.0
    POSE_BG_TEMPERATURE = 1.2

    def __init__(
        self,
        replicate_api_key: Optional[str] = None,
        mode: str = DEFAULT_MODE,
        resolution: Optional[str] = None,
        aspect_ratio: str = "2:3",
        max_workers: int = DEFAULT_MAX_WORKERS,
        embed_metadata: bool = False,
    ):
        self.setup_logging()

        self.mode = mode
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        self.replicate_api_key = (replicate_api_key or get_replicate_api_key() or "").strip()
        if not self.replicate_api_key:
            raise ValueError(
                "Replicate API key is required. Set REPLICATE_API_TOKEN or REPLICATE_API_KEY."
            )

        os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_key
        self.replicate_client = build_replicate_client(self.replicate_api_key)

        if resolution:
            self.resolution = resolution
        elif self.mode in ("upscale", "upscale_pose_bg", "both", "upscale_pose"):
            self.resolution = "4K"
        else:
            self.resolution = "2K"
        self.aspect_ratio = aspect_ratio
        self.max_workers = max_workers
        self.embed_metadata = embed_metadata
        self._rate_limiter = RollingRateLimiter(
            REPLICATE_RATE_LIMIT,
            REPLICATE_RATE_PERIOD_SECONDS,
        )
        self._concurrency_semaphore = threading.Semaphore(REPLICATE_MAX_CONCURRENT)

        self.logger.info("Replicate client initialized")
        if self.embed_metadata:
            self.logger.info("JPEG metadata: camera EXIF will be embedded in outputs")
        if self.mode == "upscale":
            self.logger.info(
                "Mode: upscale | image model: %s | resolution: %s | no LLM",
                self.IMAGE_MODEL,
                self.resolution,
            )
        elif self.mode == "upscale_pose_bg":
            self.logger.info(
                "Mode: upscale+pose/bg | prompt model: %s | image model: %s | resolution: %s | aspect: %s",
                self.PROMPT_MODEL,
                self.IMAGE_MODEL,
                self.resolution,
                self.aspect_ratio,
            )
        elif self.mode == "both":
            self.logger.info(
                "Mode: both (_1 pose/bg + _2 weird pose) | prompt: %s | image: %s | resolution: %s | aspect: %s",
                self.PROMPT_MODEL,
                self.IMAGE_MODEL,
                self.resolution,
                self.aspect_ratio,
            )
        elif self.mode == "upscale_pose":
            self.logger.info(
                "Mode: upscale_pose (_1 upscale + _2 weird pose) | prompt: %s | image: %s | resolution: %s | aspect: %s",
                self.PROMPT_MODEL,
                self.IMAGE_MODEL,
                self.resolution,
                self.aspect_ratio,
            )
        else:
            self.logger.info(
                "Mode: pose | prompt model: %s | image model: %s | resolution: %s | aspect: %s",
                self.PROMPT_MODEL,
                self.IMAGE_MODEL,
                self.resolution,
                self.aspect_ratio,
            )

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def get_image_files(self, folder_path: str) -> List[str]:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in image_extensions:
                image_files.append(file_path)

        self.logger.info("Found %d image(s) in %s", len(image_files), folder_path)
        return image_files

    def load_and_optimize_image(self, image_path: str) -> PILImage.Image:
        with PILImage.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            max_size = 2048
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)

            return img.copy()

    def output_image_path(self, input_path: str, output_folder: str) -> str:
        """Single-output modes: upscale → _1, pose → _2 (same shared folder)."""
        stem = Path(input_path).stem
        suffix = "2" if self.mode == "pose" else "1"
        return os.path.join(output_folder, f"{stem}_{suffix}.jpg")

    def output_paths_for_input(self, input_path: str, output_folder: str) -> Dict[str, str]:
        stem = Path(input_path).stem
        return {
            "image_1": os.path.join(output_folder, f"{stem}_1.jpg"),
            "image_2": os.path.join(output_folder, f"{stem}_2.jpg"),
        }

    @staticmethod
    def _output_exists(path: str) -> bool:
        return os.path.exists(path) and os.path.getsize(path) > 0

    def _run_pose_bg_stage(
        self,
        input_image: PILImage.Image,
        output_path: str,
        *,
        bg_source: Optional[str] = None,
        pose_source: Optional[str] = None,
        log_name: str = "",
    ) -> Dict[str, Any]:
        """Mode-3 stage: cross-image BG/pose transfer + 4K upscale."""
        bg_image: Optional[PILImage.Image] = None
        pose_image: Optional[PILImage.Image] = None
        if bg_source:
            bg_image = self.load_and_optimize_image(bg_source)
            self.logger.info("BG donor for %s → %s", log_name, Path(bg_source).name)
        if pose_source:
            pose_image = self.load_and_optimize_image(pose_source)
            self.logger.info("Pose donor for %s → %s", log_name, Path(pose_source).name)

        prompt_data = self.generate_prompt(
            input_image,
            bg_image=bg_image,
            pose_image=pose_image,
            mode_override="upscale_pose_bg",
        )
        image_prompt = self.build_image_prompt(prompt_data, mode_override="upscale_pose_bg")
        success = self.generate_image(
            image_prompt,
            input_image,
            output_path,
            bg_image=bg_image,
            pose_image=pose_image,
            mode_override="upscale_pose_bg",
        )
        return {
            "success": success,
            "shot_description": (prompt_data.get("shot_description") or "").strip() or None,
        }

    def _run_weird_pose_stage(
        self,
        input_image: PILImage.Image,
        output_path: str,
    ) -> Dict[str, Any]:
        """Mode-2 stage: invent weird pose, lock outfit/bg from reference."""
        prompt_data = self.generate_prompt(input_image, mode_override="pose")
        image_prompt = self.build_image_prompt(prompt_data, mode_override="pose")
        success = self.generate_image(
            image_prompt,
            input_image,
            output_path,
            mode_override="pose",
        )
        return {
            "success": success,
            "shot_description": (prompt_data.get("shot_description") or "").strip() or None,
        }

    def process_image(
        self,
        input_path: str,
        output_folder: str,
        *,
        bg_source: Optional[str] = None,
        pose_source: Optional[str] = None,
    ) -> Dict:
        name = Path(input_path).name
        self.logger.info("Processing %s", name)

        try:
            if self.mode in ("both", "upscale_pose_bg"):
                return self._process_both_image(
                    input_path,
                    output_folder,
                    bg_source=bg_source,
                    pose_source=pose_source,
                )
            if self.mode == "upscale_pose":
                return self._process_upscale_pose_image(input_path, output_folder)

            input_image = self.load_and_optimize_image(input_path)
            image_path = self.output_image_path(input_path, output_folder)
            shot_description: Optional[str] = None

            if self._output_exists(image_path):
                self.logger.info("Skipping %s (output already exists)", name)
                return {
                    "input_image": input_path,
                    "output_image": image_path,
                    "success": True,
                    "skipped": True,
                }

            if self.mode == "upscale":
                image_prompt = UPSCALE_PROMPT
                success = self.generate_image(image_prompt, input_image, image_path)
            else:
                stage = self._run_weird_pose_stage(input_image, image_path)
                success = bool(stage["success"])
                shot_description = stage.get("shot_description")

            result = {
                "input_image": input_path,
                "output_image": image_path if success else None,
                "success": success,
            }
            if shot_description:
                result["shot_description"] = shot_description
            if bg_source:
                result["bg_source"] = bg_source
            if pose_source:
                result["pose_source"] = pose_source
            return result
        except Exception as exc:
            self.logger.error("Failed on %s: %s", name, exc)
            return {
                "input_image": input_path,
                "output_image": None,
                "success": False,
                "error": str(exc),
            }

    def _process_both_image(
        self,
        input_path: str,
        output_folder: str,
        *,
        bg_source: Optional[str] = None,
        pose_source: Optional[str] = None,
    ) -> Dict:
        """_1 = pose/bg transfer, _2 = weird pose on _1 (modes: upscale_pose_bg, both)."""
        name = Path(input_path).name
        paths = self.output_paths_for_input(input_path, output_folder)
        image_1_path = paths["image_1"]
        image_2_path = paths["image_2"]
        has_1 = self._output_exists(image_1_path)
        has_2 = self._output_exists(image_2_path)

        if has_1 and has_2:
            self.logger.info("Skipping %s (both outputs already exist)", name)
            return {
                "input_image": input_path,
                "output_image": image_1_path,
                "output_image_1": image_1_path,
                "output_image_2": image_2_path,
                "success": True,
                "skipped": True,
            }

        self.logger.info(
            "Processing %s (resume: image_1=%s, image_2=%s)",
            name,
            "exists" if has_1 else "missing",
            "exists" if has_2 else "missing",
        )

        input_image = self.load_and_optimize_image(input_path)
        desc_1: Optional[str] = None
        desc_2: Optional[str] = None

        if not has_1:
            stage1 = self._run_pose_bg_stage(
                input_image,
                image_1_path,
                bg_source=bg_source,
                pose_source=pose_source,
                log_name=name,
            )
            if not stage1["success"]:
                return {
                    "input_image": input_path,
                    "output_image": None,
                    "output_image_1": None,
                    "output_image_2": None,
                    "success": False,
                    "error": "image_1 (pose/bg) generation failed",
                }
            desc_1 = stage1.get("shot_description")
            has_1 = True
        else:
            self.logger.info("Reusing existing image 1: %s", image_1_path)

        if not has_2:
            image_1_pil = self.load_and_optimize_image(image_1_path)
            stage2 = self._run_weird_pose_stage(image_1_pil, image_2_path)
            if not stage2["success"]:
                return {
                    "input_image": input_path,
                    "output_image": image_1_path,
                    "output_image_1": image_1_path,
                    "output_image_2": None,
                    "success": False,
                    "shot_description": desc_1,
                    "error": "image_2 (weird pose) generation failed",
                }
            desc_2 = stage2.get("shot_description")
        else:
            self.logger.info("Reusing existing image 2: %s", image_2_path)

        parts = []
        if desc_1:
            parts.append(f"1: {desc_1}")
        if desc_2:
            parts.append(f"2: {desc_2}")
        combined = " | ".join(parts) if parts else None

        result = {
            "input_image": input_path,
            "output_image": image_1_path,
            "output_image_1": image_1_path,
            "output_image_2": image_2_path,
            "success": True,
        }
        if combined:
            result["shot_description"] = combined
        if desc_1:
            result["shot_description_1"] = desc_1
        if desc_2:
            result["shot_description_2"] = desc_2
        if bg_source:
            result["bg_source"] = bg_source
        if pose_source:
            result["pose_source"] = pose_source
        return result

    def _process_upscale_pose_image(self, input_path: str, output_folder: str) -> Dict:
        """_1 = upscale only, _2 = weird pose on _1."""
        name = Path(input_path).name
        paths = self.output_paths_for_input(input_path, output_folder)
        image_1_path = paths["image_1"]
        image_2_path = paths["image_2"]
        has_1 = self._output_exists(image_1_path)
        has_2 = self._output_exists(image_2_path)

        if has_1 and has_2:
            self.logger.info("Skipping %s (both outputs already exist)", name)
            return {
                "input_image": input_path,
                "output_image": image_1_path,
                "output_image_1": image_1_path,
                "output_image_2": image_2_path,
                "success": True,
                "skipped": True,
            }

        self.logger.info(
            "Processing %s (resume: image_1=%s, image_2=%s)",
            name,
            "exists" if has_1 else "missing",
            "exists" if has_2 else "missing",
        )

        input_image = self.load_and_optimize_image(input_path)
        desc_2: Optional[str] = None

        if not has_1:
            success_1 = self.generate_image(
                UPSCALE_PROMPT,
                input_image,
                image_1_path,
                mode_override="upscale",
            )
            if not success_1:
                return {
                    "input_image": input_path,
                    "output_image": None,
                    "output_image_1": None,
                    "output_image_2": None,
                    "success": False,
                    "error": "image_1 (upscale) generation failed",
                }
            has_1 = True
        else:
            self.logger.info("Reusing existing image 1: %s", image_1_path)

        if not has_2:
            image_1_pil = self.load_and_optimize_image(image_1_path)
            stage2 = self._run_weird_pose_stage(image_1_pil, image_2_path)
            if not stage2["success"]:
                return {
                    "input_image": input_path,
                    "output_image": image_1_path,
                    "output_image_1": image_1_path,
                    "output_image_2": None,
                    "success": False,
                    "error": "image_2 (weird pose) generation failed",
                }
            desc_2 = stage2.get("shot_description")
        else:
            self.logger.info("Reusing existing image 2: %s", image_2_path)

        result = {
            "input_image": input_path,
            "output_image": image_1_path,
            "output_image_1": image_1_path,
            "output_image_2": image_2_path,
            "success": True,
        }
        if desc_2:
            result["shot_description"] = f"1: upscale | 2: {desc_2}"
            result["shot_description_2"] = desc_2
        else:
            result["shot_description"] = "1: upscale"
        return result

    def _load_descriptions_json(self, output_folder: str, json_name: str = DESCRIPTIONS_JSON) -> Dict[str, str]:
        json_path = os.path.join(output_folder, json_name)
        if not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                return {str(k): str(v) for k, v in existing.items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            self.logger.warning("Could not load existing descriptions: %s", exc)
        return {}

    def save_descriptions_json(self, output_folder: str, items: List[Dict]) -> None:
        if self.mode == "upscale_pose_bg":
            json_name = POSE_BG_DESCRIPTIONS_JSON
            label = "pose/bg descriptions"
        elif self.mode == "both":
            json_name = BOTH_DESCRIPTIONS_JSON
            label = "both-mode descriptions"
        elif self.mode == "upscale_pose":
            json_name = UPSCALE_POSE_DESCRIPTIONS_JSON
            label = "upscale+pose descriptions"
        else:
            json_name = DESCRIPTIONS_JSON
            label = "pose descriptions"
        descriptions = self._load_descriptions_json(output_folder, json_name)
        updated_count = 0

        for item in items:
            if not item.get("success") or item.get("skipped"):
                continue
            description = item.get("shot_description")
            if not description:
                continue
            name = Path(item["input_image"]).name
            descriptions[name] = description
            updated_count += 1

        if not descriptions:
            return

        json_path = os.path.join(output_folder, json_name)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(descriptions, handle, indent=2, ensure_ascii=False)
        if updated_count:
            self.logger.info(
                "Updated %s (%d new/updated): %s",
                label,
                updated_count,
                json_path,
            )
        else:
            self.logger.info("%s preserved: %s", label.capitalize(), json_path)

    def run(
        self,
        input_folder: str,
        output_folder: str,
        *,
        donor_folder: Optional[str] = None,
    ) -> Dict:
        image_files = sorted(self.get_image_files(input_folder))
        if not image_files:
            raise FileNotFoundError(f"No images found in {input_folder}")

        os.makedirs(output_folder, exist_ok=True)
        workers = min(self.max_workers, len(image_files))

        results = {
            "total": len(image_files),
            "success": 0,
            "failed": 0,
            "items": [],
        }

        uses_donors = self.mode in ("upscale_pose_bg", "both")
        donor_files: List[str] = []
        if uses_donors:
            if not donor_folder:
                raise ValueError("donor_folder is required for pose/bg transfer modes")
            donor_files = sorted(self.get_image_files(donor_folder))
            if len(donor_files) < 2:
                raise FileNotFoundError(
                    f"Donor folder needs at least 2 images for BG+pose transfer, found {len(donor_files)}: {donor_folder}"
                )
            needed = 2 * len(image_files)
            print(f"\nDonor folder: {donor_folder}")
            print(f"Donor images: {len(donor_files)} (batch needs {needed} unique slots for no reuse)")
            if len(donor_files) < needed:
                print(
                    f"NOTE: {len(donor_files)} donors < {needed} slots — "
                    "unique assignment first, then reuse only if the pool runs out."
                )

        donor_pairs = (
            build_donor_pairs(image_files, donor_files)
            if uses_donors
            else {p: {"bg_source": None, "pose_source": None} for p in image_files}
        )

        if uses_donors:
            assigned = sum(
                1
                for p in image_files
                if donor_pairs[p].get("bg_source") and donor_pairs[p].get("pose_source")
            )
            print(f"\nCross-image transfer ON — {assigned}/{len(image_files)} subjects got BG+pose donors")
            for path in image_files[: min(5, len(image_files))]:
                pair = donor_pairs[path]
                bg_name = Path(pair["bg_source"]).name if pair.get("bg_source") else "—"
                pose_name = Path(pair["pose_source"]).name if pair.get("pose_source") else "—"
                print(f"  {Path(path).name}: bg←{bg_name}, pose←{pose_name}")
            if len(image_files) > 5:
                print(f"  ... +{len(image_files) - 5} more pairings")

        if self.mode in ("both", "upscale_pose_bg"):
            print(
                f"Per input in {Path(output_folder).name}/: "
                "stem_1.jpg = pose/bg transfer, stem_2.jpg = weird pose on _1"
            )
        elif self.mode == "upscale_pose":
            print("Upscale+pose mode: per input → stem_1.jpg (upscale) then stem_2.jpg (weird pose on _1)")

        print(f"\nProcessing {len(image_files)} images with {workers} parallel image workers")
        print(
            f"Replicate pacing: max {REPLICATE_RATE_LIMIT} creates per "
            f"{int(REPLICATE_RATE_PERIOD_SECONDS)}s, "
            f"{REPLICATE_MAX_CONCURRENT} concurrent uploads, "
            f"unlimited parallel polling"
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for path in image_files:
                pair = donor_pairs.get(path, {})
                bg_src = pair.get("bg_source") if uses_donors else None
                pose_src = pair.get("pose_source") if uses_donors else None
                futures[
                    executor.submit(
                        self.process_image,
                        path,
                        output_folder,
                        bg_source=bg_src,
                        pose_source=pose_src,
                    )
                ] = path

            for i, future in enumerate(as_completed(futures), 1):
                input_path = futures[future]
                name = Path(input_path).name
                item = future.result()
                results["items"].append(item)

                if item["success"]:
                    results["success"] += 1
                    label = "SKIP" if item.get("skipped") else "OK"
                    extra = ""
                    if item.get("shot_description"):
                        extra = f" — {item['shot_description']}"
                    print(f"[{i}/{len(image_files)}] {label}  {name}{extra}")
                else:
                    results["failed"] += 1
                    error = item.get("error", "image generation failed")
                    print(f"[{i}/{len(image_files)}] FAIL {name} — {error}")

        if self.mode in ("pose", "upscale_pose_bg", "both", "upscale_pose"):
            self.save_descriptions_json(output_folder, results["items"])

        return results

    def _extract_json(self, text: str) -> Dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        return json.loads(text)

    def _run_replicate_with_retries(
        self,
        model: str,
        input_payload: Dict,
        *,
        use_file_output: bool = True,
    ) -> Any:
        """
        Run a Replicate model with retries.

        Creates the prediction under the concurrency limit, then polls outside it
        so many long-running jobs can progress in parallel.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, REPLICATE_MAX_RETRIES + 1):
            try:
                reset_input_payload_files(input_payload)
                self._rate_limiter.wait()
                with self._concurrency_semaphore:
                    prediction = self.replicate_client.models.predictions.create(
                        model=model,
                        input=input_payload,
                        wait=False,
                    )

                prediction.wait()

                if prediction.status == "failed":
                    raise ModelError(prediction)

                output = prediction.output
                if use_file_output:
                    return transform_output(output, self.replicate_client)
                return output
            except Exception as exc:
                if not is_retryable_replicate_error(exc):
                    raise
                last_error = exc
                if attempt >= REPLICATE_MAX_RETRIES:
                    break
                wait_for = retry_backoff_seconds(attempt, exc)
                self.logger.warning(
                    "Replicate error (%s), retrying in %.1fs (attempt %d/%d): %s",
                    type(exc).__name__,
                    wait_for,
                    attempt,
                    REPLICATE_MAX_RETRIES,
                    exc,
                )
                time.sleep(wait_for)

        if last_error:
            raise last_error
        raise RuntimeError(f"Replicate call failed after {REPLICATE_MAX_RETRIES} attempts")

    def _download_url_with_retries(self, url: str) -> bytes:
        last_error: Optional[Exception] = None
        download_timeout = httpx.Timeout(
            connect=REPLICATE_HTTP_CONNECT_TIMEOUT,
            read=REPLICATE_DOWNLOAD_READ_TIMEOUT,
            write=30.0,
            pool=REPLICATE_HTTP_POOL_TIMEOUT,
        )
        for attempt in range(1, REPLICATE_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=download_timeout, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.content
            except Exception as exc:
                if not is_retryable_replicate_error(exc):
                    raise
                last_error = exc
                if attempt >= REPLICATE_MAX_RETRIES:
                    break
                wait_for = retry_backoff_seconds(attempt, exc)
                self.logger.warning(
                    "Download error (%s), retrying in %.1fs (attempt %d/%d): %s",
                    type(exc).__name__,
                    wait_for,
                    attempt,
                    REPLICATE_MAX_RETRIES,
                    exc,
                )
                time.sleep(wait_for)

        if last_error:
            raise last_error
        raise RuntimeError(f"Download failed after {REPLICATE_MAX_RETRIES} attempts")

    def generate_prompt(
        self,
        image: PILImage.Image,
        *,
        bg_image: Optional[PILImage.Image] = None,
        pose_image: Optional[PILImage.Image] = None,
        mode_override: Optional[str] = None,
    ) -> Dict:
        self.logger.info("Generating prompt with %s", self.PROMPT_MODEL)
        mode = mode_override or self.mode

        use_transfer = False

        if mode == "upscale_pose_bg" and bg_image is not None and pose_image is not None:
            use_transfer = True
            system_instruction = SYSTEM_PROMPT_POSE_BG
            temperature = self.POSE_BG_TEMPERATURE
            user_text = (
                "You have THREE images:\n"
                "IMAGE 1 = SUBJECT (keep this person's identity + photo realism).\n"
                "IMAGE 2 = BACKGROUND DONOR (copy this REAL environment as the new background).\n"
                "IMAGE 3 = POSE DONOR (copy this REAL full-body pose).\n\n"
                "Step 1: Lock identity + realism from IMAGE 1.\n"
                "Step 2: Describe IMAGE 2 background in concrete detail — that IS new_background. "
                "Do NOT invent a generic kitchen/courtyard/terrace instead. "
                "If IMAGE 2 has other people, REMOVE them — empty environment only, subject alone.\n"
                "Step 3: Describe IMAGE 3 pose GEOMETRY in full-body detail — that IS the pose structure. "
                "Do NOT invent a different candid default pose.\n"
                "Step 4: Invent a NEW outfit that fits IMAGE 2 and the person (not IMAGE 1 clothes).\n"
                "Step 5: Adapt IMAGE 3 pose onto IMAGE 1's body in IMAGE 2. "
                "If man/woman (or body type) differs between IMAGE 1 and IMAGE 3, keep pose angles/contacts "
                "but regenerate anatomy for IMAGE 1 — never copy donor body shape, never masculinize/feminize the subject.\n"
                "Step 6: Full person visible, NO other people in background, 4K detail, not AI-looking.\n\n"
                "Include gender_pose_adapt, no_extra_people, and shot_description (5-10 words). Return ONLY valid JSON."
            )
        elif mode == "pose" and pose_image is not None:
            use_transfer = True
            system_instruction = SYSTEM_PROMPT_POSE_TRANSFER
            temperature = self.PROMPT_TEMPERATURE
            user_text = (
                "You have TWO images:\n"
                "IMAGE 1 = SUBJECT (keep person, outfit, background, lighting, camera).\n"
                "IMAGE 2 = POSE DONOR (copy this REAL full-body pose onto IMAGE 1).\n\n"
                "Step 1: Lock person/outfit/bg/lighting/camera from IMAGE 1.\n"
                "Step 2: Read IMAGE 2 pose geometry in full-body detail.\n"
                "Step 3: Adapt that pose onto IMAGE 1 environment contacts. "
                "If IMAGE 1 and IMAGE 2 differ in gender presentation, transfer pose structure only "
                "and regenerate the body correctly for IMAGE 1 — no donor anatomy bleed, no face-swap look.\n"
                "Step 4: Clear difference from IMAGE 1 original pose; anatomically clean for IMAGE 1; full body visible.\n\n"
                "Do NOT invent a different weird pose — transfer IMAGE 2's pose geometry.\n"
                "Include gender_pose_adapt, weirdness_score (7 or 8), and shot_description (5-10 words). Return ONLY valid JSON."
            )
        elif mode == "upscale_pose_bg":
            system_instruction = SYSTEM_PROMPT_POSE_BG_INVENT
            temperature = self.POSE_BG_TEMPERATURE
            user_text = (
                "Only one reference image is available (no donors). "
                "Study it carefully and invent a NEW everyday background + NEW candid pose + NEW outfit "
                "while locking identity and photo realism. Prefer concrete unique locations, not generic defaults. "
                "Background must have NO other people — subject only.\n"
                "Include no_extra_people and shot_description (5-10 words). Return ONLY valid JSON."
            )
        else:
            system_instruction = SYSTEM_PROMPT
            temperature = self.PROMPT_TEMPERATURE
            user_text = (
                "Study the reference image carefully.\n\n"
                "Step 1: Inventory every usable object/surface. Note railings, ledges, drops.\n"
                "Step 2: Invent FIVE different full-body pose ideas. Reject normal stretch/workout/prayer/leaning/fashion/tourist/Instagram-wall-art/pointing-at-prop poses and reject unsafe stunt poses. "
                "Pick the strongest safe weird option — weirdness_score must be 7 or 8.\n"
                "Step 3: Mandatory checks — pose must visibly differ from reference; hands and legs may do weird things, "
                "but the entire body must change: hips, torso, head, both arms, and both legs. "
                "At least THREE of these must change strongly: torso angle, shoulder level, head direction, arm placement, leg placement, body level. "
                "At least TWO body parts should touch or use the environment; body axis changes 35-60° but does not contort.\n"
                "Step 4: REJECT model-default failures — no single-limb gimmick, no casual lean, no hand-on-hip, "
                "no editorial stance, no same-height standing/sitting, no foot-on-wall stretch, no clasped-hands prayer pose, no yoga/gym warmup, "
                "no pointing at decor, no tourist prop pose, no seated-under-wall-art pose.\n"
                "Step 5: Run ACCEPTABILITY check. If it looks like a normal human activity, add a stronger awkward torso/arm/head change. "
                "If too extreme, reduce intensity. No railing-perching, wall-hanging, climbing, falling, splits, martial arts, or danger near drops.\n"
                "Step 6: Describe COMPLETE body reconfiguration — every limb, what touches floor/wall/object.\n\n"
                "Strong acceptably weird, playful, awkward, and safe — not normal stretching, prayer, workout, leaning, fashion posing, tourist posing, or pointing at wall art. "
                "Let hands and legs do strange things, but make the entire body unusual. "
                "Same person, outfit, background, lighting, camera. "
                "Include weirdness_score (7 or 8 only) and shot_description (5-10 words). Return ONLY valid JSON."
            )

        last_result: Optional[Dict] = None
        for attempt in range(3):
            # Fresh file objects each attempt (uploads consume streams)
            attempt_images = []
            attempt_images.append(pil_image_to_replicate_file(image, name="subject.png"))
            if use_transfer and mode == "upscale_pose_bg" and bg_image is not None and pose_image is not None:
                attempt_images.append(pil_image_to_replicate_file(bg_image, name="bg_donor.png"))
                attempt_images.append(pil_image_to_replicate_file(pose_image, name="pose_donor.png"))
            elif use_transfer and mode == "pose" and pose_image is not None:
                attempt_images.append(pil_image_to_replicate_file(pose_image, name="pose_donor.png"))

            output = self._run_replicate_with_retries(
                self.PROMPT_MODEL,
                {
                    "prompt": user_text,
                    "images": attempt_images,
                    "system_instruction": system_instruction,
                    "thinking_level": "low",
                    "temperature": temperature + (attempt * 0.05),
                },
            )

            response_text = replicate_output_to_text(output)
            if not response_text:
                raise RuntimeError("Empty response from prompt model")

            result = self._extract_json(response_text)
            if "prompt" not in result:
                raise KeyError("Prompt model response missing 'prompt' field")
            last_result = result

            if (
                mode == "upscale_pose_bg"
                and not use_transfer
                and is_foreign_background_prompt(result)
                and attempt < 2
            ):
                self.logger.warning(
                    "Foreign/European background detected (attempt %d/3), retrying",
                    attempt + 1,
                )
                user_text += (
                    "\n\nRetry instruction:\n"
                    "- Previous background was foreign/European — REJECTED.\n"
                    "- Use an everyday INDIAN location only: kitchen, courtyard, terrace, society garden, "
                    "residential lane, office corridor, living room, compound wall path.\n"
                    "- No London brick streets, cobblestone tourist lanes, European shopfronts, or foreign cafe alleys.\n"
                    "- Keep input photo realism and Indian everyday context.\n"
                )
                continue

            if not result.get("shot_description"):
                if mode == "upscale_pose_bg":
                    result["shot_description"] = (
                        result.get("new_pose")
                        or result.get("new_outfit")
                        or result.get("new_background")
                        or ""
                    )
                else:
                    result["shot_description"] = result.get("new_pose") or result.get("weird_action") or ""

            self.logger.info("Prompt generated: %s", result.get("shot_description", ""))
            return result

        if last_result is None:
            raise RuntimeError("Failed to generate prompt")
        if not last_result.get("shot_description"):
            last_result["shot_description"] = (
                last_result.get("new_pose")
                or last_result.get("new_outfit")
                or last_result.get("new_background")
                or last_result.get("weird_action")
                or ""
            )
        self.logger.warning("Using last prompt after retries")
        return last_result

    def build_image_prompt(self, prompt_data: Dict, *, mode_override: Optional[str] = None) -> str:
        """Reinforce mode-specific constraints for the image model."""
        mode = mode_override or self.mode
        if mode == "upscale_pose_bg":
            return self._build_pose_bg_image_prompt(prompt_data)
        return self._build_pose_image_prompt(prompt_data)

    def _build_pose_bg_image_prompt(self, prompt_data: Dict) -> str:
        locks = []
        for key, label in (
            ("identity_lock", "IDENTITY from IMAGE 1 (keep identical)"),
            ("input_realism_lock", "INPUT REALISM from IMAGE 1 (preserve photo family)"),
            ("context_lock", "CONTEXT"),
            ("bg_donor_reading", "BACKGROUND from IMAGE 2 (copy this real scene)"),
            ("new_background", "NEW BACKGROUND (must match IMAGE 2, empty of other people)"),
            ("no_extra_people", "NO EXTRA PEOPLE (subject only)"),
            ("pose_donor_reading", "POSE from IMAGE 3 (copy this real pose geometry)"),
            ("gender_pose_adapt", "GENDER/BODY ADAPT (keep IMAGE 1 anatomy)"),
            ("new_outfit", "NEW OUTFIT (must differ from IMAGE 1)"),
            ("lighting_description", "LIGHTING (IMAGE 2 scene + IMAGE 1 photo family)"),
            ("camera_description", "CAMERA"),
        ):
            value = prompt_data.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value = ", ".join(value)
            locks.append(f"{label}: {value}")

        lock_block = "\n".join(locks)
        pose_lines = []
        if prompt_data.get("current_pose"):
            pose_lines.append(f"FROM (IMAGE 1): {prompt_data['current_pose']}")
        if prompt_data.get("new_pose"):
            pose_lines.append(f"TO (IMAGE 3 pose geometry on IMAGE 1 body): {prompt_data['new_pose']}")
        if prompt_data.get("full_body_breakdown"):
            pose_lines.append(f"FULL BODY: {prompt_data['full_body_breakdown']}")
        pose_header = "\n".join(pose_lines)

        return (
            "UPSCALE + CROSS-IMAGE POSE/BG TRANSFER at 4K.\n"
            "IMAGE 1 = subject identity (keep face/body/gender/photo realism).\n"
            "IMAGE 2 = BACKGROUND DONOR — place the subject into THIS exact environment.\n"
            "IMAGE 3 = POSE DONOR — match THIS pose geometry (adapt contacts to IMAGE 2).\n"
            "Do NOT invent a different generic kitchen/courtyard/street. Use IMAGE 2.\n"
            "Do NOT invent a different candid pose. Use IMAGE 3's pose structure.\n"
            "NO OTHER PEOPLE IN BACKGROUND — if IMAGE 2 has bystanders/crowds/extra faces, remove them. "
            "Only the IMAGE 1 subject may appear. No distant figures, no silhouettes, no reflections of other people.\n"
            "If IMAGE 1 and IMAGE 3 differ in gender, transfer pose angles/contacts only — "
            "regenerate body correctly for IMAGE 1. Never masculinize/feminize. Never face-swap onto donor body.\n"
            "New outfit only (not IMAGE 1 clothes). Full person visible. "
            "Photorealistic 4K skin: open pores, freckles, natural skin texture, "
            "fabric detail, natural grain. NOT AI/CGI/plastic skin/beauty-filter.\n\n"
            f"{lock_block}\n\n"
            f"{pose_header}\n\n"
            f"FULL SCENE PROMPT:\n{prompt_data['prompt']}\n\n"
            "4K upscale quality — open pores, freckles, skin texture, fabric detail. "
            "Same person from IMAGE 1. Background from IMAGE 2 (empty of other people). Pose geometry from IMAGE 3."
        )

    def _build_pose_image_prompt(self, prompt_data: Dict) -> str:
        """Reinforce pose-only constraints for the image model."""
        locks = []
        for key, label in (
            ("background_lock", "BACKGROUND (keep identical)"),
            ("lighting_lock", "LIGHTING (keep identical)"),
            ("camera_lock", "CAMERA (keep identical)"),
            ("identity_lock", "IDENTITY (keep identical)"),
            ("outfit_lock", "OUTFIT (keep identical)"),
            ("pose_donor_reading", "POSE DONOR (copy this real pose geometry)"),
            ("gender_pose_adapt", "GENDER/BODY ADAPT (keep IMAGE 1 anatomy)"),
        ):
            value = prompt_data.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value = ", ".join(value)
            locks.append(f"{label}: {value}")

        lock_block = "\n".join(locks)
        pose_lines = []
        if prompt_data.get("scene_reading"):
            pose_lines.append(f"SCENE: {prompt_data['scene_reading']}")
        if prompt_data.get("environment_element"):
            pose_lines.append(f"ELEMENT: {prompt_data['environment_element']}")
        if prompt_data.get("weird_action"):
            pose_lines.append(f"ACTION: {prompt_data['weird_action']}")
        if prompt_data.get("current_pose"):
            pose_lines.append(f"FROM: {prompt_data['current_pose']}")
        if prompt_data.get("pose_category"):
            pose_lines.append(f"STYLE: {prompt_data['pose_category']}")
        if prompt_data.get("new_pose"):
            pose_lines.append(f"TO: {prompt_data['new_pose']}")
        if prompt_data.get("full_body_breakdown"):
            pose_lines.append(f"FULL BODY: {prompt_data['full_body_breakdown']}")
        pose_header = "\n".join(pose_lines)

        transfer_header = (
            "CROSS-IMAGE POSE TRANSFER using the reference images. "
            "IMAGE 1 = subject (keep person, gender, body, outfit, background, lighting, camera). "
            "IMAGE 2 = POSE DONOR — match this pose geometry, adapted to IMAGE 1 surfaces. "
            "If cross-gender, transfer angles/contacts only and regenerate anatomy for IMAGE 1 — "
            "never masculinize/feminize, never face-swap onto donor body. "
            "Do NOT invent a different pose. Do NOT keep IMAGE 1's original pose. "
            "Anatomically clean joints. Full person visible.\n\n"
            if prompt_data.get("pose_donor_reading") or prompt_data.get("pose_category") == "transfer"
            else (
                "STRONG ACCEPTABLY WEIRD POSE-ONLY EDIT using the reference image. "
                "Clearly replace the body pose with something odd and memorable, but keep it believable as a safe quirky photo. "
                "Subject should be crouched, seated sideways, folded, twisted, braced, reaching, kneeling, or awkwardly committed to a scene-specific action "
                "using real background objects while staying visibly stable and safe. "
                "Hands and legs may do weird things: grabbing, reaching, kicking, hooking, bracing, threading, flailing, or dangling. "
                "Make the pose weird through full-body asymmetry, torso angle, shoulder/head/hand contact, level change, and object contact. "
                "At least TWO body parts interact with environment. Body axis changes vs reference. "
                "NOT a single-limb gimmick. NOT a casual lean. NOT same pose with one hand moved. "
                "NOT pointing at decor. NOT a tourist prop pose. NOT an Instagram wall-art pose. NOT seated under wings/crown pointing up. "
                "NOT a normal stretch/workout pose. NOT foot on wall with clasped hands. NOT prayer/thinking pose. NOT yoga/gym warmup. "
                "NOT a stunt, NOT hanging over a wall, NOT perched on a railing, NOT climbing, NOT falling, NOT martial arts, NOT dancer split. "
                "No leaning over drops, no death setups. Anatomically clean joints. "
                "Same person, outfit, background, lighting, camera.\n\n"
            )
        )

        return (
            f"{transfer_header}"
            f"{lock_block}\n\n"
            f"{pose_header}\n\n"
            f"FULL POSE PROMPT:\n{prompt_data['prompt']}\n\n"
            "OUTPUT FRAMING: clean full-bleed 2:3 portrait photo only. "
            "No phone screenshot chrome, status bars, black letterbox bars, navigation UI, or 'Screenshot saved' overlays."
        )

    def save_output_jpeg(self, image_bytes: bytes, output_path: str, prompt: str = "") -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pil_image = PILImage.open(BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        save_kwargs: Dict[str, Any] = {"format": "JPEG", "quality": 100}
        if self.embed_metadata:
            seed = Path(output_path).name
            save_kwargs["exif"] = build_jpeg_exif(seed, pil_image, prompt)
        pil_image.save(output_path, **save_kwargs)

    def _nb2_aspect_ratio(self, *, mode_override: Optional[str] = None) -> str:
        # Always force configured aspect (default 2:3) for upscale and weird pose —
        # never match_input_image, so phone screenshots do not keep tall UI chrome framing.
        return self.aspect_ratio or "2:3"

    def generate_image(
        self,
        prompt: str,
        input_image: PILImage.Image,
        output_path: str,
        *,
        bg_image: Optional[PILImage.Image] = None,
        pose_image: Optional[PILImage.Image] = None,
        mode_override: Optional[str] = None,
    ) -> bool:
        mode = mode_override or self.mode
        self.logger.info("Generating image with %s", self.IMAGE_MODEL)
        self.logger.info("Prompt preview: %s...", prompt[:120])

        image_input = [pil_image_to_replicate_file(input_image, name="subject.png")]
        if mode == "upscale_pose_bg" and bg_image is not None:
            image_input.append(pil_image_to_replicate_file(bg_image, name="bg_donor.png"))
        if pose_image is not None and mode == "upscale_pose_bg":
            image_input.append(pil_image_to_replicate_file(pose_image, name="pose_donor.png"))

        output = self._run_replicate_with_retries(
            self.IMAGE_MODEL,
            {
                "prompt": prompt,
                "image_input": image_input,
                "aspect_ratio": self._nb2_aspect_ratio(mode_override=mode),
                "resolution": self.resolution,
                "output_format": "jpg",
            },
        )

        output_url = str(output).strip()
        if not output_url:
            self.logger.error("No image URL returned from %s", self.IMAGE_MODEL)
            return False

        image_bytes = self._download_url_with_retries(output_url)

        self.save_output_jpeg(image_bytes, output_path, prompt=prompt)
        self.logger.info("Saved generated image: %s", output_path)
        return True


def output_folder_for_mode(mode: str) -> str:
    # Upscale (_1) and pose-change (_2) always share one folder
    if mode in ("upscale", "pose", "upscale_pose"):
        return "output_upscale_pose"
    if mode == "upscale_pose_bg":
        return "output_upscale_pose_bg"
    if mode == "both":
        return "output_both"
    return "output"


def select_embed_metadata(cli_value: Optional[bool] = None) -> bool:
    """Interactive prompt for optional camera EXIF metadata — works for any mode."""
    if cli_value is not None:
        return cli_value

    env_value = os.getenv("EMBED_METADATA", "").strip().lower()
    if env_value in {"1", "true", "yes", "y"}:
        return True
    if env_value in {"0", "false", "no", "n"}:
        return False
    if not sys.stdin.isatty():
        return False

    print("\nEmbed camera metadata (EXIF) in output JPEGs?")
    print("  y) Yes — add camera make/model, ISO, focal length, exposure, etc.")
    print("  n) No  — plain JPEGs (default)")
    choice = input("Metadata [y/N]: ").strip().lower()
    return choice in {"y", "yes", "1"}


def select_pipeline_mode(cli_mode: Optional[str] = None) -> str:
    """Interactive mode select — same style as Model_change/scenario_transform_gpt.py."""
    if cli_mode in PIPELINE_MODES:
        return cli_mode

    env_value = os.getenv("PIPELINE_MODE", "").strip().lower()
    if env_value in {"1", "upscale"}:
        return "upscale"
    if env_value in {"2", "pose", "pose_change"}:
        return "pose"
    if env_value in {
        "3",
        "upscale_pose_bg",
        "upscale+pose/bg",
        "upscale+pose_bg",
        "pose_bg",
        "pose/bg",
    }:
        return "upscale_pose_bg"
    if env_value in {"4", "both", "combo", "all", "2+3"}:
        return "both"
    if env_value in {"5", "upscale_pose", "upscale+pose", "1+2"}:
        return "upscale_pose"

    if not sys.stdin.isatty():
        return DEFAULT_MODE

    print("\nSelect pipeline mode:")
    print("  1) Upscale — Nano Banana 2 only @ 4K → *_1.jpg")
    print("  2) Pose change — same outfit/background, weird pose @ 2K → *_2.jpg")
    print("  3) Upscale + Pose/BG + Weird Pose — donor folder BG/pose, then weird pose @ 4K → *_1 + *_2")
    print("  4) Both — same as 3 into output_both/")
    print("  5) Upscale + Pose — _1 upscale then _2 weird pose on _1 @ 4K (same folder as 1+2)")
    choice = input("Mode [1/2/3/4/5, Enter=pose]: ").strip().lower()
    if choice in {"1", "upscale"}:
        return "upscale"
    if choice in {"3", "upscale_pose_bg", "upscale+pose/bg", "pose_bg", "pose/bg"}:
        return "upscale_pose_bg"
    if choice in {"4", "both", "combo", "all"}:
        return "both"
    if choice in {"5", "upscale_pose", "upscale+pose"}:
        return "upscale_pose"
    return "pose"


def main():
    parser = argparse.ArgumentParser(description="Pose transform / upscale / pose-bg batch via Replicate")
    parser.add_argument(
        "--mode",
        choices=PIPELINE_MODES,
        default=None,
        help="Optional non-interactive mode override",
    )
    parser.add_argument(
        "--donor-folder",
        default=None,
        help="Folder of BG/pose donor images (modes 3/4; or set DONOR_FOLDER)",
    )
    parser.add_argument(
        "--embed-metadata",
        action="store_true",
        help="Embed Nikon D7500-style EXIF in output JPEGs (any mode)",
    )
    parser.add_argument(
        "--no-embed-metadata",
        action="store_true",
        help="Skip EXIF metadata prompt (plain JPEGs)",
    )
    args = parser.parse_args()
    mode = select_pipeline_mode(args.mode)

    titles = {
        "upscale": "Image Upscale (4K)",
        "pose": "Editorial Pose Transformation",
        "upscale_pose_bg": "Upscale + Pose/BG + Weird Pose (4K)",
        "both": "Both — Pose/BG + Weird Pose (4K)",
        "upscale_pose": "Upscale + Weird Pose (4K)",
    }
    print(titles.get(mode, "Pose Transformation"))
    print("=" * 40)

    script_dir = Path(__file__).resolve().parent
    input_folder = script_dir / "input"
    output_folder = script_dir / output_folder_for_mode(mode)

    if args.no_embed_metadata:
        embed_metadata = False
    elif args.embed_metadata:
        embed_metadata = True
    else:
        embed_metadata = select_embed_metadata()

    replicate_api_key = get_replicate_api_key()
    print(f"Mode: {mode}")
    print(f"EXIF metadata: {'ON (NIKON D7500)' if embed_metadata else 'OFF'}")
    if mode == "upscale":
        print("Using Nano Banana 2 upscale only (no LLM) @ 4K")
        print(f"Writes: {{stem}}_1.jpg → {output_folder.name}/")
    elif mode in ("upscale_pose_bg", "both"):
        print("Using Gemini 3 Flash + Nano Banana 2 @ 4K")
        print(f"Per input in {output_folder.name}/: _1 = pose/bg transfer, _2 = weird pose on _1")
        print("Donors: separate folder (not input) — unique BG+pose refs across batch")
    elif mode == "upscale_pose":
        print("Using Nano Banana 2 upscale + Gemini weird pose @ 4K")
        print(f"Per input in {output_folder.name}/: stem_1.jpg = upscale, stem_2.jpg = weird pose on _1")
    else:
        print("Using Gemini 3 Flash + Nano Banana 2 @ 2K")
        print("Pose invent: same person/outfit/bg — weird pose from subject only")
        print(f"Writes: {{stem}}_2.jpg → {output_folder.name}/")
    if replicate_api_key:
        print("Replicate API key loaded from environment")
    else:
        print("ERROR: Set REPLICATE_API_TOKEN or REPLICATE_API_KEY in the environment")
        return

    if not input_folder.exists():
        print(f"Input folder not found: {input_folder}")
        return

    donor_folder: Optional[Path] = None
    if mode in ("upscale_pose_bg", "both"):
        if args.donor_folder:
            donor_folder = Path(args.donor_folder).expanduser()
            if not donor_folder.is_absolute():
                donor_folder = (script_dir / donor_folder).resolve()
            else:
                donor_folder = donor_folder.resolve()
            if not donor_folder.is_dir():
                print(f"Donor folder not found: {donor_folder}")
                return
            if _norm_path(str(donor_folder)) == _norm_path(str(input_folder.resolve())):
                print("Donor folder cannot be the same as the input folder.")
                return
        else:
            try:
                donor_folder = select_donor_folder(input_folder, script_dir)
            except ValueError as exc:
                print(f"ERROR: {exc}")
                return
        print(f"Donor folder: {donor_folder}")

    max_workers = int(
        os.getenv(
            "MAX_WORKERS",
            str(
                DEFAULT_UPSCALE_WORKERS
                if mode in ("upscale", "upscale_pose_bg", "both", "upscale_pose")
                else DEFAULT_MAX_WORKERS
            ),
        )
    )
    if mode == "upscale":
        max_workers = min(max_workers, DEFAULT_UPSCALE_WORKERS)

    try:
        transformer = EditorialPoseTransformer(
            replicate_api_key=replicate_api_key,
            mode=mode,
            max_workers=max_workers,
            embed_metadata=embed_metadata,
        )
        results = transformer.run(
            str(input_folder),
            str(output_folder),
            donor_folder=str(donor_folder) if donor_folder else None,
        )

        print("\n" + "=" * 40)
        print("Batch complete")
        print(f"Total:   {results['total']}")
        print(f"Success: {results['success']}")
        print(f"Failed:  {results['failed']}")
        print(f"Output:  {output_folder}")
        if mode in ("upscale", "pose", "upscale_pose"):
            print(f"Naming in {output_folder.name}/: {{stem}}_1.jpg = upscale, {{stem}}_2.jpg = pose change")
        elif mode in ("both", "upscale_pose_bg"):
            print(f"Naming in {output_folder.name}/: {{stem}}_1.jpg = pose/bg, {{stem}}_2.jpg = weird pose")
        if mode == "pose":
            print(f"Descriptions: {output_folder / DESCRIPTIONS_JSON}")
        elif mode == "upscale_pose_bg":
            print(f"Descriptions: {output_folder / POSE_BG_DESCRIPTIONS_JSON}")
        elif mode == "both":
            print(f"Descriptions: {output_folder / BOTH_DESCRIPTIONS_JSON}")
        elif mode == "upscale_pose":
            print(f"Descriptions: {output_folder / UPSCALE_POSE_DESCRIPTIONS_JSON}")

    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as exc:
        print(f"\nError: {exc}")
        logging.error("Unexpected error", exc_info=True)


if __name__ == "__main__":
    main()
