
import re
import pandas as pd


# ============================================================
# 1. CLINICAL VOCABULARY
# ============================================================

CLINICAL_TERMS = {
    "Diagnoses": [
        "pneumonia",
        "infection",
        "heart failure",
        "diabetes",
        "cancer",
        "fracture",
        "anemia",
        "anaemia",
        "asthma",
        "hypertension",
    ],
    "Symptoms": [
        "pain",
        "abdominal pain",
        "tenderness",
        "chest pain",
        "neck pain",
        "lower abdominal pain",
        "headache",
        "fever",
        "cough",
        "breathing difficulties",
        "shortness of breath",
        "fatigue",
    ],
    "Procedures": [
        "surgery",
        "biopsy",
        "CT scan",
        "MRI",
        "tracheostomy",
        "chest X-ray",
        "blood cultures",
    ],
    "Medications": [
        "insulin",
        "saline infusion",
        "antibiotics",
    ],
}


# ============================================================
# 2. TEXT NORMALIZATION AND TERM MATCHING
# ============================================================

def normalize_text(text):
    """Lowercase text and normalize whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def canonical_term(term):
    """Map supported spelling variants to one canonical form."""
    term = normalize_text(term)

    spelling_variants = {
        "anaemia": "anemia",
    }

    return spelling_variants.get(term, term)


def term_pattern(term):
    """
    Create a case-insensitive regex for a term.

    Supports word boundaries, flexible whitespace,
    common plural forms, and anemia/anaemia variants.
    """
    term = normalize_text(term)

    if not term:
        return re.compile(r"(?!)")

    if canonical_term(term) == "anemia":
        return re.compile(
            r"(?<!\w)(?:anemia|anaemia)(?!\w)",
            re.IGNORECASE,
        )

    words = term.split()

    def plural_pattern(word):
        if word.endswith("y") and len(word) > 1:
            return re.escape(word[:-1]) + r"(?:y|ies)"

        if word.endswith(("s", "x", "z", "ch", "sh")):
            return re.escape(word) + r"(?:es)?"

        return re.escape(word) + r"s?"

    if len(words) == 1:
        pattern = plural_pattern(words[0])
    else:
        prefix = r"\s+".join(
            re.escape(word) for word in words[:-1]
        )
        pattern = prefix + r"\s+" + plural_pattern(words[-1])

    return re.compile(
        r"(?<!\w)" + pattern + r"(?!\w)",
        re.IGNORECASE,
    )


def _term_spans(text, term):
    """Return all matches for a configured term."""
    return list(term_pattern(term).finditer(str(text)))


# ============================================================
# 3. NEGATION AND UNCERTAINTY
# ============================================================

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies?\b",
    r"\bdenied\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsence of\b",
    r"\bfree of\b",
    r"\bruled out\b",
    r"\bwas excluded\b",
    r"\bwere excluded\b",
    r"\bhas been excluded\b",
    r"\bhas been ruled out\b",
    r"\bnot present\b",
    r"\bno evidence of\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bcould represent\b",
    r"\bnot confirmed\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\bsuspected\b",
    r"\bsuspicion of\b",
    r"\bmay have\b",
    r"\bmight have\b",
    r"\bcould have\b",
    r"\buncertain\b",
    r"\bquestionable\b",
    r"\bpotential\b",
]

PROCEDURE_CATEGORIES = {"Procedures"}
MEDICATION_CATEGORIES = {"Medications"}

CONTRAST_PATTERN = re.compile(
    r"\b(?:but|however|although|whereas)\b",
    re.IGNORECASE,
)


def _clause_bounds(text, start, end):
    """Find the clause containing a term occurrence."""
    left = 0
    right = len(text)

    for match in re.finditer(r"[.!?;,\n]", text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break

    before = text[left:start]
    contrasts = list(CONTRAST_PATTERN.finditer(before))

    if contrasts:
        left += contrasts[-1].end()

    return left, right


def _is_negated(text, term_start, term_end):
    """Check local negation cues around a term occurrence."""
    before = text[:term_start]
    after = text[term_end:]

    if re.match(r"\s*-\s*free\b", after, re.IGNORECASE):
        return True

    local_before = " ".join(before.split()[-8:])

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, local_before, re.IGNORECASE):
            return True

    local_after = " ".join(after.split()[:8])

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, local_after, re.IGNORECASE):
            return True

    return False


def find_occurrence_status(text, term, start, category):
    """Determine the status of one term occurrence."""
    text = str(text)
    occurrence = None

    for match in _term_spans(text, term):
        if match.start() == start:
            occurrence = match
            break

    if occurrence is None:
        return "NOT_FOUND"

    if (
        category in PROCEDURE_CATEGORIES
        or category in MEDICATION_CATEGORIES
    ):
        return "MENTIONED"

    left, right = _clause_bounds(
        text,
        occurrence.start(),
        occurrence.end(),
    )

    clause = text[left:right]
    relative_start = occurrence.start() - left
    relative_end = occurrence.end() - left

    if _is_negated(clause, relative_start, relative_end):
        return "NEGATED"

    before = clause[:relative_start]

    # SAFE_DISCHARGE_LOW_RISK_INFECTION_FIX
    # A low-risk screening statement does not establish a confirmed infection.
    if (
        category == "Diagnoses"
        and term.lower() == "infection"
        and re.search(r"\blow\s+risk\s+of\b", before, re.IGNORECASE)
    ):
        return "UNCERTAIN"

    for pattern in UNCERTAINTY_PATTERNS:
        if re.search(pattern, before, re.IGNORECASE):
            return "UNCERTAIN"

    return "AFFIRMED_OR_MENTIONED"


def find_term_status(text, term, category):
    """Determine the overall status across all occurrences."""
    matches = _term_spans(text, term)

    if not matches:
        return "NOT_FOUND"

    statuses = [
        find_occurrence_status(
            text,
            term,
            match.start(),
            category,
        )
        for match in matches
    ]

    if "AFFIRMED_OR_MENTIONED" in statuses:
        return "AFFIRMED_OR_MENTIONED"

    if "UNCERTAIN" in statuses:
        return "UNCERTAIN"

    if "NEGATED" in statuses:
        return "NEGATED"

    if "MENTIONED" in statuses:
        return "MENTIONED"

    return "NOT_FOUND"


# ============================================================
# 4. CONFIGURED TERMS AND SOURCE MENTIONS
# ============================================================

def _configured_terms():
    """Return all configured (term, category) pairs."""
    if isinstance(CLINICAL_TERMS, dict):
        return [
            (term, category)
            for category, terms in CLINICAL_TERMS.items()
            for term in terms
        ]

    if isinstance(CLINICAL_TERMS, pd.DataFrame):
        columns = {
            column.lower(): column
            for column in CLINICAL_TERMS.columns
        }

        term_column = columns.get("term")
        category_column = columns.get("category")

        if term_column and category_column:
            return list(
                zip(
                    CLINICAL_TERMS[term_column].astype(str),
                    CLINICAL_TERMS[category_column].astype(str),
                )
            )

    return []


def find_source_mentions(text):
    """Find configured terms, preferring longer overlapping phrases."""
    text = str(text)
    mentions = []

    for term, category in _configured_terms():
        for match in _term_spans(text, term):
            mentions.append({
                "Term": term,
                "Category": category,
                "Start": match.start(),
                "End": match.end(),
                "Status": find_occurrence_status(
                    text,
                    term,
                    match.start(),
                    category,
                ),
            })

    mentions.sort(
        key=lambda item: (
            item["Start"],
            -(item["End"] - item["Start"]),
        )
    )

    filtered = []

    for mention in mentions:
        overlaps_longer = any(
            other["Start"] <= mention["Start"]
            and other["End"] >= mention["End"]
            and (
                other["Start"] < mention["Start"]
                or other["End"] > mention["End"]
            )
            for other in mentions
        )

        if not overlaps_longer:
            filtered.append(mention)

    return filtered


# ============================================================
# 5. SUMMARY STATUS AND PRIORITY
# ============================================================

PAIN_PHRASES = [
    "lower abdominal pain",
    "abdominal pain",
    "chest pain",
    "neck pain",
]


def _independent_summary_matches(summary, term):
    """
    Return matches for a term that are not contained inside
    a longer configured clinical phrase.
    """
    summary = str(summary)
    normalized_term = normalize_text(term)

    # SAFE_DISCHARGE_PILOT_ALIAS_FIX
    aliases = {
        "anemia": ["anemia", "anaemia"],
        "anaemia": ["anaemia", "anemia"],
        "cancer": [
            "cancer", "cancers", "adenocarcinoma",
            "carcinoma", "malignant neoplasm",
        ],
        "infection": [
            "infection", "infections", "infective endocarditis",
        ],
    }

    variants = aliases.get(normalized_term, [term])
    matches = []

    for variant in variants:
        for match in term_pattern(variant).finditer(summary):
            matches.append((variant, match))

    if not matches:
        return []

    configured_terms = _configured_terms()
    independent_matches = []

    for matched_term, match in matches:
        contained_in_longer_phrase = False

        for configured_term, _ in configured_terms:
            longer_term = normalize_text(configured_term)

            if len(longer_term) <= len(normalized_term):
                continue

            for longer_match in term_pattern(
                longer_term
            ).finditer(summary):
                if (
                    longer_match.start() <= match.start()
                    and longer_match.end() >= match.end()
                    and (
                        longer_match.start() < match.start()
                        or longer_match.end() > match.end()
                    )
                ):
                    # Allow a configured specific phrase to satisfy
                    # its shorter nested phrase in the summary.
                    # Keep the special handling for standalone "pain".
                    if normalized_term in {"pain", "abdominal pain"}:
                        continue

                    contained_in_longer_phrase = True
                    break

            if contained_in_longer_phrase:
                break

        if not contained_in_longer_phrase:
            independent_matches.append((matched_term, match))

    return independent_matches


def summary_contains_term(summary, term):
    """Check whether a term appears independently in the summary."""
    return bool(_independent_summary_matches(summary, term))


def _summary_status(summary, term, category):
    """Determine the status of independent summary matches."""
    independent_matches = _independent_summary_matches(
        summary,
        term,
    )

    if not independent_matches:
        return "NOT_FOUND"

    statuses = [
        find_occurrence_status(
            summary,
            matched_term,
            match.start(),
            category,
        )
        for matched_term, match in independent_matches
    ]

    if "AFFIRMED_OR_MENTIONED" in statuses:
        return "AFFIRMED_OR_MENTIONED"

    if "UNCERTAIN" in statuses:
        return "UNCERTAIN"

    if "NEGATED" in statuses:
        return "NEGATED"

    if "MENTIONED" in statuses:
        return "MENTIONED"

    return "NOT_FOUND"


def _priority(term, category):
    """Assign a simple rule-based review priority."""
    high_priority = {
        "pneumonia",
        "infection",
        "heart failure",
        "cancer",
        "fracture",
    }

    if canonical_term(term) in high_priority:
        return "High"

    if category in {"Diagnoses", "Medications"}:
        return "Medium"

    return "Low"


# ============================================================
# 6. VERIFY SUMMARY AGAINST SOURCE
# ============================================================

def _pain_phrase_mentions(text, category):
    """
    Return (phrase, status) for each specific pain mention in text.

    A phrase nested inside a longer one is dropped.
    """
    text = str(text)

    spans = [
        (phrase, match.start(), match.end())
        for phrase in PAIN_PHRASES
        for match in term_pattern(phrase).finditer(text)
    ]

    independent = [
        (phrase, start, end)
        for phrase, start, end in spans
        if not any(
            other_start <= start
            and other_end >= end
            and (other_start < start or other_end > end)
            for _, other_start, other_end in spans
        )
    ]

    independent.sort(key=lambda item: item[1])

    return [
        (
            phrase,
            find_occurrence_status(
                text,
                phrase,
                start,
                category,
            ),
        )
        for phrase, start, _ in independent
    ]


# ============================================================
# 7. MEASURED CLINICAL FINDING DETECTOR
# ============================================================

def _extract_measured_findings(text):
    """
    Extract a limited set of measured lesions, masses, tumors,
    nodules, cysts, and fibroids.

    This rule-based detector does not identify every possible
    clinical finding and requires human review.
    """
    text = str(text)

    kind_pattern = (
        r"lesions?|masses?|tumou?rs?|nodules?|cysts?|fibroids?"
    )
    size_pattern = r"\b\d+(?:\.\d+)?\s*-?\s*(?:cm|mm)\b"

    site_pattern = (
        r"iliac\s+fossa|frontoparietal\s+region|parietal\s+lobe|"
        r"adrenal\s+gland|kidneys?|livers?|lungs?|breasts?|thyroid|"
        r"brain|pancreas|ovaries|ovary|colon|bowel|uterus|uterine|"
        r"mediastinum|mediastinal|subcutaneous|pelvis|pelvic|"
        r"abdomen|abdominal|scrotum|scrotal|skin|neck|chest"
    )

    # Pattern A:
    # "4 cm lesion in the left kidney"
    # "an 8-cm uterine fibroid"
    # "80 mm mass"
    before_pattern = re.compile(
        rf"(?P<size>{size_pattern})\s+"
        rf"(?:(?P<preceding_site>{site_pattern})\s+)?"
        rf"(?P<kind>{kind_pattern})\b"
        rf"(?:\s+(?:in|of)\s+(?:the\s+)?"
        rf"(?P<side>left|right|bilateral)?\s*"
        rf"(?P<following_site>{site_pattern})\b)?",
        re.IGNORECASE,
    )

    # Pattern B:
    # "mass in the right iliac fossa measuring 7 cm in diameter"
    # "mediastinal mass measuring 8 cm in its largest dimension"
    after_pattern = re.compile(
        rf"(?P<kind>{kind_pattern})\b"
        rf"(?:(?![.!?]).){{0,120}}?"
        rf"\b(?:measur\w*|diameter|dimension|thick(?:ness)?)\b"
        rf"(?:(?![.!?]).){{0,45}}?"
        rf"(?P<size>{size_pattern})",
        re.IGNORECASE,
    )

    findings = []
    seen = set()

    for pattern in (before_pattern, after_pattern):
        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()

            sentence_start = max(
                text.rfind(".", 0, start),
                text.rfind("!", 0, start),
                text.rfind("?", 0, start),
                text.rfind("\n", 0, start),
            ) + 1

            prefix = text[sentence_start:start].lower()

            # Conservative check for negation and uncertainty cues.
            if re.search(
                r"\b(?:no|not|without|negative\s+for|absence\s+of|"
                r"possible|possibly|suspected|unlikely)\b"
                r"(?:\W+\w+){0,5}\W*$",
                prefix,
            ):
                continue

            size_match = re.search(
                size_pattern,
                match.group("size"),
                re.IGNORECASE,
            )

            if not size_match:
                continue

            raw_kind = match.group("kind").lower().rstrip("s")

            if raw_kind in ("tumor", "tumour"):
                kind = "tumor"
            elif raw_kind == "fibroid":
                kind = "fibroid"
            elif raw_kind == "cyst":
                kind = "cyst"
            elif raw_kind == "nodule":
                kind = "nodule"
            elif raw_kind == "mass":
                kind = "mass"
            else:
                kind = "lesion"

            context_start = max(sentence_start, start - 60)
            context_end = min(len(text), end + 80)
            phrase = text[context_start:context_end]

            side_match = re.search(
                r"\b(left|right|bilateral)\b",
                phrase,
                re.IGNORECASE,
            )

            # Prefer a site explicitly captured next to the finding.
            site_value = (
                match.groupdict().get("following_site")
                or match.groupdict().get("preceding_site")
            )

            # For the "mass measuring 7 cm" form, find the closest
            # anatomical site in the same local context.
            if not site_value:
                site_matches = list(
                    re.finditer(site_pattern, phrase, re.IGNORECASE)
                )

                if site_matches:
                    kind_position = phrase.lower().find(
                        match.group("kind").lower()
                    )

                    closest_site = min(
                        site_matches,
                        key=lambda item: abs(
                            item.start() - kind_position
                        ),
                    )

                    site_value = closest_site.group(0)

            side_value = (
                match.groupdict().get("side")
                or (side_match.group(1) if side_match else "")
            )

            finding = {
                "term": re.sub(
                    r"\s+", " ", match.group(0)
                ).strip(),
                "size": re.sub(
                    r"\s+",
                    "",
                    size_match.group(0),
                ).replace("-", ""),
                "kind": kind,
                "side": side_value.lower() if side_value else "",
                "site": (
                    re.sub(r"\s+", " ", site_value).lower()
                    if site_value else ""
                ),
            }

            key = (
                finding["size"],
                finding["kind"],
                finding["side"],
                finding["site"],
            )

            if key not in seen:
                findings.append(finding)
                seen.add(key)

    return findings


def _finding_is_preserved(finding, summary):
    """
    Check whether the summary preserves the measured finding.

    The measurement and finding type must be present. If a side
    or anatomical site was extracted, it must also be present.
    """
    summary = str(summary)

    size_value = re.match(
        r"(\d+(?:\.\d+)?)\s*-?\s*(cm|mm)",
        finding["size"],
        re.IGNORECASE,
    )

    if not size_value:
        return False

    size_pattern = re.compile(
        r"\b"
        + re.escape(size_value.group(1))
        + r"\s*-?\s*"
        + re.escape(size_value.group(2))
        + r"\b",
        re.IGNORECASE,
    )

    kind_pattern = re.compile(
        r"\b" + re.escape(finding["kind"]) + r"s?\b",
        re.IGNORECASE,
    )

    size_ok = size_pattern.search(summary) is not None
    kind_ok = kind_pattern.search(summary) is not None

    side_ok = (
        not finding["side"]
        or re.search(
            r"\b" + re.escape(finding["side"]) + r"\b",
            summary,
            re.IGNORECASE,
        ) is not None
    )

    site_ok = (
        not finding["site"]
        or re.search(
            r"\b" + re.escape(finding["site"]) + r"\b",
            summary,
            re.IGNORECASE,
        ) is not None
    )

    return size_ok and kind_ok and side_ok and site_ok


# ============================================================
# 8. LOCATION-SPECIFIC TENDERNESS DETECTOR
# ============================================================

def _extract_location_specific_tenderness(text):
    """
    Extract tenderness mentions that specify a side and anatomical site.

    Examples:
    - tenderness in the left abdominal region
    - right abdominal tenderness

    This is a limited rule-based detector requiring human review.
    """
    text = str(text)

    site_pattern = (
        r"abdominal|abdomen|chest|pelvic|pelvis|"
        r"iliac\s+fossa|neck|back|flank|"
        r"shoulder|arm|leg|thigh|calf|"
        r"wrist|ankle|knee|elbow"
    )

    patterns = [
        # Example: tenderness in the left abdominal region
        re.compile(
            rf"\btenderness\b\s+(?:in|over|of|along)\s+"
            rf"(?:the\s+)?(?P<side>left|right|bilateral)\s+"
            rf"(?P<site>{site_pattern})"
            rf"(?:\s+(?:region|area))?\b",
            re.IGNORECASE,
        ),

        # Example: left abdominal tenderness
        re.compile(
            rf"\b(?P<side>left|right|bilateral)\s+"
            rf"(?P<site>{site_pattern})\s+"
            rf"tenderness\b",
            re.IGNORECASE,
        ),
    ]

    findings = []
    seen = set()

    for pattern in patterns:
        for match in pattern.finditer(text):
            side = match.group("side").lower()
            site = re.sub(
                r"\s+", " ", match.group("site").lower()
            )

            finding = {
                "term": f"{side} {site} tenderness",
                "side": side,
                "site": site,
            }

            key = (side, site)

            if key not in seen:
                findings.append(finding)
                seen.add(key)

    return findings


def _tenderness_location_status(finding, summary):
    """
    Check whether tenderness and its specified location appear
    to be preserved in the summary.

    Returns:
    - preserved
    - missing
    - mismatch
    """
    summary = str(summary).lower()
    side = finding["side"]
    site = finding["site"]

    # Tenderness itself must appear in the summary.
    if not re.search(r"\btenderness\b", summary):
        return "missing"

    # The anatomical site must appear.
    if not re.search(r"\b" + re.escape(site) + r"\b", summary):
        return "missing"

    # Check whether the documented side is preserved.
    if not re.search(r"\b" + re.escape(side) + r"\b", summary):
        opposite_sides = {
            "left": ("right",),
            "right": ("left",),
            "bilateral": ("left", "right"),
        }

        if any(
            re.search(r"\b" + other + r"\b", summary)
            for other in opposite_sides.get(side, ())
        ):
            return "mismatch"

        return "missing"

    return "preserved"



# ============================================================
# 9. MAIN SUMMARY VERIFICATION
# ============================================================


def _extract_critical_clinical_details(source_note):
    """
    Extract selected clinical details that need explicit preservation.

    This is a limited rule-based detector, not a complete
    clinical finding or complication extractor.
    """
    source_note = str(source_note)
    details = []

    # 1. Spinal canal stenosis
    stenosis_pattern = (
        r"\bspinal\s+canal\s+stenosis\b|"
        r"\bstenosis\s+of\s+the\s+spinal\s+canal\b"
    )

    if re.search(stenosis_pattern, source_note, re.IGNORECASE):
        details.append({
            "term": "spinal canal stenosis",
            "summary_pattern": stenosis_pattern,
        })

    # 2. Measured tracheal lumen
    airway_pattern = (
        r"\btracheal\s+lumen\b"
        r"(?:(?![.!?]).){0,100}?"
        r"\b(\d+(?:\.\d+)?)\s*-?\s*(mm|cm)\b"
    )

    airway_match = re.search(
        airway_pattern,
        source_note,
        re.IGNORECASE,
    )

    if airway_match:
        value = airway_match.group(1)
        unit = airway_match.group(2)

        details.append({
            "term": f"tracheal lumen measurement ({value} {unit})",
            "summary_pattern": (
                r"\btracheal\s+lumen\b"
                r"(?:(?![.!?]).){0,100}?"
                + rf"\b{re.escape(value)}\s*-?\s*"
                + rf"{re.escape(unit)}\b"
            ),
        })

    # 3. Explicit bile-duct removal or injury
    # The event verb may appear before or after the duct phrase.
    bile_duct_patterns = [
        (
            r"\b(?:bile|biliary)\s+duct\b"
            r"(?:(?![.!?]).){0,100}?"
            r"\b(?:inadvertently\s+|accidentally\s+|"
            r"unintentionally\s+)?"
            r"(?:removed|resected|excised|injured|transected)\b"
        ),
        (
            r"\b(?:removed|resected|excised|injured|transected)\b"
            r"(?:(?![.!?]).){0,100}?"
            r"\b(?:bile|biliary)\s+duct\b"
        ),
    ]

    # A stent "to be removed" is a planned future procedure, not
    # evidence that the bile duct itself was removed or injured.
    bile_detection_text = re.sub(
        r"\bto\s+be\s+removed\b",
        "",
        source_note,
        flags=re.IGNORECASE,
    )

    if any(
        re.search(pattern, bile_detection_text, re.IGNORECASE)
        for pattern in bile_duct_patterns
    ):
        details.append({
            "term": "bile duct removal or injury complication",
            "summary_pattern": (
                r"\b(?:bile|biliary)\s+duct\b"
                r"(?:(?![.!?]).){0,100}?"
                r"\b(?:inadvertently\s+|accidentally\s+|"
                r"unintentionally\s+)?"
                r"(?:removed|resected|excised|injured|transected)\b"
                r"|"
                r"\b(?:removed|resected|excised|injured|transected)\b"
                r"(?:(?![.!?]).){0,100}?"
                r"\b(?:bile|biliary)\s+duct\b"
            ),
        })

    return details


def verify_summary(source_note, summary):
    """
    Compare configured source terms and selected clinical details
    against the generated summary.

    Returns a DataFrame of rule-based verification flags.
    Flags indicate possible omissions and require human review.
    """
    columns = [
        "Term",
        "Category",
        "Source Status",
        "Summary Status",
        "Verification",
        "Priority",
    ]

    source_note = str(source_note)
    summary = str(summary)

    rows = []
    seen = set()

    # --------------------------------------------------------
    # A. Verify configured clinical vocabulary
    # --------------------------------------------------------

    for term, category in _configured_terms():
        if not _term_spans(source_note, term):
            continue

        key = (canonical_term(term), category)

        if key in seen:
            continue

        seen.add(key)

        source_status = find_term_status(
            source_note,
            term,
            category,
        )

        summary_status = _summary_status(
            summary,
            term,
            category,
        )

        # The general term "pain" must account for every specific
        # non-negated pain phrase found in the source.
        if normalize_text(term) == "pain":
            source_pain = _pain_phrase_mentions(
                source_note,
                category,
            )

            summary_pain = {
                phrase
                for phrase, _ in _pain_phrase_mentions(
                    summary,
                    category,
                )
            }

            if source_pain:
                required = {
                    phrase
                    for phrase, status in source_pain
                    if status != "NEGATED"
                }

                if required:
                    covered = required <= summary_pain
                else:
                    covered = any(
                        phrase in summary_pain
                        for phrase, _ in source_pain
                    )

                if not covered:
                    summary_status = "NOT_FOUND"

        if summary_status == "NOT_FOUND":
            if source_status == "NEGATED":
                verification = "NEGATED IN SOURCE - REVIEW"
            elif source_status == "UNCERTAIN":
                verification = "UNCERTAIN IN SOURCE - REVIEW"
            else:
                verification = "POSSIBLY MISSING"

        elif (
            source_status == "NEGATED"
            and summary_status != "NEGATED"
        ) or (
            source_status == "AFFIRMED_OR_MENTIONED"
            and summary_status == "NEGATED"
        ):
            verification = "CHECK STATUS MISMATCH"

        elif (
            source_status == "UNCERTAIN"
            and summary_status == "AFFIRMED_OR_MENTIONED"
        ):
            verification = "CHECK STATUS MISMATCH"

        else:
            verification = "TERM FOUND - REVIEW CONTEXT"

        rows.append({
            "Term": term,
            "Category": category,
            "Source Status": source_status,
            "Summary Status": summary_status,
            "Verification": verification,
            "Priority": _priority(term, category),
        })

    # --------------------------------------------------------
    # B. Verify measured clinical findings
    # --------------------------------------------------------

    for finding in _extract_measured_findings(source_note):
        preserved = _finding_is_preserved(
            finding,
            summary,
        )

        rows.append({
            "Term": finding["term"],
            "Category": "Clinical Findings",
            "Source Status": "AFFIRMED_OR_MENTIONED",
            "Summary Status": (
                "AFFIRMED_OR_MENTIONED"
                if preserved
                else "NOT_FOUND"
            ),
            "Verification": (
                "TERM FOUND - REVIEW CONTEXT"
                if preserved
                else "POSSIBLY MISSING"
            ),
            "Priority": "High",
        })

    # --------------------------------------------------------
    # C. Verify location-specific tenderness
    # --------------------------------------------------------

    for finding in _extract_location_specific_tenderness(
        source_note
    ):
        status = _tenderness_location_status(
            finding,
            summary,
        )

        rows.append({
            "Term": finding["term"],
            "Category": "Clinical Findings",
            "Source Status": "AFFIRMED_OR_MENTIONED",
            "Summary Status": (
                "AFFIRMED_OR_MENTIONED"
                if status == "preserved"
                else "NOT_FOUND"
            ),
            "Verification": (
                "TERM FOUND - REVIEW CONTEXT"
                if status == "preserved"
                else (
                    "CHECK STATUS MISMATCH"
                    if status == "mismatch"
                    else "POSSIBLY MISSING"
                )
            ),
            "Priority": "High",
        })

    # --------------------------------------------------------
    # D. Verify selected critical clinical details
    # --------------------------------------------------------

    for detail in _extract_critical_clinical_details(source_note):
        preserved = re.search(
            detail["summary_pattern"],
            summary,
            re.IGNORECASE,
        ) is not None

        rows.append({
            "Term": detail["term"],
            "Category": "Clinical Findings",
            "Source Status": "AFFIRMED_OR_MENTIONED",
            "Summary Status": (
                "AFFIRMED_OR_MENTIONED"
                if preserved
                else "NOT_FOUND"
            ),
            "Verification": (
                "TERM FOUND - REVIEW CONTEXT"
                if preserved
                else "POSSIBLY MISSING"
            ),
            "Priority": "High",
        })

    return pd.DataFrame(rows, columns=columns)
