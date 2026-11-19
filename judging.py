"""The staged judge's building blocks: quick deterministic checks, few-shot prompts, and verification
of what the judge model returns. Everything here is pure (no model calls, no project imports), so it
is cheap to test; `agents.judge()` runs the model calls and ties the stages together.

Faithfulness is judged in stages instead of all at once:
  0. deterministic checks, no model:
       - an answer that only declines to answer is graded directly
       - a number in the answer that appears nowhere in the context or the question fails the answer
       - names found nowhere in the context become hints for the claim check
  1. the model splits the answer into separate claims                        (few-shot prompt)
  2. the model checks each claim against the context, quoting evidence       (few-shot prompt)
  3. code verifies each quoted evidence: real text of the context, long enough, related to the claim
  -> the answer is faithful only if every claim is supported by verified evidence.
Relevance is a second, shorter staged call: say what was asked, then decide whether the answer
addresses it.
"""
import json
import re

CLAIM_LIMIT = 10
ABSTENTION_MAX_WORDS = 20
MIN_EVIDENCE_WORDS = 2


class JudgeOutputError(ValueError):
    """The judge model's reply could not be used (not JSON, wrong shape, too many claims)."""


# ------------------------------------------------------------------ text helpers

CONTEXT_HEADER = re.compile(r"^\s*\[[^\]\n]+\](?:\s*\(page [^)\n]*\))?\s*$|^\s*-{3,}\s*$", re.MULTILINE)


def squash(text):
    """Whitespace runs as single spaces."""
    return " ".join(str(text).split())


def strip_context_headers(context):
    """The context without the '[c3] (page 2)' header lines and '---' separators the app adds.

    Those lines are not document text, so they must not count as evidence, and their page numbers
    must not count as numbers the document states.
    """
    return CONTEXT_HEADER.sub("", context)


def _match_form(text):
    """A forgiving form for comparing a model's quote with the document: case, quote and dash styles,
    markdown and table pipes, and whitespace are ignored. Words and numbers must still match."""
    text = (str(text).replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))
    text = re.sub(r"[|*_#`>]", " ", text)
    return squash(text).lower()


def evidence_in_context(evidence, context):
    """True when every fragment of the evidence is real document text and the evidence is long enough.

    '...' or '…' separates fragments, so a claim built from two sentences can quote both. The context's
    header lines are removed first, so '[c1]' or '(page 1)' can never count as evidence.
    """
    body = _match_form(strip_context_headers(context))
    fragments = [f for f in re.split(r"\.{3}|…", str(evidence)) if _match_form(f)]
    if not fragments:
        return False
    if sum(len(_match_form(f).split()) for f in fragments) < MIN_EVIDENCE_WORDS:
        return False
    return all(_match_form(f) in body for f in fragments)


def _content_stems(text):
    """Words of four or more letters (cut to four, so 'opens' matches 'open') and every number."""
    stems = {word[:4] for word in re.findall(r"[a-z]{4,}", text.lower())}
    return stems | extract_numbers(text)


def evidence_relates_to_claim(claim, evidence):
    """Cheap sanity check that the quoted evidence is about the claim: they share a content word or number."""
    return bool(_content_stems(claim) & _content_stems(evidence))


# ------------------------------------------------------------------ deterministic checks

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER_TOKEN = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)

ABSTENTION_PHRASES = re.compile(
    r"\b(i do not know|i don't know|i can't verify|i cannot verify|can't verify|cannot verify|"
    r"not (?:stated|mentioned|provided|specified|found|covered) in|does not (?:say|mention|state|specify|provide)|"
    r"do not have (?:enough )?information|no information (?:about|on|in)|unable to find|"
    r"not in the (?:context|document|text|provided))\b", re.IGNORECASE)
CONTINUATION = re.compile(r"\b(but|however|although|though|yet|except|and|while|whereas|so|because|since)\b", re.IGNORECASE)
FILLER_SENTENCES = {"sorry", "unfortunately", "hmm", "well"}
SENTENCE_STARTERS = {
    "the", "a", "an", "it", "its", "this", "that", "these", "those", "there", "they", "we", "he", "she", "i", "you",
    "yes", "no", "not", "according", "however", "based", "in", "on", "at", "if", "as", "so", "but", "and", "or",
    "for", "to", "from", "with", "by", "because", "since", "while", "also", "per", "each", "all", "some", "any",
    "both", "either", "neither", "when", "where", "what", "who", "which", "how", "why", "after", "before", "during",
    "under", "over", "between", "about", "most", "many", "one", "two", "three", "sorry", "unfortunately",
}


def _canonical(token):
    cleaned = token.replace("$", "").replace("%", "").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


def _word_value(run):
    total = current = 0
    for word in run:
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
    return total + current


def extract_numbers(text):
    """Every number in a text as a canonical string: digits ('1,200' -> '1200', '4.50' -> '4.5') and
    spelled-out numbers up to the thousands ('twenty four' -> '24'). A lone 'one' is ignored because
    it is usually a pronoun, not a quantity, and list markers ('1. First') are not quantities."""
    text = LIST_MARKER.sub("", text)
    found = set()
    for token in NUMBER_TOKEN.findall(text):
        canonical = _canonical(token)
        if canonical is not None:
            found.add(canonical)
    words = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    run = []

    def flush():
        if run and run != ["one"]:
            found.add(str(_word_value(run)))
        run.clear()

    for word in words:
        if word in UNITS or word in TENS or word in ("hundred", "thousand") or (word == "and" and run and run[-1] in ("hundred", "thousand")):
            run.append(word)
        else:
            flush()
    flush()
    return found


def extract_names(text):
    """Capitalised words that look like names, places or acronyms. The first word of a sentence counts
    too unless it is an ordinary sentence starter ('The', 'Yes', 'According')."""
    names = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        for position, token in enumerate(tokens):
            if not token[0].isupper() or not (len(token) > 2 or token.isupper()):
                continue
            if token == "I" or token.startswith("I'"):
                continue  # the pronoun, not a name
            if position == 0 and token.lower() in SENTENCE_STARTERS:
                continue
            if token not in names:
                names.append(token)
    return names


def is_abstention(answer):
    """An answer whose whole point is that it cannot or will not answer. It makes no claims to check.

    Deliberately strict, because an abstention skips the model entirely. It must be short; contain no
    number (digits or spelled out) and no name; use no word that continues a thought ('but', 'and',
    'because'); and every sentence must be a decline or a one-word filler like 'Sorry'. So "I do not
    know, but the CEO was fired" and "Alice stole it. I do not know why." both go to the model.
    """
    text = answer.strip()
    if not text or len(text.split()) > ABSTENTION_MAX_WORDS:
        return False
    if extract_numbers(text) or re.search(r"\d", text) or extract_names(text) or CONTINUATION.search(text):
        return False
    for sentence in re.split(r"[.!?;]+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        words = re.findall(r"[A-Za-z']+", sentence)
        if ABSTENTION_PHRASES.search(sentence) or (len(words) == 1 and words[0].lower() in FILLER_SENTENCES):
            continue
        return False
    return bool(ABSTENTION_PHRASES.search(text))


def _tokens(text):
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def precheck(question, answer, context, citations=None):
    """Quick checks that need no model.

    Returns:
      abstention             the answer only declines to answer (graded without the model)
      unverified_numbers     numbers in the answer found in neither the context nor the question; the
                             judge fails the answer's faithfulness on these without asking the model
      unverified_names       capitalised words in the answer found in neither: hints for the claim check
      no_verified_citation   True when `citations` was given, the answer is not an abstention and no
                             citation survived verification (a flag only; it changes no verdict)
    The context is compared without its '[c3] (page 2)' header lines, so a page number is never
    mistaken for a number the document states. Limits: '5', '$5' and '5%' count as the same number,
    and scale words ('million', 'dozen') are not understood.
    """
    abstention = is_abstention(answer)
    body = strip_context_headers(context)
    known_numbers = extract_numbers(body) | extract_numbers(question)
    known_words = _tokens(body) | _tokens(question)
    return {
        "abstention": abstention,
        "unverified_numbers": sorted(extract_numbers(answer) - known_numbers, key=lambda n: (len(n), n)),
        "unverified_names": [name for name in extract_names(answer) if name.lower() not in known_words][:5],
        "no_verified_citation": citations is not None and not abstention and len(citations) == 0,
    }


def hint_text(checks):
    """The name findings, worded for the claim-check prompt ('' when there is nothing to say)."""
    if not checks.get("unverified_names"):
        return ""
    return ("Automatic pre-check: these names appear in the answer but nowhere in the context or the question: "
            + ", ".join(checks["unverified_names"]) + ". A claim that depends on one is unsupported.")


def numbers_reasoning(numbers):
    """The explanation recorded when an answer fails because of numbers the document never states."""
    return ("The answer states number(s) found nowhere in the context or the question: "
            + ", ".join(numbers) + ". Failed by a deterministic check, without the model.")


# ------------------------------------------------------------------ prompts and schemas

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
}

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "verdict", "evidence", "reason"],
            },
        }
    },
    "required": ["results"],
}

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "asked": {"type": "string"},
        "addresses": {"type": "string", "enum": ["yes", "partly", "no"]},
        "reason": {"type": "string"},
    },
    "required": ["asked", "addresses", "reason"],
}

# The examples are deliberately about unrelated everyday topics, so the judge learns the rules
# instead of anything about the documents being asked about.

CLAIMS_PROMPT = """You split an answer into its separate factual claims so that each one can be checked on its own.

Rules:
- One short, self-contained sentence per claim, in plain words.
- Keep every number, name, unit and date exactly as written in the answer.
- Include only what the answer states. Never add facts and never correct mistakes.
- Skip greetings and filler. At most 10 claims.
- Reply with JSON only, in the shape {"claims": ["...", "..."]}.

Example 1
Answer: The library opens at 9 am on weekdays and closes at 5 pm.
Reply: {"claims": ["The library opens at 9 am on weekdays.", "The library closes at 5 pm on weekdays."]}

Example 2
Answer: Yes, returns are accepted within 30 days of delivery.
Reply: {"claims": ["Returns are accepted within 30 days of delivery."]}

Example 3
Answer: The recipe needs 2 eggs and serves 4 people, so it is cheap to make.
Reply: {"claims": ["The recipe needs 2 eggs.", "The recipe serves 4 people.", "The recipe is cheap to make."]}

Answer to split:
<<ANSWER>>"""

SUPPORT_PROMPT = """You check whether each numbered claim is supported by a context. Use ONLY the context. Do not use what you know about the world.

For each claim decide:
- "supported": the context states it, or says the same thing in other words.
- "unsupported": the context does not state it, contradicts it, states only part of it, or only outside knowledge could confirm it. A true fact that is not in the context is still unsupported. A figure the context never states is unsupported, even if it could be calculated.
For a supported claim, "evidence" must be text copied exactly, word for word, from the context. If the claim needs two sentences, copy both and separate them with " ... ". For an unsupported claim leave "evidence" empty. Keep "reason" to one short sentence.
Reply with JSON only, in the shape {"results": [{"claim": 1, "verdict": "supported", "evidence": "...", "reason": "..."}]}, with one entry per claim.

Example 1
Context: [c1] The museum is open Tuesday to Sunday from 10 am to 6 pm. Entry is free for children under 12.
Claims:
1. Children under 12 enter for free.
2. The museum opens at 10 am.
3. Adult tickets cost 8 dollars.
Reply: {"results": [{"claim": 1, "verdict": "supported", "evidence": "Entry is free for children under 12.", "reason": "Stated directly."}, {"claim": 2, "verdict": "supported", "evidence": "The museum is open Tuesday to Sunday from 10 am to 6 pm.", "reason": "It opens at 10 am."}, {"claim": 3, "verdict": "unsupported", "evidence": "", "reason": "No ticket price is given."}]}

Example 2
Context: [c4] The course has 12 weekly lessons. Each lesson lasts 90 minutes.
Claims:
1. The course has 12 lessons.
2. Each lesson lasts 2 hours.
Reply: {"results": [{"claim": 1, "verdict": "supported", "evidence": "The course has 12 weekly lessons.", "reason": "Stated directly."}, {"claim": 2, "verdict": "unsupported", "evidence": "", "reason": "The context says 90 minutes, not 2 hours."}]}

Example 3
Context: [c2] The bridge was completed in 1932 and spans 503 meters.
Claims:
1. The bridge spans 503 meters.
2. The bridge is the longest in the country.
Reply: {"results": [{"claim": 1, "verdict": "supported", "evidence": "The bridge was completed in 1932 and spans 503 meters.", "reason": "Stated directly."}, {"claim": 2, "verdict": "unsupported", "evidence": "", "reason": "The context never says it is the longest."}]}

Now the real task.
<<HINTS>>
Context:
<<CONTEXT>>

Claims:
<<CLAIMS>>"""

RELEVANCE_PROMPT = """You judge whether an answer addresses the question that was asked. Ignore whether the answer is correct; judge only whether it speaks to the question.

Work in two steps and reply with JSON only, in the shape {"asked": "...", "addresses": "yes", "reason": "..."}:
1. "asked": say in one short sentence what the question is asking for.
2. "addresses": "yes" if the answer gives what was asked for; "partly" if it gives some of it, is vague, or leaves out a main part; "no" if it answers a different question, is off-topic, or declines to answer. Add a one-sentence "reason".

Example 1
Question: What time does the shop close on Saturdays?
Answer: The shop closes at 6 pm on Saturdays.
Reply: {"asked": "The closing time of the shop on Saturdays.", "addresses": "yes", "reason": "It gives the closing time."}

Example 2
Question: How much does shipping cost to Canada?
Answer: Shipping is handled by a courier and takes about a week.
Reply: {"asked": "The price of shipping to Canada.", "addresses": "partly", "reason": "It describes shipping but gives no price."}

Example 3
Question: Who wrote the report?
Answer: The report was published in March.
Reply: {"asked": "The author of the report.", "addresses": "no", "reason": "It gives a date, not an author."}

Now the real task.
Question: <<QUESTION>>
Answer: <<ANSWER>>"""

SLOT = re.compile(r"<<([A-Z]+)>>")


def _fill(template, **values):
    """Fill the <<SLOT>> markers in one pass, so text that happens to contain a marker is left alone."""
    return SLOT.sub(lambda match: values.get(match.group(1).lower(), match.group(0)), template)


def build_claims_prompt(answer):
    return _fill(CLAIMS_PROMPT, answer=answer)


def build_support_prompt(context, claims, checks=None):
    numbered = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, start=1))
    return _fill(SUPPORT_PROMPT, hints=hint_text(checks) if checks else "", context=context, claims=numbered)


def build_relevance_prompt(question, answer):
    return _fill(RELEVANCE_PROMPT, question=question, answer=answer)


# ------------------------------------------------------------------ reading and verifying replies


def _load_object(raw, what):
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise JudgeOutputError(f"the {what} reply was not valid JSON") from error
    if not isinstance(data, dict):
        raise JudgeOutputError(f"the {what} reply was not a JSON object")
    return data


def parse_claims(raw, answer):
    """The claims to check. If the model lists none, the whole answer is one claim. More than
    CLAIM_LIMIT is an error, not a silent cut: unchecked claims must never turn into a pass."""
    data = _load_object(raw, "claim-splitting")
    claims = [squash(c) for c in data.get("claims") or [] if isinstance(c, str) and squash(c)]
    if len(claims) > CLAIM_LIMIT:
        raise JudgeOutputError(f"the answer was split into {len(claims)} claims, more than the {CLAIM_LIMIT} the judge will check")
    return claims or [squash(answer)]


def _claim_number(entry):
    value = entry.get("claim")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_support(raw, claims, context):
    """One result per claim, in claim order, with the evidence verified against the context.

    A 'supported' verdict only stands when its evidence is real document text, is at least two words
    long, and shares a content word or number with the claim; otherwise it is downgraded to
    unsupported, so the judge cannot assert support it cannot show. A claim the reply skipped is
    'unchecked' (not guessed at). Raises JudgeOutputError when there is no results list at all.
    """
    data = _load_object(raw, "claim-checking")
    entries = data.get("results")
    if not isinstance(entries, list):
        raise JudgeOutputError("the claim-checking reply had no results list")
    by_number = {}
    for entry in entries:
        if isinstance(entry, dict) and _claim_number(entry) is not None:
            by_number.setdefault(_claim_number(entry), entry)
    results = []
    for number, claim in enumerate(claims, start=1):
        entry = by_number.get(number)
        if entry is None:
            results.append({"claim": claim, "verdict": "unchecked", "evidence": "", "reason": "the judge skipped this claim"})
            continue
        verdict = "supported" if entry.get("verdict") == "supported" else "unsupported"
        evidence = squash(entry.get("evidence") or "")
        reason = squash(entry.get("reason") or "")
        if verdict == "supported":
            if not evidence_in_context(evidence, context):
                verdict, reason = "unsupported", "the quoted evidence is not in the context"
            elif not evidence_relates_to_claim(claim, evidence):
                verdict, reason = "unsupported", "the quoted evidence does not relate to the claim"
        results.append({"claim": claim, "verdict": verdict, "evidence": evidence if verdict == "supported" else "", "reason": reason})
    return results


def faithfulness_verdict(results):
    """('pass' | 'fail' | 'error', explanation).

    Any unsupported claim fails the answer, even if other claims could not be checked: a known failure
    outranks an incomplete reply. Faithful only when every claim is supported."""
    unsupported = [r for r in results if r["verdict"] == "unsupported"]
    unchecked = [r for r in results if r["verdict"] == "unchecked"]
    summary = f"{sum(r['verdict'] == 'supported' for r in results)} of {len(results)} claims supported"
    if unsupported:
        detail = "; ".join(f"{r['claim']} ({r['reason'] or 'no reason given'})" for r in unsupported)
        return "fail", f"{summary}. Unsupported: {detail}"
    if unchecked:
        return "error", f"{summary}. The judge skipped {len(unchecked)} claim(s), so the answer could not be fully checked"
    return "pass", summary


def parse_relevance(raw):
    """('pass' | 'fail', explanation). Only a clear 'yes' passes: partly answering is not answering."""
    data = _load_object(raw, "relevance")
    addresses = str(data.get("addresses") or "").strip().lower()
    if addresses not in ("yes", "partly", "no"):
        raise JudgeOutputError("the relevance reply had no yes, partly or no verdict")
    reason = squash(data.get("reason") or "")
    asked = squash(data.get("asked") or "")
    return ("pass" if addresses == "yes" else "fail"), f"asked: {asked} | {addresses}: {reason}"
