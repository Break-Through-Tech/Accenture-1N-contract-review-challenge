import re
import unicodedata

import pandas as pd

# our list of little cleanup jobs to do on the text, one after another
_CLEAN_STEPS = [
    (re.compile(r"(?i)\bpage \d+ of \d+\b"), lambda m: " "),   # strip leftover PDF page-number boilerplate, wherever it sits
    (re.compile(r"[ \t]+"), lambda m: " "),                    # collapse runs of spaces/tabs to one space
    (re.compile(r" *\n *"), lambda m: "\n"),                   # drop spaces hugging newlines
    (re.compile(r"\n{3,}"), lambda m: "\n\n"),                 # collapse 3+ newlines (2+ blank lines) down to one blank line
]


def _apply_step(s, origin, pattern, repl_fn):
    """Cleans up one small thing in the text (like squishing extra spaces) and while doing that, keeps a
    note for every letter saying 'this came from that spot in the original text',
    so that we never lose track of where anything came from."""
    out_chars, out_origin, last_end = [], [], 0
    for m in pattern.finditer(s):
        start, end = m.span()
        out_chars.append(s[last_end:start])                     # grab the untouched text before this match
        out_origin.extend(origin[last_end:start])               # keep its original tracking as-is, nothing changed here
        rep = repl_fn(m)                                        # figure out what we're swapping this match for
        out_chars.append(rep)                                   # tack on the replacement
        out_origin.extend([origin[start]] * len(rep))           # just say this replacement "came from" the start of the match
        last_end = end                                          # remember where we left off
    out_chars.append(s[last_end:])                              # don't forget whatever's left after the last match
    out_origin.extend(origin[last_end:])                        # same for its tracking
    return "".join(out_chars), out_origin


def clean_text_with_map(raw: str):
    """Cleans up the text, but also hands back a note saying where every cleaned letter came from."""
    if not isinstance(raw, str):
        return raw, []                                          # nothing to clean so just return

    # NFKC per character (not on the whole string) so multi-char expansions like ™ -> "TM" stay trackable
    chars, origin = [], []
    for i, c in enumerate(raw):
        nc = unicodedata.normalize("NFKC", c)                   # swap this one letter for its "plain" version if it has one
        chars.append(nc)
        origin.extend([i] * len(nc))                            # note down "this came from spot i" for every letter it turned into
    s = "".join(chars)                                          # glue all the letters back into one string

    for pattern, repl_fn in _CLEAN_STEPS:                       # run through list of cleanup jobs, one by one
        s, origin = _apply_step(s, origin, pattern, repl_fn)

    lstripped = len(s) - len(s.lstrip())                        # how much empty space is sitting at the very start
    end = len(s.rstrip())                                       # where the real text stops (before trailing empty space)
    return s[lstripped:end], origin[lstripped:end]              # trim that empty space off both the text and its notes, and return both


def build_reverse_map(raw_len, clean_to_raw, clean_len):
    """For every spot in the original text, figures out where that spot ended up in the cleaned text."""
    reverse = [-1] * (raw_len + 1)                              # one answer-slot per raw spot (plus one extra for "the very end"), -1 = "not found yet"
    for j, r in enumerate(clean_to_raw):
        if reverse[r] == -1:                                    # if we haven't already answered for this raw spot...
            reverse[r] = j                                      # ...say "raw spot r shows up at cleaned position j"
    next_val = clean_len                                        # if we walk off the end with nothing found, just point to the very end
    for r in range(raw_len, -1, -1):
        if reverse[r] == -1:                                    # this raw spot got deleted during cleaning, never showed up above
            reverse[r] = next_val                               # so just point it to the nearest surviving spot after it
        else:
            next_val = reverse[r]                               # this spot DID survive, remember it as the "nearest surviving spot" for the next ones back
    return reverse


def relocate_answer(row, title_to_reverse_map):
    """Find exactly where the answer ended up in the cleaned text, via the raw-to-clean offset map (no guessing)."""
    if row["is_impossible"]:
        return pd.Series({"annotation_text": None, "annotation_start": None})

    reverse = title_to_reverse_map[row["title"]]
    raw_start = int(row["answer_start"])
    raw_end = raw_start + len(row["answer_text"])
    clean_start = reverse[raw_start]
    clean_end = reverse[raw_end]

    return pd.Series({
        "annotation_text": row["text"][clean_start:clean_end],
        "annotation_start": clean_start,
    })
