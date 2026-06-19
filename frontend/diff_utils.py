"""Word-level diff rendering for the Tailored Resume tab.

Produces HTML where removed words (from the original) are struck through in red and
added words (in the rewrite) are highlighted in green.
"""

from __future__ import annotations

import difflib
import html


def render_word_diff(original: str, rewritten: str) -> str:
    orig_words = original.split()
    new_words = rewritten.split()
    sm = difflib.SequenceMatcher(a=orig_words, b=new_words)

    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(" ".join(orig_words[i1:i2])))
        elif tag == "delete":
            parts.append(
                f'<span style="background:#fdd;color:#900;text-decoration:line-through;">'
                f'{html.escape(" ".join(orig_words[i1:i2]))}</span>'
            )
        elif tag == "insert":
            parts.append(
                f'<span style="background:#dfd;color:#060;">'
                f'{html.escape(" ".join(new_words[j1:j2]))}</span>'
            )
        elif tag == "replace":
            parts.append(
                f'<span style="background:#fdd;color:#900;text-decoration:line-through;">'
                f'{html.escape(" ".join(orig_words[i1:i2]))}</span> '
                f'<span style="background:#dfd;color:#060;">'
                f'{html.escape(" ".join(new_words[j1:j2]))}</span>'
            )
    return " ".join(parts)
