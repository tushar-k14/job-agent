"""Tests for frontend/diff_utils.py — word-level diff rendering."""

from __future__ import annotations

from frontend.diff_utils import render_word_diff


class TestRenderWordDiff:
    def test_identical_strings(self):
        html = render_word_diff("led a team", "led a team")
        assert "led" in html
        assert "background" not in html  # no highlights when nothing changed

    def test_insertion(self):
        html = render_word_diff("led a team", "led a high-performing team")
        assert "#dfd" in html  # green for insertion
        assert "high-performing" in html
        assert "#fdd" not in html  # nothing deleted

    def test_deletion(self):
        html = render_word_diff("led a large team", "led a team")
        assert "#fdd" in html  # red for deletion
        assert "large" in html
        assert "#dfd" not in html

    def test_replacement(self):
        html = render_word_diff("managed 3 people", "led 3 engineers")
        assert "#fdd" in html
        assert "#dfd" in html
        assert "managed" in html  # original shown struck through
        assert "led" in html      # replacement shown in green

    def test_empty_original(self):
        html = render_word_diff("", "new content")
        assert "new" in html
        assert "#dfd" in html

    def test_empty_rewritten(self):
        html = render_word_diff("old content", "")
        assert "old" in html
        assert "#fdd" in html

    def test_both_empty(self):
        html = render_word_diff("", "")
        assert html == ""

    def test_html_escaping(self):
        # Angle brackets in the input must not create raw HTML tags
        html = render_word_diff("a <b> c", "a <b> c")
        assert "<b>" not in html
        assert "&lt;b&gt;" in html

    def test_multiword_change(self):
        orig = "responsible for maintaining legacy PHP systems"
        new =  "owned maintenance and modernisation of backend systems"
        html = render_word_diff(orig, new)
        # Should have both insertion and deletion markers
        assert "#dfd" in html
        assert "#fdd" in html
