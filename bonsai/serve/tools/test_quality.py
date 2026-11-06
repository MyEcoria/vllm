"""Unit tests for the pure parts of quality.py (no server)."""
import math
import urllib.error

import pytest

import quality


def fake_server(monkeypatch, down=False):
    """Patch post_json with a whitespace tokenizer (one word is one token)."""
    def post(url, headers, body):
        if down:
            raise urllib.error.URLError("server down")
        if url.endswith("/tokenize"):
            words = body["prompt"].split()
            return {"count": len(words), "tokens": words}
        return {"prompt": " ".join(body["tokens"])}
    monkeypatch.setattr(quality, "post_json", post)


def test_token_logprobs_skips_first_none():
    """The first prompt token has no logprob and is skipped."""
    pl = [None, {"5": {"logprob": -1.0}}, {"7": {"logprob": -3.0}}]
    assert quality.token_logprobs(pl) == [-1.0, -3.0]


def test_mean_nll():
    """NLL is the negated mean logprob."""
    assert quality.mean_nll([-1.0, -3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        quality.mean_nll([])


def test_segment_nll():
    """Each segment averages its own slice; empty slices are dropped."""
    lps = [-1.0] * 3 + [-2.0] * 3
    assert quality.segment_nll(lps, ((0, 3), (3, 6), (6, 9))) == [1.0, 2.0]
    assert math.isclose(quality.segment_nll(lps, ((0, 6),))[0], 1.5)


def test_insert_needle_and_found():
    """The needle lands on a line start and is detected in an answer."""
    text = "a\nb\nc\nd\n"
    out = quality.insert_needle(text, 4242, 0.5)
    assert "# The secret code is 4242.\n" in out
    assert out.replace("# The secret code is 4242.\n", "") == text
    assert quality.needle_found("it is 4242", 4242)
    assert not quality.needle_found("unknown", 4242)
    assert not quality.needle_found(None, 4242)


def test_chat_text_ignores_reasoning(monkeypatch):
    """A code that appears only in reasoning is not part of the answer."""
    reply = {"choices": [{"message": {"reasoning_content": "code is 4242",
                                      "content": "unknown"}}]}
    monkeypatch.setattr(quality, "post_json", lambda *a: reply)
    args = quality.parse_args(["needle"])
    text = quality.chat_text(args, {}, "prompt", 8)
    assert text == "unknown"
    assert not quality.needle_found(text, 4242)


def test_greedy_compare_rejects_count_mismatch():
    """A baseline with a different number of prompts is refused, not zipped."""
    with pytest.raises(SystemExit):
        quality.check_same_count(["a", "b"], ["a"])
    quality.check_same_count(["a"], ["b"])


def test_first_divergence():
    """Equal strings give None; otherwise the first differing index."""
    assert quality.first_divergence("abc", "abc") is None
    assert quality.first_divergence("abcd", "abxd") == 2
    assert quality.first_divergence("abc", "abcde") == 3


def test_match_ratio():
    """Identical strings match 1.0, different ones less."""
    assert quality.match_ratio("abc", "abc") == 1.0
    assert 0.0 <= quality.match_ratio("abcd", "abxd") < 1.0


def test_parse_args_validation():
    """Bad ports and non-positive token counts are rejected."""
    with pytest.raises(SystemExit):
        quality.parse_args(["--port", "0", "nll"])
    with pytest.raises(SystemExit):
        quality.parse_args(["needle", "--tokens", "-5"])
    assert quality.parse_args(["--port", "8001", "greedy"]).url.endswith(":8001")


def test_select_files_stable():
    """File selection is sorted, .py only, capped and order-independent."""
    a = "vllm/b.py\nvllm/a.py\nvllm/x.txt\nvllm/c.py\n"
    b = "vllm/c.py\nvllm/x.txt\nvllm/a.py\nvllm/b.py\n"
    assert quality.select_files(a) == ["vllm/a.py", "vllm/b.py", "vllm/c.py"]
    assert quality.select_files(a) == quality.select_files(b)
    many = "\n".join(f"vllm/{i:05d}.py" for i in range(quality.MAX_FILES + 5))
    assert len(quality.select_files(many)) == quality.MAX_FILES


def test_fit_text_and_json_after_subcommand():
    """Oversized text is trimmed, margin applies without count, --json works late."""
    assert quality.fit_text("x" * 200, 50, 100) == "x" * 100
    assert quality.fit_text("x" * 200, 50, 40) == "x" * 200
    assert len(quality.fit_text("x" * 400, 100, None)) < 400
    a = quality.parse_args(["greedy", "--json", "o.json"])
    assert a.json_path == "o.json"
    assert quality.parse_args(["--json", "p.json", "nll"]).json_path == "p.json"
    assert quality.parse_args(["nll"]).json_path is None


def test_trim_nll_text_cuts_to_exact_token_budget(monkeypatch):
    """Long text is cut to the budget via /tokenize + /detokenize, not sent whole."""
    fake_server(monkeypatch)
    text = " ".join(f"w{i}" for i in range(1000))
    args = quality.parse_args(["nll"])
    assert quality.trim_nll_text(args, {}, text, 100) == " ".join(text.split()[:100])


def test_trim_nll_text_keeps_short_text(monkeypatch):
    """Text already within the budget is returned unchanged."""
    fake_server(monkeypatch)
    args = quality.parse_args(["nll"])
    assert quality.trim_nll_text(args, {}, "a b c", 100) == "a b c"


def test_trim_nll_text_chars_fallback_with_margin(monkeypatch):
    """Without a server tokenizer, chars/4 is used with a 5% margin."""
    fake_server(monkeypatch, down=True)
    args = quality.parse_args(["nll"])
    assert len(quality.trim_nll_text(args, {}, "x" * 10000, 100)) == 380


def test_nll_budget_keeps_headroom_below_max_context():
    """The prompt budget stays 64 tokens below --max-context."""
    big = quality.parse_args(["nll", "--tokens", "200000"])
    assert quality.nll_budget(big) == 100000 - 64
    small = quality.parse_args(["nll", "--tokens", "500"])
    assert quality.nll_budget(small) == 500
    assert quality.nll_budget(quality.parse_args(["nll"])) == 100000 - 64
