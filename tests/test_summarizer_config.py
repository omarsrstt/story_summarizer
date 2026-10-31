"""Tests for prompt-set resolution in the transformers path.

Prompts were moved into a shared ``prompts`` block so the GGUF and transformers
backends, and every model that opts in, are driven by identical wording. A
regression here would silently drop the system prompt, which is what keeps
recaps faithful, so it is worth pinning down.
"""

import pytest

from summarizer import resolve_entry


CONFIG = {
    "prompts": {
        "recap_faithful": {
            "system_prompt": "Recap faithfully.",
            "prompt_template": "Excerpt:\n{chapter}\n\nRecap:",
            "meta_prompt_template": "Merge {summaries}",
        }
    },
    "model_config": {
        "shared": {"prompt_set": "recap_faithful", "chunk_size": 12000},
        "inline": {"system_prompt": "Inline prompt.", "chunk_size": 7000},
    },
}


def test_shared_prompt_set_is_merged_into_the_entry():
    entry = resolve_entry(CONFIG, "shared")

    assert entry["system_prompt"] == "Recap faithfully."
    assert entry["prompt_template"] == "Excerpt:\n{chapter}\n\nRecap:"
    assert entry["meta_prompt_template"] == "Merge {summaries}"


def test_per_model_settings_win_over_the_shared_prompt_set():
    entry = resolve_entry(CONFIG, "shared")
    assert entry["chunk_size"] == 12000


def test_entry_can_override_one_field_of_a_shared_prompt_set():
    """A model may share a prompt set but change, say, its template only."""
    config = {
        "prompts": CONFIG["prompts"],
        "model_config": {
            "custom": {
                "prompt_set": "recap_faithful",
                "prompt_template": "Custom: {chapter}",
            }
        },
    }
    entry = resolve_entry(config, "custom")

    assert entry["system_prompt"] == "Recap faithfully."
    assert entry["prompt_template"] == "Custom: {chapter}"


def test_inline_prompts_still_work():
    entry = resolve_entry(CONFIG, "inline")
    assert entry["system_prompt"] == "Inline prompt."
    assert entry["chunk_size"] == 7000


def test_unknown_model_names_the_configured_ones():
    with pytest.raises(KeyError) as err:
        resolve_entry(CONFIG, "typo")

    message = str(err.value)
    assert "typo" in message
    assert "shared" in message
    assert "inline" in message


def test_unknown_prompt_set_is_reported_clearly():
    config = {"prompts": {}, "model_config": {"m": {"prompt_set": "missing"}}}
    with pytest.raises(KeyError) as err:
        resolve_entry(config, "m")
    assert "missing" in str(err.value)


def test_resolution_does_not_mutate_the_config():
    original = {
        "prompts": CONFIG["prompts"],
        "model_config": {"shared": {"prompt_set": "recap_faithful"}},
    }
    resolve_entry(original, "shared")
    assert original["model_config"]["shared"] == {"prompt_set": "recap_faithful"}
