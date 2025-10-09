import pytest
from unittest.mock import MagicMock, patch, ANY
from PIL import Image
from live_photo_commentary.describer import Describer


class MockDescriber:
    """Mock describer for testing prompt building logic without actual model calls."""

    @staticmethod
    def create_describer(**kwargs):
        """Create a mock describer that captures prompt_model calls."""
        with patch("live_photo_commentary.remote_describers.gemini.genai.Client"):
            describer = Describer(
                local=False,
                provider="gemini",
                **kwargs
            )

            # Track all calls to prompt_model
            calls = []
            original_prompt_model = describer.prompt_model

            def mock_prompt_model(user_prompt, images=None, system_prompt=None):
                calls.append({
                    "user_prompt": user_prompt,
                    "images": images,
                    "system_prompt": system_prompt,
                    "num_images": len(images) if images else 0
                })
                # Return a fake response
                return f"Response {len(calls)}"

            describer.prompt_model = mock_prompt_model
            describer._calls = calls

            return describer


def test_first_prompt_no_previous_image():
    """Test that first_prompt is used when no previous image is provided."""
    describer = MockDescriber.create_describer(
        first_prompt="FIRST_PROMPT",
        prompt="REGULAR_PROMPT",
        system_prompt="SYSTEM"
    )

    img = Image.new("RGB", (100, 100))
    describer(img)

    assert len(describer._calls) == 1
    assert describer._calls[0] == {
        "user_prompt": "FIRST_PROMPT",
        "images": ANY,
        "system_prompt": "SYSTEM",
        "num_images": 1
    }


def test_regular_prompt_with_previous_image_no_history():
    """Test that regular prompt is used with previous image but no history tracking."""
    describer = MockDescriber.create_describer(
        first_prompt="FIRST_PROMPT",
        prompt="REGULAR_PROMPT",
        max_history_size=False  # No history tracking
    )

    img1 = Image.new("RGB", (100, 100))
    img2 = Image.new("RGB", (100, 100))

    describer(img1)
    describer(img2, previous_image=img1)

    assert len(describer._calls) == 2
    assert describer._calls[0] == {
        "user_prompt": "FIRST_PROMPT",
        "images": ANY,
        "system_prompt": ANY,
        "num_images": 1
    }
    assert describer._calls[1] == {
        "user_prompt": "REGULAR_PROMPT",
        "images": ANY,
        "system_prompt": ANY,
        "num_images": 2
    }


def test_prompt_with_history():
    """Test that history is included in prompt when max_history_size is set."""
    describer = MockDescriber.create_describer(
        first_prompt="FIRST_PROMPT",
        prompt="REGULAR_PROMPT",
        history_prompt="HISTORY_HEADER",
        max_history_size=5
    )

    img1 = Image.new("RGB", (100, 100))
    img2 = Image.new("RGB", (100, 100))
    img3 = Image.new("RGB", (100, 100))

    # First call - no history
    describer(img1)
    assert describer._calls[0] == {
        "user_prompt": "FIRST_PROMPT",
        "images": ANY,
        "system_prompt": ANY,
        "num_images": 1
    }

    # Second call - should include history
    describer(img2, previous_image=img1)
    expected_prompt = '\n'.join([
        "HISTORY_HEADER",
        "Response 1",
        "---",
        "REGULAR_PROMPT"
    ])
    assert describer._calls[1] == {
        "user_prompt": expected_prompt,
        "images": ANY,
        "system_prompt": ANY,
        "num_images": 2
    }

    # Third call - should include both previous responses in history
    describer(img3, previous_image=img2)
    expected_prompt = '\n'.join([
        "HISTORY_HEADER",
        "Response 1",
        "Response 2",
        "---",
        "REGULAR_PROMPT"
    ])
    assert describer._calls[2] == {
        "user_prompt": expected_prompt,
        "images": ANY,
        "system_prompt": ANY,
        "num_images": 2
    }


def test_history_compaction_triggered():
    """Test that history compaction is triggered when max_history_size is reached."""
    describer = MockDescriber.create_describer(
        first_prompt="FIRST",
        prompt="REGULAR",
        history_prompt="HISTORY",
        compact_prompt="COMPACT",
        max_history_size=3,
        min_history_size=1
    )

    img = Image.new("RGB", (100, 100))

    # Build up history to max
    describer(img)  # Response 1
    describer(img, previous_image=img)  # Response 2
    describer(img, previous_image=img)  # Response 3

    assert len(describer.history) == 3

    # Next call should trigger compaction
    describer(img, previous_image=img)

    # Check that compact_history was called
    # We should see a call with the compact_prompt
    compact_calls = [c for c in describer._calls if "COMPACT" in c["user_prompt"]]
    assert len(compact_calls) == 1

    # The compact call should contain the old history items to be compacted
    compact_prompt = compact_calls[0]["user_prompt"]
    assert "Response 1" in compact_prompt
    assert "Response 2" in compact_prompt
    # Response 3 should be preserved, not compacted
    assert "Response 3" not in compact_prompt

    # History should now have compacted version + preserved items
    assert len(describer.history) == 3  # Still at max


def test_history_compaction_no_min_history():
    """Test history compaction when min_history_size is not set."""
    describer = MockDescriber.create_describer(
        first_prompt="FIRST",
        prompt="REGULAR",
        compact_prompt="COMPACT",
        max_history_size=2,
        min_history_size=False  # No minimum to preserve
    )

    img = Image.new("RGB", (100, 100))

    # Build up to max
    describer(img)  # Response 1
    describer(img, previous_image=img)  # Response 2

    # Trigger compaction
    describer(img, previous_image=img)

    # All old history should be compacted
    compact_calls = [c for c in describer._calls if "COMPACT" in c["user_prompt"]]
    assert len(compact_calls) == 1

    compact_prompt = compact_calls[0]["user_prompt"]
    assert "Response 1" in compact_prompt
    assert "Response 2" in compact_prompt


def test_reset_clears_history():
    """Test that reset() clears the history."""
    describer = MockDescriber.create_describer(
        max_history_size=5
    )

    img = Image.new("RGB", (100, 100))

    describer(img)
    describer(img, previous_image=img)

    assert len(describer.history) == 2

    describer.reset()

    assert len(describer.history) == 0


def test_compact_history_without_system_prompt():
    """Test that compact_history doesn't pass system_prompt to prompt_model."""
    describer = MockDescriber.create_describer(
        compact_prompt="COMPACT",
        max_history_size=2,
        system_prompt="SYSTEM"
    )

    img = Image.new("RGB", (100, 100))

    # Build up to max
    describer(img)
    describer(img, previous_image=img)

    # Trigger compaction
    describer(img, previous_image=img)

    # Find the compact call
    compact_calls = [c for c in describer._calls if "COMPACT" in c["user_prompt"]]
    assert len(compact_calls) == 1

    # Compact call should NOT have system_prompt
    assert compact_calls[0] == {
        "user_prompt": ANY,
        "images": ANY,
        "system_prompt": None,
        "num_images": ANY
    }


def test_custom_prompts():
    """Test that custom prompts are used correctly."""
    describer = MockDescriber.create_describer(
        first_prompt="CUSTOM_FIRST",
        prompt="CUSTOM_REGULAR",
        history_prompt="CUSTOM_HISTORY",
        max_history_size=3
    )

    img = Image.new("RGB", (100, 100))

    describer(img)
    assert describer._calls[0] == {
        "user_prompt": "CUSTOM_FIRST",
        "images": ANY,
        "system_prompt": ANY,
        "num_images": 1
    }

    describer(img, previous_image=img)
    # Check that the user_prompt contains both custom history and regular prompts
    user_prompt = describer._calls[1]["user_prompt"]
    assert "CUSTOM_HISTORY" in user_prompt
    assert "CUSTOM_REGULAR" in user_prompt
