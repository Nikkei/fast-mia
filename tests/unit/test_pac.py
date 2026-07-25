from unittest import mock

import pytest

from src.methods.pac import (
    PACMethod,
    calculate_polarized_distance,
    eda,
    random_swap,
)


def make_output(token_log_probs):
    """Build a mock RequestOutput whose prompt token IDs index their logprobs."""
    output = mock.MagicMock()
    output.prompt_token_ids = list(range(len(token_log_probs)))
    prompt_logprobs = []
    for token_id, value in enumerate(token_log_probs):
        entry = mock.MagicMock()
        entry.logprob = value
        prompt_logprobs.append({token_id: entry})
    output.prompt_logprobs = prompt_logprobs
    return output


class TestPACValidation:
    def test_default_params_valid(self):
        method = PACMethod({})
        assert method.alpha == 0.3
        assert method.N == 5

    def test_alpha_zero_rejected(self):
        with pytest.raises(ValueError, match="'alpha' > 0"):
            PACMethod({"alpha": 0})

    def test_alpha_negative_rejected(self):
        with pytest.raises(ValueError, match="'alpha' > 0"):
            PACMethod({"alpha": -0.1})

    def test_n_zero_rejected(self):
        with pytest.raises(ValueError, match="'N' >= 1"):
            PACMethod({"N": 0})


class TestRandomSwap:
    def test_preserves_multiset_of_words(self):
        words = ["a", "b", "c", "d"]
        swapped = random_swap(words, n=2)
        assert sorted(swapped) == sorted(words)

    def test_does_not_mutate_original(self):
        words = ["a", "b", "c", "d"]
        random_swap(words, n=3)
        assert words == ["a", "b", "c", "d"]

    def test_single_word_is_noop(self):
        # swap_word bails out after failing to find a distinct index.
        assert random_swap(["only"], n=1) == ["only"]


class TestEda:
    def test_generates_requested_number_of_augmentations(self):
        augmented = eda("the quick brown fox", alpha=0.3, num_aug=5)
        assert len(augmented) == 5
        for sentence in augmented:
            assert sorted(sentence.split(" ")) == sorted("the quick brown fox".split(" "))


class TestCalculatePolarizedDistance:
    def test_far_minus_local_region(self):
        # 10 values: local region (lowest 30% -> 3), far region (highest 5% -> 1).
        probs = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        # local = mean([-5, -4, -3]) = -4.0, far = mean([4.0]) = 4.0
        assert calculate_polarized_distance(probs) == pytest.approx(8.0)

    def test_regions_have_at_least_one_element(self):
        # Small lists still yield a finite distance (max(int(...), 1)).
        result = calculate_polarized_distance([-1.0, 1.0])
        assert result == pytest.approx(2.0)


class TestPACProcessOutput:
    def test_delegates_to_polarized_distance(self):
        method = PACMethod({})
        probs = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        output = make_output(probs)
        assert method.process_output(output) == pytest.approx(8.0)


class TestPACRun:
    def test_calibrated_and_negated_scores(self):
        method = PACMethod({"alpha": 0.3, "N": 2})
        texts = ["the quick brown fox jumps"]

        probs = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        original_output = make_output(probs)  # polarized distance = 8.0

        def fake_get_outputs(input_texts, *args, **kwargs):
            # One entry per input text (originals or the N augmentations).
            return [make_output(probs) for _ in input_texts]

        with mock.patch.object(
            method, "get_outputs", side_effect=fake_get_outputs
        ):
            scores = method.run(texts, mock.MagicMock(), mock.MagicMock())

        # original PD (8.0) - mean augmented PD (8.0) = 0.0, negated -> 0.0
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.0)
