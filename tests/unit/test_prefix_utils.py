from unittest import mock

from src.methods.prefix_utils import (
    build_conditional_texts,
    compute_conditional_prefix_length,
    compute_prefix_loss,
    extract_prefix,
    get_separator_token,
    process_prefix,
    select_prefix_candidates,
)


def make_model(max_model_len):
    model = mock.MagicMock()
    model.llm_engine.model_config.max_model_len = max_model_len
    return model


def make_tokenizer(tokens_per_shot):
    tokenizer = mock.MagicMock()
    tokenizer.encode.side_effect = lambda text: [0] * tokens_per_shot
    return tokenizer


class TestProcessPrefix:
    def test_pass_window_skips_check(self):
        prefix = ["shot1", "shot2"]
        result, num_shots = process_prefix(
            make_model(1), make_tokenizer(100), prefix,
            avg_length=100, pass_window=True, num_shots=2,
        )
        assert result == prefix
        assert num_shots == 2

    def test_all_shots_fit(self):
        prefix = ["shot1", "shot2"]
        # 2 shots * 10 tokens + avg_length 10 = 30 <= 100
        result, num_shots = process_prefix(
            make_model(100), make_tokenizer(10), prefix,
            avg_length=10, pass_window=False, num_shots=2,
        )
        assert result == prefix
        assert num_shots == 2

    def test_truncates_to_fitting_shots(self):
        prefix = ["shot1", "shot2", "shot3"]
        # avg_length 10 + 1 shot * 10 = 20 <= 25, 2 shots would be 30 > 25
        result, num_shots = process_prefix(
            make_model(25), make_tokenizer(10), prefix,
            avg_length=10, pass_window=False, num_shots=3,
        )
        assert num_shots == 1
        assert result == ["shot3"]

    def test_no_shot_fits_returns_empty(self):
        prefix = ["shot1", "shot2", "shot3"]
        # avg_length alone already exceeds max_model_len: no shot fits.
        # prefix[-0:] would wrongly return the whole prefix here.
        result, num_shots = process_prefix(
            make_model(50), make_tokenizer(100), prefix,
            avg_length=60, pass_window=False, num_shots=3,
        )
        assert num_shots == 0
        assert result == []


class TestExtractPrefix:
    def test_selects_requested_number_of_shots(self):
        texts = ["a", "b", "c", "d", "e"]
        result = extract_prefix(texts, num_shots=3)
        assert len(result) == 3
        assert set(result) <= set(texts)

    def test_caps_at_available_texts(self):
        texts = ["a", "b"]
        result = extract_prefix(texts, num_shots=10)
        assert sorted(result) == ["a", "b"]

    def test_does_not_mutate_input(self):
        texts = ["a", "b", "c"]
        extract_prefix(texts, num_shots=2)
        assert texts == ["a", "b", "c"]


def make_prefix_output(token_log_probs):
    """Build a mock RequestOutput whose prompt token IDs index their logprobs."""
    output = mock.MagicMock()
    output.prompt_token_ids = list(range(len(token_log_probs)))
    prompt_logprobs = []
    for token_id, value in enumerate(token_log_probs):
        if value is None:
            prompt_logprobs.append(None)
            continue
        entry = mock.MagicMock()
        entry.logprob = value
        prompt_logprobs.append({token_id: entry})
    output.prompt_logprobs = prompt_logprobs
    return output


class TestComputePrefixLoss:
    def test_excludes_prefix_tokens(self):
        # 4 tokens, exclude the first 2: loss = -mean([-2.0, -4.0]) = 3.0
        output = make_prefix_output([-1.0, -1.0, -2.0, -4.0])
        assert compute_prefix_loss(output, prefix_token_length=2) == 3.0

    def test_skips_none_logprobs(self):
        # First token has no logprob (None) and must be ignored.
        output = make_prefix_output([None, -2.0, -4.0])
        assert compute_prefix_loss(output, prefix_token_length=0) == 3.0

    def test_no_prefix_uses_all_tokens(self):
        output = make_prefix_output([-2.0, -4.0])
        assert compute_prefix_loss(output, prefix_token_length=0) == 3.0


class TestSeparatorAndSelfLeakagePrevention:
    def test_get_separator_token_bos_vs_bos_less(self):
        tok_bos = mock.MagicMock()
        tok_bos.bos_token = "<s>"
        assert get_separator_token(tok_bos) == "<s>"

        tok_no_bos = mock.MagicMock()
        tok_no_bos.bos_token = None
        assert get_separator_token(tok_no_bos) == "\n"

    def test_select_prefix_candidates_leaves_room_for_backfill(self):
        pool = ["n1", "n2", "n3"]
        selected = select_prefix_candidates(pool, num_shots=5)
        assert len(selected) == 2

    def test_build_conditional_texts_prevents_self_reference_with_backfill(self):
        texts = ["m1", "n1", "n2", "n3"]
        pool_texts = ["n1", "n2", "n3"]
        processed_prefix = ["n1", "n2"]

        cond_texts = build_conditional_texts(
            texts=texts,
            pool_texts=pool_texts,
            processed_prefix=processed_prefix,
            model=make_model(1000),
            tokenizer=make_tokenizer(5),
            avg_length=5,
            pass_window=False,
            sep_token="<s>",
        )

        # m1 and n3 get the base prefix "n1n2<s>..."
        assert cond_texts[0] == "n1n2<s>m1"
        assert cond_texts[3] == "n1n2<s>n3"
        # n1 and n2 never include themselves before the separator
        assert not cond_texts[1].split("<s>")[0].__contains__("n1")
        assert "n3" in cond_texts[1].split("<s>")[0]
        assert not cond_texts[2].split("<s>")[0].__contains__("n2")
        assert "n3" in cond_texts[2].split("<s>")[0]

    def test_build_conditional_texts_sole_candidate_falls_back_to_unprefixed(self):
        texts = ["m1", "n1"]
        cond_texts = build_conditional_texts(
            texts=texts,
            pool_texts=["n1"],
            processed_prefix=["n1"],
            model=make_model(1000),
            tokenizer=make_tokenizer(5),
            avg_length=5,
            pass_window=True,
            sep_token="<s>",
        )
        assert cond_texts[0] == "n1<s>m1"
        assert cond_texts[1] == "n1"


class TestComputeConditionalPrefixLength:
    @staticmethod
    def _ids_output(ids):
        out = mock.MagicMock()
        out.prompt_token_ids = list(ids)
        return out

    def test_bos_model_excludes_prefix_and_separator_bos(self):
        # Unconditional: [BOS=1, 10, 20, 30] -> 3 target tokens
        # Conditional: [BOS=1, 90, 91, BOS=1, 10, 20, 30] -> exclude first 4
        uncond = self._ids_output([1, 10, 20, 30])
        cond = self._ids_output([1, 90, 91, 1, 10, 20, 30])
        assert compute_conditional_prefix_length(cond, uncond, bos_token_id=1) == 4

    def test_bos_less_bpe_boundary_merge_does_not_drop_target_token(self):
        # Suppose prefix "foo " alone had 2 tokens [50, 51] and target "bar baz"
        # had [10, 20], but "foo bar baz" merged " " + "bar" into token 11:
        # cond_ids = [50, 11, 20]. Prefix length must be 1 (only [50]), not 2!
        uncond = self._ids_output([10, 20])
        cond = self._ids_output([50, 11, 20])
        assert compute_conditional_prefix_length(cond, uncond, bos_token_id=None) == 1
