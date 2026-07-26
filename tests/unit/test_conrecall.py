from unittest import mock

from src.methods.conrecall import CONReCaLLMethod


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


def make_tokenizer(prefix_token_length):
    tokenizer = mock.MagicMock()
    tokenizer.encode.side_effect = lambda text: [0] * 4
    tokenizer.bos_token = "<s>"
    tokenizer.return_value.input_ids.size.return_value = prefix_token_length
    return tokenizer


class TestConReCaLLInit:
    def test_defaults(self):
        method = CONReCaLLMethod({})
        assert method.num_shots == 10
        assert method.gamma == 0.5


class TestConReCaLLRun:
    def test_score_combines_member_and_nonmember_losses(self):
        method = CONReCaLLMethod({"num_shots": 1, "gamma": 0.5})
        texts = ["member text", "nonmember text"]
        labels = [1, 0]

        # Each get_outputs call returns two per-text outputs. There are three
        # calls: unconditional, member-conditional, nonmember-conditional.
        def two(values):
            return [make_output(values), make_output(values)]

        uncond = two([-2.0, -4.0])  # loss 3.0
        member_cond = two([-2.0, -4.0])  # loss 3.0
        nonmember_cond = two([-2.0, -4.0])  # loss 3.0

        with (
            mock.patch(
                "src.methods.conrecall.process_prefix",
                return_value=(["prefix "], 1),
            ),
            mock.patch.object(
                method,
                "get_outputs",
                side_effect=[uncond, member_cond, nonmember_cond],
            ),
        ):
            scores = method.run(
                texts,
                labels,
                mock.MagicMock(),
                make_tokenizer(prefix_token_length=0),
                mock.MagicMock(),
            )

        # (nonmember 3.0 - 0.5 * member 3.0) / unconditional 3.0 = 0.5
        assert len(scores) == 2
        for score in scores:
            assert abs(score - 0.5) < 1e-6

    def test_num_shots_change_warns_for_both_prefixes(self):
        method = CONReCaLLMethod({"num_shots": 5})
        texts = ["member text", "nonmember text"]
        labels = [1, 0]

        def two(values):
            return [make_output(values), make_output(values)]

        outputs = [two([-2.0, -4.0]), two([-2.0, -4.0]), two([-2.0, -4.0])]

        with (
            mock.patch(
                "src.methods.conrecall.process_prefix",
                return_value=(["prefix "], 1),  # 1 != requested 5
            ),
            mock.patch.object(method, "get_outputs", side_effect=outputs),
            mock.patch("src.methods.conrecall.logging.warning") as warn,
        ):
            method.run(
                texts,
                labels,
                mock.MagicMock(),
                make_tokenizer(prefix_token_length=0),
                mock.MagicMock(),
            )

        # Both the member and non-member prefixes trigger a warning.
        assert warn.call_count == 2

    def test_non_space_delimited_language_strips_spaces(self):
        method = CONReCaLLMethod({"num_shots": 1})
        texts = ["mem ber", "non mem"]
        labels = [1, 0]

        def two(values):
            return [make_output(values), make_output(values)]

        outputs = [two([-2.0, -4.0]), two([-2.0, -4.0]), two([-2.0, -4.0])]

        with (
            mock.patch(
                "src.methods.conrecall.process_prefix",
                return_value=(["pre fix"], 1),
            ),
            mock.patch.object(
                method, "get_outputs", side_effect=outputs
            ) as get_outputs,
        ):
            method.run(
                texts,
                labels,
                mock.MagicMock(),
                make_tokenizer(prefix_token_length=0),
                mock.MagicMock(),
                data_config={"space_delimited_language": False},
            )

        unconditional_texts = get_outputs.call_args_list[0][0][0]
        assert unconditional_texts == ["member", "nonmem"]
