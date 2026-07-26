from unittest import mock

from src.methods.recall import ReCaLLMethod


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


class TestReCaLLRun:
    def test_score_is_conditional_over_unconditional_loss(self):
        method = ReCaLLMethod({"num_shots": 1})
        texts = ["member text", "nonmember text"]
        labels = [1, 0]

        # Unconditional loss 3.0; conditional loss (first 2 tokens excluded) 3.0.
        uncond = [make_output([-2.0, -4.0]), make_output([-2.0, -4.0])]
        cond = [
            make_output([-1.0, -1.0, -2.0, -4.0]),
            make_output([-1.0, -1.0, -2.0, -4.0]),
        ]

        with (
            mock.patch(
                "src.methods.recall.process_prefix",
                return_value=(["prefix "], 1),
            ),
            mock.patch.object(method, "get_outputs", side_effect=[uncond, cond]),
        ):
            scores = method.run(
                texts,
                labels,
                mock.MagicMock(),
                make_tokenizer(prefix_token_length=2),
                mock.MagicMock(),
            )

        # conditional (3.0) / unconditional (3.0) ~= 1.0 for both samples.
        assert len(scores) == 2
        for score in scores:
            assert abs(score - 1.0) < 1e-6

    def test_num_shots_change_is_logged(self):
        method = ReCaLLMethod({"num_shots": 5})
        texts = ["member text", "nonmember text"]
        labels = [1, 0]
        uncond = [make_output([-2.0, -4.0]), make_output([-2.0, -4.0])]
        cond = [make_output([-2.0, -4.0]), make_output([-2.0, -4.0])]

        with (
            mock.patch(
                "src.methods.recall.process_prefix",
                return_value=(["prefix "], 1),  # 1 != requested 5 -> warning path
            ),
            mock.patch.object(method, "get_outputs", side_effect=[uncond, cond]),
            mock.patch("src.methods.recall.logging.warning") as warn,
        ):
            method.run(
                texts,
                labels,
                mock.MagicMock(),
                make_tokenizer(prefix_token_length=0),
                mock.MagicMock(),
            )

        warn.assert_called_once()

    def test_non_space_delimited_language_strips_spaces(self):
        method = ReCaLLMethod({"num_shots": 1})
        texts = ["mem ber", "non mem"]
        labels = [1, 0]
        uncond = [make_output([-2.0, -4.0]), make_output([-2.0, -4.0])]
        cond = [make_output([-2.0, -4.0]), make_output([-2.0, -4.0])]

        with (
            mock.patch(
                "src.methods.recall.process_prefix",
                return_value=(["pre fix"], 1),
            ),
            mock.patch.object(
                method, "get_outputs", side_effect=[uncond, cond]
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

        # Unconditional texts must have had their spaces stripped.
        unconditional_texts = get_outputs.call_args_list[0][0][0]
        assert unconditional_texts == ["member", "nonmem"]
