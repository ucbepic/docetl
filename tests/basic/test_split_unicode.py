import pytest
import tiktoken

from docetl.frame import from_list
from docetl.operations.split import SplitOperation


def split_text(runner, text, num_tokens, model):
    operation = SplitOperation(
        runner,
        {
            "name": "split_unicode",
            "type": "split",
            "split_key": "text",
            "method": "token_count",
            "method_kwargs": {"num_tokens": num_tokens, "model": model},
        },
        model,
        1,
    )
    result, cost = operation.execute([{"text": text, "source": "document"}])
    assert cost == 0
    return result


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4"])
@pytest.mark.parametrize("num_tokens", [4, 5])
@pytest.mark.parametrize(
    "text", ["hello 🚀 world", "a 𠮷 b", "नमस्ते दुनिया", "café 😀 文書"]
)
def test_token_split_preserves_unicode(runner, model, num_tokens, text):
    text = (text + " ") * 6
    result = split_text(runner, text, num_tokens, model)
    chunks = [row["text_chunk"] for row in result]
    encoder = tiktoken.encoding_for_model(model)

    assert "".join(chunks) == text
    assert all(chunk and len(encoder.encode(chunk)) <= num_tokens for chunk in chunks)
    assert [row["split_unicode_chunk_num"] for row in result] == list(
        range(1, len(result) + 1)
    )
    assert len({row["split_unicode_id"] for row in result}) == 1
    assert all(row["text"] == text and row["source"] == "document" for row in result)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4"])
def test_token_limit_too_small_for_character(runner, model):
    with pytest.raises(ValueError, match="num_tokens.*Unicode character"):
        split_text(runner, "intro 🚀", 1, model)


@pytest.mark.parametrize("text", ["", "ASCII text only"])
def test_token_split_empty_and_ascii(runner, text):
    result = split_text(runner, text, 1, "gpt-4o")
    assert "".join(row["text_chunk"] for row in result) == text
    if not text:
        assert result == []


@pytest.mark.parametrize(
    "num_tokens,text",
    [(2, "hello 🚀 world"), (500, "a " * 499 + "🚀 end")],
    ids=["two-token-limit", "five-hundred-token-limit"],
)
def test_frame_token_split_preserves_unicode(num_tokens, text):
    result = (
        from_list([{"text": text}])
        .split(
            split_key="text",
            method="token_count",
            method_kwargs={"num_tokens": num_tokens, "model": "gpt-4o"},
        )
        .collect()
    )
    assert "".join(row["text_chunk"] for row in result) == text
