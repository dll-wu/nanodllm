from nanodllm.data import ByteTokenizer


def test_byte_tokenizer_encode_is_content_only():
    tok = ByteTokenizer()
    assert tok.encode("abc") == [97, 98, 99]


def test_byte_tokenizer_special_ids():
    tok = ByteTokenizer()
    assert tok.get_bos_token_id() == 256
    assert tok.get_mask_token_id() == 257
    assert tok.get_eos_token_id() == 258
    assert tok.get_pad_token_id() == 259
    assert tok.get_eos_token_id() not in tok.get_generation_blocked_token_ids()


def test_byte_decode_stops_at_eos():
    tok = ByteTokenizer()
    ids = [97, 98, tok.get_eos_token_id(), 99]
    assert tok.decode(ids) == "ab"
