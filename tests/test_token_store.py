import pytest

from nn_toolkit.token_store import TokenStore


@pytest.fixture
def token_store():
    return TokenStore()


def test_init(token_store):
    assert token_store.size == 4
    assert token_store.pad_idx == 0
    assert token_store.unk_idx == 1


def test_add_token(token_store):
    token_store.add_token('hello')
    assert token_store.size == 5
    assert token_store['hello'] == 4
    assert token_store.int_to_token[4] == 'hello'
