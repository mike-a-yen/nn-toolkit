import pytest

from nn_toolkit.token_store import TokenStore
from nn_toolkit.vocab import Vocab


@pytest.fixture
def token_store():
    return TokenStore()


@pytest.fixture
def vocab(token_store):
    return Vocab(token_store)


def test_init(vocab):
    assert vocab.size == 4


def test_add_token(vocab):
    token = 'hello'
    vocab.add_token(token)
    assert vocab.size == 5
    assert vocab[token] == 4
    assert vocab.int_to_token[4] == token
