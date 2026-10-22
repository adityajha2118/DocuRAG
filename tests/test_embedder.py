"""Tests for src/embeddings/embedder.py"""

import pytest
from unittest.mock import patch, MagicMock
from src.embeddings.embedder import embed_texts

@patch("src.embeddings.embedder.get_embedder")
def test_embed_texts_single(mock_get_embedder):
    # Setup mock
    mock_model = MagicMock()
    # When passed a single string, we return one vector
    mock_model.encode.return_value = [0.1, 0.2, 0.3]
    mock_get_embedder.return_value = mock_model
    
    result = embed_texts("Hello world")
    assert isinstance(result, list)
    assert len(result) == 3  # The single returned vector
    assert result == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with("Hello world", batch_size=64, show_progress_bar=False, convert_to_numpy=False)

@patch("src.embeddings.embedder.get_embedder")
def test_embed_texts_batch(mock_get_embedder):
    # Setup mock
    mock_model = MagicMock()
    # When passed a list of strings, return a list of vectors
    mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_get_embedder.return_value = mock_model
    
    texts = ["Doc 1", "Doc 2"]
    result = embed_texts(texts, batch_size=32)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.assert_called_once_with(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=False)
