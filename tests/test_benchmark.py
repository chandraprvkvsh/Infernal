import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import re
from infernal import InfernalLLM

@pytest.fixture
def infernal_instance(tmp_path):
    """Fixture for InfernalLLM instance with mocked models dir."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    infernal = InfernalLLM(str(models_dir))
    infernal.config["models"]["test_model"] = {"filename": "test_model.gguf"}
    (models_dir / "test_model.gguf").touch()
    return infernal

@pytest.fixture
def mock_llama():
    """Mock Llama class to simulate inference without real model loading."""
    mock = MagicMock()
    mock.tokenize.side_effect = lambda x, *args: list(range(len(x)))
    return mock

def test_benchmark_single_prompt_no_repeat(infernal_instance, mock_llama, capsys):
    """Test benchmark with single prompt, no repeat: Verify metric calculations."""
    with (
        patch("infernal.Llama", return_value=mock_llama),
        patch("time.perf_counter", side_effect=[0.0, 1.0, 3.0]),
        patch("psutil.Process") as mock_process
    ):
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        mock_llama.return_value = iter([
            {"choices": [{"text": "token1"}]},
            {"choices": [{"text": "token2"}]}
        ])

        infernal_instance.benchmark_model("test_model", prompts=["Hello"], repeat=1)

        captured = capsys.readouterr().out

        assert re.search(r" Time to first token: \d+\.\d{3} seconds", captured)
        assert re.search(r" Generation time \(After-TTFT\): \d+\.\d{3} seconds", captured)
        assert re.search(r" Throughput \(generated tok/sec\): \d+\.\d{2}", captured)
        assert re.search(r" Total tokens/sec: \d+\.\d{2}", captured)
        assert re.search(r" Peak memory usage: 100\.00 MB", captured)
        assert "Averages across all runs" not in captured

def test_benchmark_with_repeat(infernal_instance, mock_llama, capsys):
    """Test benchmark with repeat: Verify averaging across runs."""
    with (
        patch("infernal.Llama", return_value=mock_llama),
        patch("time.perf_counter", side_effect=[0, 1, 2, 0, 1.5, 2.5]),
        patch("psutil.Process") as mock_process
    ):
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_llama.return_value = iter([{"choices": [{"text": "resp"}]}])

        infernal_instance.benchmark_model("test_model", prompts=["Hello"], repeat=2)

        captured = capsys.readouterr().out

        assert "Iteration 1/2" in captured
        assert "Iteration 2/2" in captured
        assert re.search(r" Avg Time to first token: \d+\.\d{3} seconds", captured)
        assert re.search(r" Avg throughput \(generated tok/sec\): \d+\.\d{2}", captured)
        assert re.search(r" Avg total tokens/sec: \d+\.\d{2}", captured)
        assert re.search(r" Avg Peak memory usage: 100\.00 MB", captured)

def test_benchmark_multiple_prompts(infernal_instance, mock_llama, capsys):
    """Test benchmark with multiple prompts."""
    with (
        patch("infernal.Llama", return_value=mock_llama),
        patch("time.perf_counter", side_effect=[0,1,2] * 2),
        patch("psutil.Process") as mock_process
    ):
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_llama.return_value = iter([{"choices": [{"text": "resp"}]}])

        prompts = ["Prompt1", "Prompt2"]
        infernal_instance.benchmark_model("test_model", prompts, repeat=1)

        captured = capsys.readouterr().out

        assert "Benchmarking prompt 1/2" in captured
        assert "Benchmarking prompt 2/2" in captured
        assert captured.count("Time to first token") == 3

def test_benchmark_model_not_found(infernal_instance, capsys):
    """Test error handling: Model not found."""
    infernal_instance.benchmark_model("non_existent_model", prompts=["Hello"], repeat=1)

    captured = capsys.readouterr().out

    assert "Model 'non_existent_model' not found" in captured
    assert "Time to first token" not in captured
