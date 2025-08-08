import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from infernal import InfernalLLM

@pytest.fixture
def infernal_with_model(tmp_path):
    """Create InfernalLLM instance with a test model"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    infernal = InfernalLLM(str(models_dir))
    infernal.config["models"]["test_model"] = {
        "filename": "test_model.gguf",
        "repo_id": "test/repo",
        "size": 1024000
    }
    
    # Create mock model file
    model_file = models_dir / "test_model.gguf"
    model_file.touch()
    
    return infernal

class TestInference:
    @patch('infernal.Llama')
    def test_run_inference_single_prompt(self, mock_llama_class, infernal_with_model, capsys):
        """Test running inference with a single prompt"""
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        mock_llm.return_value = {
            "choices": [{"text": "Hello! How can I help you?"}]
        }
        
        infernal_with_model.run_inference("test_model", "Hello")
        
        mock_llama_class.assert_called_once()
        mock_llm.assert_called_once()
        
        captured = capsys.readouterr()
        assert "Loading test_model" in captured.out

    @patch('infernal.Llama')
    def test_run_inference_model_not_found(self, mock_llama_class, infernal_with_model, capsys):
        """Test running inference with non-existent model"""
        infernal_with_model.run_inference("nonexistent", "Hello")
        
        mock_llama_class.assert_not_called()
        
        captured = capsys.readouterr()
        assert "Model nonexistent not found" in captured.out

    @patch('infernal.Llama')
    @patch('infernal.Prompt.ask')
    def test_run_inference_no_prompt(self, mock_prompt, mock_llama_class, infernal_with_model):
        """Test running inference without providing prompt"""
        mock_prompt.return_value = "Test prompt"
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        mock_llm.return_value = {"choices": [{"text": "Response"}]}
        
        infernal_with_model.run_inference("test_model")
        
        mock_prompt.assert_called_once_with("Enter prompt")
        mock_llm.assert_called_once()

    @patch('infernal.Llama')
    def test_run_inference_llama_error(self, mock_llama_class, infernal_with_model, capsys):
        """Test handling Llama initialization error"""
        mock_llama_class.side_effect = Exception("Failed to load model")
        
        infernal_with_model.run_inference("test_model", "Hello")
        
        captured = capsys.readouterr()
        assert "Inference error" in captured.out

    @patch('infernal.Llama')
    @patch('infernal.Prompt.ask')
    def test_interactive_mode_basic(self, mock_prompt, mock_llama_class, infernal_with_model, capsys):
        """Test basic interactive mode functionality"""
        mock_prompt.side_effect = ["Hello", "quit"]
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        mock_llm.return_value = {"choices": [{"text": "Hi there!"}]}
        
        infernal_with_model.run_inference("test_model", interactive=True)
        
        captured = capsys.readouterr()
        assert "Infernal Chat: test_model" in captured.out
        assert "Infernal session ended" in captured.out

    @patch('infernal.Llama')
    @patch('infernal.Prompt.ask')
    def test_interactive_mode_empty_input(self, mock_prompt, mock_llama_class, infernal_with_model):
        """Test interactive mode with empty input"""
        mock_prompt.side_effect = ["", "   ", "quit"]
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        
        infernal_with_model.run_inference("test_model", interactive=True)
        
        # Should not call the model for empty inputs
        mock_llm.assert_not_called()

    @patch('infernal.Llama')
    @patch('infernal.Prompt.ask')
    def test_interactive_mode_keyboard_interrupt(self, mock_prompt, mock_llama_class, infernal_with_model, capsys):
        """Test interactive mode handling keyboard interrupt"""
        mock_prompt.side_effect = KeyboardInterrupt()
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm
        
        infernal_with_model.run_inference("test_model", interactive=True)
        
        captured = capsys.readouterr()
        assert "Infernal session ended" in captured.out
