import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from infernal import InfernalLLM

@pytest.fixture
def infernal_instance(tmp_path):
    """Create InfernalLLM instance with temporary directory"""
    return InfernalLLM(str(tmp_path / "models"))

class TestInfernalLLM:
    def test_init_default_dir(self):
        """Test InfernalLLM initialization with default directory"""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/fake/home")
            infernal = InfernalLLM()
            assert infernal.models_dir.name == "models"

    def test_init_custom_dir(self, tmp_path):
        """Test InfernalLLM initialization with custom directory"""
        models_dir = tmp_path / "custom_models"
        infernal = InfernalLLM(str(models_dir))
        assert infernal.models_dir == models_dir
        assert models_dir.exists()

    def test_config_creation(self, infernal_instance):
        """Test configuration file creation"""
        assert infernal_instance.config_file.exists()
        assert "models" in infernal_instance.config
        assert "settings" in infernal_instance.config

    def test_get_model_path_existing(self, infernal_instance):
        """Test getting path for existing model"""
        model_file = infernal_instance.models_dir / "test.gguf"
        model_file.touch()  # Create the file
        infernal_instance.config["models"]["test_model"] = {"filename": "test.gguf"}
        
        path = infernal_instance.get_model_path("test_model")
        assert path == model_file

    def test_get_model_path_nonexistent(self, infernal_instance):
        """Test getting path for non-existent model"""
        path = infernal_instance.get_model_path("nonexistent")
        assert path is None

    def test_list_models_empty(self, infernal_instance, capsys):
        """Test listing models when none exist - Fixed assertion"""
        infernal_instance.list_models()
        captured = capsys.readouterr()
        # Fixed: Use the exact text from your code
        assert "No models found" in captured.out

    def test_list_models_with_models(self, infernal_instance, capsys):
        """Test listing models when models exist - Fixed assertion"""
        # Setup test model
        model_file = infernal_instance.models_dir / "test.gguf"
        model_file.write_bytes(b"fake model data")
        
        infernal_instance.config["models"]["test_model"] = {
            "filename": "test.gguf",
            "size": len(b"fake model data")
        }
        
        infernal_instance.list_models()
        captured = capsys.readouterr()
        # Fixed: Use the exact text from your code
        assert "Available Models:" in captured.out
        assert "test_model" in captured.out

class TestModelDownload:
    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_pull_model_success(self, mock_download, tmp_path):
        """Test successful model download"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        infernal = InfernalLLM(str(models_dir))
        
        downloaded_file = models_dir / "model.gguf"
        
        def mock_download_side_effect(*args, **kwargs):
            downloaded_file.write_bytes(b"fake model data")
            return str(downloaded_file)
        
        mock_download.side_effect = mock_download_side_effect
        
        result = infernal.download_from_huggingface("test/repo", "model.gguf")
        
        expected_file = models_dir / "repo.gguf"
        assert expected_file.exists()
        assert result == expected_file

    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_pull_from_url_valid(self, mock_download, tmp_path):
        """Test pull from valid URL"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        infernal = InfernalLLM(str(models_dir))
        
        downloaded_file = models_dir / "model.gguf"
        
        def mock_download_side_effect(*args, **kwargs):
            downloaded_file.write_bytes(b"fake model data")
            return str(downloaded_file)
        
        mock_download.side_effect = mock_download_side_effect
        
        url = "https://huggingface.co/test/repo/resolve/main/model.gguf"
        result = infernal.download_model(url)
        
        expected_file = models_dir / "repo.gguf"
        assert expected_file.exists()
        assert result == expected_file

    def test_pull_from_url_invalid(self, tmp_path):
        """Test pull from invalid URL"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        infernal = InfernalLLM(str(models_dir))
        
        with pytest.raises(ValueError) as exc_info:
            infernal.download_model("https://example.com/model.gguf")
        
        assert "Only Hugging Face URLs are supported" in str(exc_info.value)

    def test_remove_model_success(self, tmp_path):
        """Test successful model removal"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        infernal = InfernalLLM(str(models_dir))
        
        model_file = models_dir / "test.gguf"
        model_file.write_bytes(b"test data")
        
        infernal.config["models"]["test_model"] = {"filename": "test.gguf"}
        
        infernal.remove_model("test_model")
        
        assert not model_file.exists()
        assert "test_model" not in infernal.config["models"]

    def test_remove_model_not_found(self, infernal_instance, capsys):
        """Test removing non-existent model"""
        infernal_instance.remove_model("nonexistent")
        captured = capsys.readouterr()
        assert "not found" in captured.out
