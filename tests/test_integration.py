import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from infernal import InfernalLLM

@pytest.fixture
def infernal_instance(tmp_path):
    """Create InfernalLLM instance for integration tests"""
    return InfernalLLM(str(tmp_path / "models"))

class TestIntegration:
    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_full_model_lifecycle(self, mock_download, infernal_instance):
        """Test complete model lifecycle - Fixed config assertion"""
        model_file = infernal_instance.models_dir / "model.gguf"
        
        def mock_download_side_effect(*args, **kwargs):
            model_file.write_bytes(b"fake model data")
            return str(model_file)
        
        mock_download.side_effect = mock_download_side_effect
        
        result = infernal_instance.download_from_huggingface("test/repo", "model.gguf")
        
        expected_file = infernal_instance.models_dir / "repo.gguf"
        assert expected_file.exists()
        assert "repo" in infernal_instance.config["models"]  # Fixed: use "repo" not "test_repo"
        
        model_path = infernal_instance.get_model_path("repo")
        assert model_path == expected_file
        
        infernal_instance.remove_model("repo")
        assert not expected_file.exists()
        assert "repo" not in infernal_instance.config["models"]

    @patch('infernal.huggingface_hub.hf_hub_download')
    def test_url_parsing_and_download(self, mock_download, infernal_instance):
        """Test URL parsing and download - Mocked to avoid 401 error"""
        model_file = infernal_instance.models_dir / "model.gguf"
        
        def mock_download_side_effect(*args, **kwargs):
            model_file.write_bytes(b"fake model data")
            return str(model_file)
        
        mock_download.side_effect = mock_download_side_effect
        
        url = "https://huggingface.co/test/repo/resolve/main/model.gguf"
        result = infernal_instance.download_model(url)
        
        expected_file = infernal_instance.models_dir / "repo.gguf"
        assert expected_file.exists()
        assert "repo" in infernal_instance.config["models"]
        
        assert infernal_instance.config["models"]["repo"]["repo_id"] == "test/repo"
        assert infernal_instance.config["models"]["repo"]["original_filename"] == "model.gguf"

    def test_config_persistence(self, tmp_path):
        """Test that configuration persists across instances"""
        models_dir = tmp_path / "models"
        
        infernal1 = InfernalLLM(str(models_dir))
        infernal1.config["models"]["test"] = {"filename": "test.gguf", "size": 1024}
        infernal1.save_config()
        
        infernal2 = InfernalLLM(str(models_dir))
        assert "test" in infernal2.config["models"]
        assert infernal2.config["models"]["test"]["size"] == 1024
