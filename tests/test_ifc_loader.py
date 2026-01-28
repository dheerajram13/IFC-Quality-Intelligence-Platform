"""Unit tests for IFC loader module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from ifcqi.ifc_loader import (
    IFCLoadError,
    extract_elements,
    generate_summary_report,
    get_element_summary,
    get_model_metadata,
    load_ifc,
    save_elements_to_csv,
    validate_ifc_file,
)


@pytest.fixture
def mock_ifc_model():
    """Create a mock IFC model for testing."""
    model = Mock()
    model.schema = "IFC4"

    # Create mock products
    mock_wall = Mock()
    mock_wall.GlobalId = "wall-001"
    mock_wall.is_a.return_value = "IfcWall"
    mock_wall.Name = "Wall 1"
    mock_wall.Description = "External wall"
    mock_wall.ObjectType = "LoadBearing"
    mock_wall.PredefinedType = "SOLIDWALL"

    mock_door = Mock()
    mock_door.GlobalId = "door-001"
    mock_door.is_a.return_value = "IfcDoor"
    mock_door.Name = "Door 1"
    mock_door.Description = None
    mock_door.ObjectType = "Single"
    mock_door.PredefinedType = "DOOR"

    mock_window = Mock()
    mock_window.GlobalId = "window-001"
    mock_window.is_a.return_value = "IfcWindow"
    mock_window.Name = None
    mock_window.Description = None
    mock_window.ObjectType = None
    mock_window.PredefinedType = None

    model.by_type.return_value = [mock_wall, mock_door, mock_window]

    # Mock header
    mock_header = Mock()
    mock_file_desc = Mock()
    mock_file_desc.description = ["ViewDefinition [CoordinationView]"]
    mock_file_desc.implementation_level = "2;1"
    mock_header.file_description = mock_file_desc

    mock_file_name = Mock()
    mock_file_name.name = "test_model.ifc"
    mock_file_name.time_stamp = "2024-01-28T10:00:00"
    mock_file_name.author = ["Test Author"]
    mock_file_name.organization = ["Test Org"]
    mock_file_name.preprocessor_version = "IfcOpenShell 0.7.0"
    mock_file_name.originating_system = "TestApp"
    mock_file_name.authorization = "None"
    mock_header.file_name = mock_file_name

    model.header = mock_header

    return model


class TestLoadIFC:
    """Tests for load_ifc function."""

    def test_load_ifc_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_ifc(Path("nonexistent.ifc"))

    @patch("ifcopenshell.open")
    def test_load_ifc_success(self, mock_open, mock_ifc_model, tmp_path):
        """Test successful IFC file loading."""
        test_file = tmp_path / "test.ifc"
        test_file.touch()

        mock_open.return_value = mock_ifc_model

        model = load_ifc(test_file)

        assert model == mock_ifc_model
        mock_open.assert_called_once_with(str(test_file))

    @patch("ifcopenshell.open")
    def test_load_ifc_parse_error(self, mock_open, tmp_path):
        """Test that IFCLoadError is raised when file cannot be parsed."""
        test_file = tmp_path / "invalid.ifc"
        test_file.touch()

        mock_open.side_effect = Exception("Invalid IFC format")

        with pytest.raises(IFCLoadError, match="Failed to load IFC file"):
            load_ifc(test_file)


class TestExtractElements:
    """Tests for extract_elements function."""

    def test_extract_elements_all(self, mock_ifc_model):
        """Test extracting all elements from model."""
        df = extract_elements(mock_ifc_model)

        assert len(df) == 3
        assert list(df.columns) == [
            "global_id",
            "ifc_type",
            "name",
            "description",
            "object_type",
            "predefined_type",
        ]

        # Check first element (wall)
        assert df.iloc[0]["global_id"] == "wall-001"
        assert df.iloc[0]["ifc_type"] == "IfcWall"
        assert df.iloc[0]["name"] == "Wall 1"
        assert df.iloc[0]["description"] == "External wall"

        # Check element with None values (window)
        assert df.iloc[2]["global_id"] == "window-001"
        assert df.iloc[2]["name"] is None
        assert df.iloc[2]["object_type"] is None

    def test_extract_elements_with_limit(self, mock_ifc_model):
        """Test extracting limited number of elements."""
        df = extract_elements(mock_ifc_model, max_elements=2)

        assert len(df) == 2

    def test_extract_elements_empty_model(self):
        """Test extraction with empty model."""
        mock_model = Mock()
        mock_model.by_type.return_value = []

        df = extract_elements(mock_model)

        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_extract_elements_with_error(self, mock_ifc_model):
        """Test that elements with extraction errors are skipped."""
        # Make one element raise an error when accessing is_a()
        mock_element_with_error = Mock()
        mock_element_with_error.GlobalId = "error-001"
        mock_element_with_error.is_a.side_effect = Exception("Error accessing type")

        # Replace the second element with the error element
        products = list(mock_ifc_model.by_type.return_value)
        products[1] = mock_element_with_error
        mock_ifc_model.by_type.return_value = products

        df = extract_elements(mock_ifc_model)

        # Should have 2 elements (1 skipped due to error)
        assert len(df) == 2


class TestGetElementSummary:
    """Tests for get_element_summary function."""

    def test_get_element_summary(self, mock_ifc_model):
        """Test generating element summary."""
        df = extract_elements(mock_ifc_model)
        summary = get_element_summary(df)

        assert isinstance(summary, dict)
        assert summary["IfcWall"] == 1
        assert summary["IfcDoor"] == 1
        assert summary["IfcWindow"] == 1

    def test_get_element_summary_empty_df(self):
        """Test summary with empty DataFrame."""
        df = pd.DataFrame()
        summary = get_element_summary(df)

        assert summary == {}

    def test_get_element_summary_multiple_same_type(self):
        """Test summary with multiple elements of same type."""
        df = pd.DataFrame(
            {
                "global_id": ["1", "2", "3"],
                "ifc_type": ["IfcWall", "IfcWall", "IfcDoor"],
                "name": [None, None, None],
            }
        )

        summary = get_element_summary(df)

        assert summary["IfcWall"] == 2
        assert summary["IfcDoor"] == 1


class TestGetModelMetadata:
    """Tests for get_model_metadata function."""

    def test_get_model_metadata_full(self, mock_ifc_model):
        """Test extracting full metadata."""
        metadata = get_model_metadata(mock_ifc_model)

        assert metadata["schema"] == "IFC4"
        assert metadata["name"] == "test_model.ifc"
        assert metadata["author"] == "Test Author"
        assert metadata["organization"] == "Test Org"
        assert metadata["originating_system"] == "TestApp"

    def test_get_model_metadata_minimal(self):
        """Test metadata extraction with minimal header info."""
        mock_model = Mock()
        mock_model.schema = "IFC2X3"
        mock_model.header = Mock(side_effect=Exception("No header"))

        metadata = get_model_metadata(mock_model)

        # Should at least return schema
        assert metadata["schema"] == "IFC2X3"

    def test_get_model_metadata_empty_lists(self):
        """Test metadata with empty author/organization lists."""
        mock_model = Mock()
        mock_model.schema = "IFC4"

        mock_header = Mock()
        mock_file_desc = Mock()
        mock_file_desc.description = []
        mock_file_desc.implementation_level = "2;1"
        mock_header.file_description = mock_file_desc

        mock_file_name = Mock()
        mock_file_name.name = "test.ifc"
        mock_file_name.author = []
        mock_file_name.organization = []
        mock_header.file_name = mock_file_name

        mock_model.header = mock_header

        metadata = get_model_metadata(mock_model)

        assert metadata["author"] is None
        assert metadata["organization"] is None


class TestSaveElementsToCSV:
    """Tests for save_elements_to_csv function."""

    def test_save_elements_to_csv(self, tmp_path):
        """Test saving elements to CSV."""
        df = pd.DataFrame(
            {
                "global_id": ["1", "2"],
                "ifc_type": ["IfcWall", "IfcDoor"],
                "name": ["Wall", "Door"],
            }
        )

        output_path = tmp_path / "output" / "elements.csv"
        save_elements_to_csv(df, output_path)

        assert output_path.exists()

        # Verify content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == ["global_id", "ifc_type", "name"]


class TestGenerateSummaryReport:
    """Tests for generate_summary_report function."""

    def test_generate_summary_report(self, mock_ifc_model, tmp_path):
        """Test generating summary report."""
        df = extract_elements(mock_ifc_model)
        output_path = tmp_path / "summary.json"

        generate_summary_report(mock_ifc_model, df, output_path)

        assert output_path.exists()

        # Verify content
        import json

        with open(output_path) as f:
            report = json.load(f)

        assert "metadata" in report
        assert "statistics" in report
        assert report["statistics"]["total_elements"] == 3
        assert report["statistics"]["unique_types"] == 3
        assert "IfcWall" in report["statistics"]["elements_by_type"]


class TestValidateIFCFile:
    """Tests for validate_ifc_file function."""

    def test_validate_file_not_exists(self):
        """Test validation of non-existent file."""
        is_valid, error = validate_ifc_file(Path("nonexistent.ifc"))

        assert not is_valid
        assert "does not exist" in error

    def test_validate_file_wrong_extension(self, tmp_path):
        """Test validation of file with wrong extension."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        is_valid, error = validate_ifc_file(test_file)

        assert not is_valid
        assert "extension must be" in error

    @patch("ifcopenshell.open")
    def test_validate_valid_ifc_file(self, mock_open, mock_ifc_model, tmp_path):
        """Test validation of valid IFC file."""
        test_file = tmp_path / "test.ifc"
        test_file.touch()

        mock_open.return_value = mock_ifc_model

        is_valid, error = validate_ifc_file(test_file)

        assert is_valid
        assert error is None

    @patch("ifcopenshell.open")
    def test_validate_invalid_ifc_content(self, mock_open, tmp_path):
        """Test validation of file with invalid IFC content."""
        test_file = tmp_path / "invalid.ifc"
        test_file.touch()

        mock_open.side_effect = Exception("Parse error")

        is_valid, error = validate_ifc_file(test_file)

        assert not is_valid
        assert "Failed to parse" in error

    def test_validate_directory_not_file(self, tmp_path):
        """Test validation when path is a directory."""
        is_valid, error = validate_ifc_file(tmp_path)

        assert not is_valid
        assert "not a file" in error
