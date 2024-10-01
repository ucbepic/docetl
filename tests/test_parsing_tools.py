import pytest
import os
import tempfile
from docetl import parsing_tools


@pytest.fixture
def temp_audio_file():
    import requests

    url = "https://listenaminute.com/a/animals.mp3"
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response.content)
    yield temp_file.name
    return temp_file.name


@pytest.fixture
def temp_xlsx_file():
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Age", "City"])
    ws.append(["Alice", 30, "New York"])
    ws.append(["Bob", 25, "London"])
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        wb.save(temp_file.name)
    yield temp_file.name
    return temp_file.name


@pytest.fixture
def temp_txt_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as temp_file:
        temp_file.write("This is a test text file.\nIt has multiple lines.")
    yield temp_file.name
    return temp_file.name


@pytest.fixture
def temp_docx_file():
    from docx import Document

    doc = Document()
    doc.add_paragraph("This is a test Word document.")
    doc.add_paragraph("It has multiple paragraphs.")
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
        doc.save(temp_file.name)
    yield temp_file.name
    return temp_file.name


def test_whisper_speech_to_text(temp_audio_file):
    result = parsing_tools.whisper_speech_to_text(temp_audio_file)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0  # Ensure some text was transcribed


def test_xlsx_to_string(temp_xlsx_file):
    result = parsing_tools.xlsx_to_string(temp_xlsx_file)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "Name: Alice" in result[0]
    assert "Age: 30" in result[0]
    assert "City: New York" in result[0]
    assert "Name: Bob" in result[0]
    assert "Age: 25" in result[0]
    assert "City: London" in result[0]


def test_xlsx_to_string_row_orientation(temp_xlsx_file):
    result = parsing_tools.xlsx_to_string(temp_xlsx_file, orientation="row")

    assert isinstance(result, list)
    assert len(result) == 1
    assert "Name: Alice | Age: 30 | City: New York" in result[0]
    assert "Name: Bob | Age: 25 | City: London" in result[0]


def test_xlsx_to_string_doc_per_sheet(temp_xlsx_file):
    result = parsing_tools.xlsx_to_string(temp_xlsx_file, doc_per_sheet=True)

    assert isinstance(result, list)
    assert len(result) == 1  # Only one sheet in our test file
    assert "Name: Alice" in result[0]
    assert "Age: 30" in result[0]
    assert "City: New York" in result[0]


def test_txt_to_string(temp_txt_file):
    result = parsing_tools.txt_to_string(temp_txt_file)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "This is a test text file.\nIt has multiple lines."


def test_docx_to_string(temp_docx_file):
    result = parsing_tools.docx_to_string(temp_docx_file)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "This is a test Word document." in result[0]
    assert "It has multiple paragraphs." in result[0]


# Clean up temporary files after all tests have passed
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        for fixture in [temp_audio_file, temp_xlsx_file, temp_txt_file, temp_docx_file]:
            file_path = session.config.cache.get(fixture.__name__, None)
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                os.unlink(file_path)
