import os
from typing import Optional, List
from litellm import transcription


def whisper_speech_to_text(filename: str) -> List[str]:
    """
    Transcribe speech from an audio file to text using Whisper model via litellm.
    If the file is larger than 25 MB, it's split into 10-minute chunks with 30-second overlap.

    Args:
        filename (str): Path to the mp3 or mp4 file.

    Returns:
        List[str]: Transcribed text.
    """
    import os

    file_size = os.path.getsize(filename)
    if file_size > 25 * 1024 * 1024:  # 25 MB in bytes
        from pydub import AudioSegment

        audio = AudioSegment.from_file(filename)
        chunk_length = 10 * 60 * 1000  # 10 minutes in milliseconds
        overlap = 30 * 1000  # 30 seconds in milliseconds

        chunks = []
        for i in range(0, len(audio), chunk_length - overlap):
            chunk = audio[i : i + chunk_length]
            chunks.append(chunk)

        transcriptions = []
        for i, chunk in enumerate(chunks):
            temp_filename = f"temp_chunk_{i}_{os.path.basename(filename)}"
            chunk.export(temp_filename, format="mp3")

            with open(temp_filename, "rb") as audio_file:
                response = transcription(model="whisper-1", file=audio_file)
            transcriptions.append(response.text)

            os.remove(temp_filename)

        return transcriptions
    else:
        with open(filename, "rb") as audio_file:
            response = transcription(model="whisper-1", file=audio_file)

        return [response.text]


def xlsx_to_string(
    filename: str,
    orientation: str = "col",
    col_order: Optional[List[str]] = None,
    doc_per_sheet: bool = False,
) -> List[str]:
    """
    Convert an Excel file to a string representation or a list of string representations.

    Args:
        filename (str): Path to the xlsx file.
        orientation (str): Either "row" or "col" for cell arrangement.
        col_order (Optional[List[str]]): List of column names to specify the order.
        doc_per_sheet (bool): If True, return a list of strings, one per sheet.

    Returns:
        List[str]: String representation(s) of the Excel file content.
    """
    import openpyxl

    wb = openpyxl.load_workbook(filename)

    def process_sheet(sheet):
        if col_order:
            headers = col_order
        else:
            headers = [cell.value for cell in sheet[1]]

        result = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            row_dict = dict(zip(headers, row))
            if orientation == "col":
                result.extend(
                    [f"{header}: {value}" for header, value in row_dict.items()]
                )
                result.append("")  # Empty line between rows
            else:  # row
                result.append(
                    " | ".join(
                        [f"{header}: {value}" for header, value in row_dict.items()]
                    )
                )

        return "\n".join(result)

    if doc_per_sheet:
        return [process_sheet(sheet) for sheet in wb.worksheets]
    else:
        return [process_sheet(wb.active)]


def txt_to_string(filename: str) -> List[str]:
    """
    Read the content of a text file and return it as a list of strings (only one element).

    Args:
        filename (str): Path to the txt or md file.

    Returns:
        List[str]: Content of the file as a list of strings.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return [file.read()]


def docx_to_string(filename: str) -> List[str]:
    """
    Extract text from a Word document.

    Args:
        filename (str): Path to the docx file.

    Returns:
        List[str]: Extracted text from the document.
    """
    from docx import Document

    doc = Document(filename)
    return ["\n".join([paragraph.text for paragraph in doc.paragraphs])]


# Define a dictionary mapping function names to their corresponding functions
PARSING_TOOLS = {
    "whisper_speech_to_text": whisper_speech_to_text,
    "xlsx_to_string": xlsx_to_string,
    "txt_to_string": txt_to_string,
    "docx_to_string": docx_to_string,
}
