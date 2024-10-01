# Custom Parsing in DocETL

DocETL provides some custom parsing capabilities that allow you to preprocess your data before it enters the main pipeline. This guide will walk you through creating a pipeline with custom parsing tools using a concrete example.

## Example Scenario

Imagine you have:

- A folder called "sales_data" containing JSON files with paths to Excel spreadsheets of monthly sales reports.
- A folder called "receipts" with JSON files containing paths to scanned receipts in PDF format that you want to process using OCR.

## Setting Up Custom Parsing

Let's walk through setting up a pipeline with custom parsing for this scenario:

### 1. Create a Configuration File

First, create a configuration file (`config.yaml`) that defines your dataset, parsing tools, and pipeline:

```yaml
default_model: "gpt-4o-mini"

parsing_tools:
  - name: ocr_parser
    function_code: |
      import pytesseract
      from pdf2image import convert_from_path
      def ocr_parser(filename: str) -> List[str]:
          images = convert_from_path(filename)
          text = ""
          for image in images:
              text += pytesseract.image_to_string(image)
          return [text]  # Return as a list with one element

operations:
  - name: summarize_sales
    type: map
    prompt: |
      Summarize the following sales data:
      {{ input.sales_data }}
    output:
      schema:
        summary: string
    model: "gpt-4o-mini"
  - name: extract_receipt_info
    type: map
    prompt: |
      Extract the total amount and date from the following receipt text:
      {{ input.receipt_text }}
    output:
      schema:
        total_amount: float
        date: string
    model: "gpt-4o-mini"

datasets:
  sales_reports:
    type: file
    source: local
    path: "sales_data/sales_paths.json"
    parsing_tools:
      - input_key: excel_path
        function: xlsx_to_string
        output_key: sales_data
        function_kwargs:
            orientation: "col"

  receipts:
    type: file
    source: local
    path: "receipts/receipt_paths.json"
    parsing_tools:
      - input_key: pdf_path
        function: ocr_parser
        output_key: receipt_text

pipeline:
  steps:
    - name: process_sales
      input: sales_reports
      operations:
        - summarize_sales
    - name: process_receipts
      input: receipts
      operations:
        - extract_receipt_info

output:
  type: file
  path: "output.json"
```

### 2. Configuration Breakdown

In this configuration:

- We define a custom parsing tool `ocr_parser` for PDF files.
- We use the built-in `xlsx_to_string` parsing tool for Excel files.
- We create two datasets: `sales_reports` for Excel files and `receipts` for PDF files.
- We apply the parsing tools to their respective datasets.
- We define map operations to process the parsed data.

### 3. Prepare Required Files

Ensure you have the necessary input files:

#### JSON file for Excel paths (`sales_data/sales_paths.json`):

```json
[
  { "id": 1, "excel_path": "sales_data/january_sales.xlsx" },
  { "id": 2, "excel_path": "sales_data/february_sales.xlsx" }
]
```

#### JSON file for PDF paths (`receipts/receipt_paths.json`):

```json
[
  { "id": 1, "pdf_path": "receipts/receipt001.pdf" },
  { "id": 2, "pdf_path": "receipts/receipt002.pdf" }
]
```


#### Parsing Process

Let's examine how the input files would be parsed using the logic defined in `parsing_tools.py`:

1. For the Excel files (`sales_data/january_sales.xlsx` and `sales_data/february_sales.xlsx`):
   - The `xlsx_to_string` function is used.
   - By default, it processes the active sheet of each Excel file.
   - The function returns a list containing a single string for each file.
   - The string representation includes column headers followed by their respective values.
   - For example, if the Excel file has columns "Date", "Product", and "Amount", the output might look like:

     Date:
     2023-01-01
     2023-01-02
     ...

     Product:
     Widget A
     Widget B
     ...

     Amount:
     100
     150
     ...

2. For the PDF files (`receipts/receipt001.pdf` and `receipts/receipt002.pdf`):
   - The custom `ocr_parser` function is used.
   - It converts each page of the PDF to an image using `pdf2image`.
   - Then, it applies OCR to each image using `pytesseract`.
   - The function combines the text from all pages and returns it as a list with a single string element.
   - The output might look like:

     RECEIPT
     Store: Example Store
     Date: 2023-05-15
     Items:
     1. Product A - $10.99
     2. Product B - $15.50
     Total: $26.49

These parsed strings are then passed to the respective operations (`summarize_sales` and `extract_receipt_info`) for further processing in the pipeline.


### 4. Run the Pipeline

Execute the pipeline using the DocETL CLI:

```bash
docetl run config.yaml
```

### 5. Check the Output

After running the pipeline, you'll find the output in `output.json`. It will contain summaries of the sales data and extracted information from the receipts.

## Understanding the Parsing Tools

In this example, we used two parsing tools:

1. **xlsx_to_string**: A built-in parsing tool provided by DocETL. It reads Excel files and converts them to a string representation.

2. **ocr_parser**: A custom parsing tool we defined for OCR processing of PDF files. *Note that it returns a list containing a single string, which is the format expected by DocETL for parsing tools.*

## Built-in Parsing Tools

DocETL provides several built-in parsing tools to handle common file formats and data processing tasks. These tools can be used directly in your configuration by specifying their names in the `function` field of your parsing tools configuration. Here's an overview of the available built-in parsing tools:

::: docetl.parsing_tools.xlsx_to_string
    options:
        heading_level: 3

::: docetl.parsing_tools.txt_to_string
    options:
        heading_level: 3

::: docetl.parsing_tools.docx_to_string
    options:
        heading_level: 3

::: docetl.parsing_tools.whisper_speech_to_text
    options:
        heading_level: 3

::: docetl.parsing_tools.pptx_to_string
    options:
        heading_level: 3


### Using Function Arguments with Parsing Tools

When using parsing tools in your DocETL configuration, you can pass additional arguments to the parsing functions using the function_kwargs field. This allows you to customize the behavior of the parsing tools without modifying their implementation.

For example, when using the xlsx_to_string parsing tool, you can specify options like the orientation of the data, the order of columns, or whether to process each sheet separately. Here's an example of how to use function_kwargs in your configuration:

```yaml
datasets:
  my_sales:
    type: file
    source: local
    path: "sales_data/sales_paths.json"
    parsing_tools:
      - name: excel_parser
        function: xlsx_to_string
        function_kwargs:
          orientation: row
          col_order: ["Date", "Product", "Quantity", "Price"]
          doc_per_sheet: true
```

