# Custom Dataset Parsing in DocETL

In DocETL, you have full control over your dataset JSONs. These JSONs typically contain objects with key-value pairs, where you can specify paths or references to external files that you want to process in your pipeline. But what if these external files are in formats that need special handling before they can be used in your main pipeline? This is where custom parsing in DocETL becomes useful.


!!! info "Why Use Custom Parsing?"

    Consider these scenarios:

    - Your dataset JSON contains paths to Excel spreadsheets with sales data.
    - You have references to scanned receipts in PDF format that need OCR processing.
    - You want to extract text from Word documents or PowerPoint presentations.

    In these cases, custom parsing enables you to transform your raw external data into a format that DocETL can process effectively within your pipeline.

## Dataset JSON Example

Let's look at a typical dataset JSON file that you might create:

```json
[
  { "id": 1, "excel_path": "sales_data/january_sales.xlsx" },
  { "id": 2, "excel_path": "sales_data/february_sales.xlsx" },
]
```

In this example, you've specified paths to Excel files. DocETL will use these paths to locate and process the external files. However, without custom parsing, DocETL wouldn't know how to handle the contents of these files. This is where parsing tools come in handy.

## Custom Parsing in Action

### 1. Configuration

To use custom parsing, you need to define parsing tools in your DocETL configuration file. Here's an example:

```yaml
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
          return [text]

datasets:
  sales_reports:
    type: file
    source: local
    path: "sales_data/sales_paths.json"
    parsing:
      - input_key: excel_path
        function: xlsx_to_string
        output_key: sales_data
        function_kwargs:
            orientation: "col"

  receipts:
    type: file
    source: local
    path: "receipts/receipt_paths.json"
    parsing:
      - input_key: pdf_path
        function: ocr_parser
        output_key: receipt_text
```

In this configuration:
- We define a custom `ocr_parser` for PDF files.
- We use the built-in `xlsx_to_string` parser for Excel files.
- We apply these parsing tools to the external files referenced in the respective datasets.

### 2. Pipeline Integration

Once you've defined your parsing tools and datasets, you can use the processed data in your pipeline:

```yaml
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
```

This pipeline will use the parsed data from both Excel files and PDFs for further processing.

## Built-in Parsing Tools

DocETL provides several built-in parsing tools to handle common file formats and data processing tasks. These can be used directly in your configuration by specifying their names in the `function` field of your parsing configuration.

[Insert the existing documentation for built-in parsing tools here]

## Creating Custom Parsing Tools

If the built-in tools don't meet your needs, you can create your own custom parsing tools. Here's how:

1. Define your parsing function in the `parsing_tools` section of your configuration.
2. Ensure your function takes a filename as input and returns a list of strings.
3. Use your custom parser in the `parsing` section of your dataset configuration.

For example:

```yaml
parsing_tools:
  - name: my_custom_parser
    function_code: |
      def my_custom_parser(filename: str) -> List[str]:
          # Your custom parsing logic here
          return [processed_data]

datasets:
  my_dataset:
    type: file
    source: local
    path: "data/paths.json"
    parsing:
      - input_key: file_path
        function: my_custom_parser
        output_key: processed_data
```

### Understanding the Parsing Tools

In this example, we used two parsing tools:

1. **xlsx_to_string**: A built-in parsing tool provided by DocETL. It reads Excel files and converts them to a string representation.

2. **ocr_parser**: A custom parsing tool we defined for OCR processing of PDF files. *Note that it returns a list containing a single string, which is the format expected by DocETL for parsing tools.*


## How Data Gets Parsed and Formatted

When you run your DocETL pipeline, the parsing tools you've specified in your configuration file are applied to the external files referenced in your dataset JSONs. Here's what happens:

1. DocETL reads your dataset JSON file.
2. For each entry in the dataset, it looks at the parsing configuration you've specified.
3. It applies the appropriate parsing function to the file path provided in the dataset JSON.
4. The parsing function processes the file and returns the data in a format DocETL can work with (typically a list of strings).

Let's look at how this works for our earlier examples:

### Excel Files (using xlsx_to_string)

For an Excel file like "sales_data/january_sales.xlsx":

1. The `xlsx_to_string` function reads the Excel file.
2. It converts the data to a string representation.
3. The output might look like this:

```
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
```

### PDF Files (using ocr_parser)

For a PDF file like "receipts/receipt001.pdf":

1. The `ocr_parser` function converts each page of the PDF to an image.
2. It applies OCR to each image.
3. The function combines the text from all pages.
4. The output might look like this:

```
RECEIPT
Store: Example Store
Date: 2023-05-15
Items:
1. Product A - $10.99
2. Product B - $15.50
Total: $26.49
```

This parsed and formatted data is then passed to the respective operations in your pipeline for further processing.

## Running the Pipeline

Once you've set up your pipeline configuration file with the appropriate parsing tools and dataset definitions, you can run your DocETL pipeline. Here's how:

1. Ensure you have DocETL installed in your environment.
2. Open a terminal or command prompt.
3. Navigate to the directory containing your pipeline configuration file.
4. Run the following command:

```bash
docetl run pipeline.yaml
```

Replace `pipeline.yaml` with the name of your pipeline file if it's different.

When you run this command:

1. DocETL reads your pipeline file.
2. It processes each dataset using the specified parsing tools.
3. The pipeline steps are executed in the order you defined.
4. Any operations you've specified (like `summarize_sales` or `extract_receipt_info`) are applied to the parsed data.
5. The results are saved according to your output configuration.



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

## Contributing Built-in Parsing Tools

While DocETL provides several built-in parsing tools, the community can always benefit from additional utilities. If you've developed a parsing tool that you think could be useful for others, consider contributing it to the DocETL repository. Here's how you can add new built-in parsing utilities:

1. Fork the DocETL repository on GitHub.
2. Clone your forked repository to your local machine.
3. Navigate to the `docetl/parsing_tools.py` file.
4. Add your new parsing function to this file. The function should also be added to the `PARSING_TOOLS` dictionary.
5. Update the documentation in the function's docstring.
6. Create a pull request to merge your changes into the main DocETL repository.

!!! note "Guidelines for Contributing Parsing Tools"

    When contributing a new parsing tool, make sure it follows these guidelines:

    - The function should have a clear, descriptive name.
    - Include comprehensive docstrings explaining the function's purpose, parameters, and return value. The return value should be a list of strings.
    - Handle potential errors gracefully and provide informative error messages.
    - If your parser requires additional dependencies, make sure to mention them in the pull request.

