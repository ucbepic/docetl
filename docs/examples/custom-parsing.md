# Pointing to External Data and Custom Parsing

In DocETL, you have full control over your dataset JSONs. These JSONs typically contain objects with key-value pairs, where you can reference external files that you want to process in your pipeline. This referencing mechanism, which we call "pointing", allows DocETL to locate and process external files that require special handling before they can be used in your main pipeline.

!!! info "Why Use Custom Parsing?"

    Consider these scenarios where custom parsing of referenced files is beneficial:

    - Your dataset JSON references Excel spreadsheets containing sales data.
    - You have entries pointing to scanned receipts in PDF format that need OCR processing.
    - You want to extract text from Word documents or PowerPoint presentations by referencing their file locations.

    In these cases, custom parsing enables you to transform your raw external data into a format that DocETL can process effectively within your pipeline. The pointing mechanism allows DocETL to locate these external files and apply custom parsing seamlessly. _(Pointing in DocETL refers to the practice of including references or paths to external files within your dataset JSON. Instead of embedding the entire content of these files, you simply "point" to their locations, allowing DocETL to access and process them as needed during the pipeline execution.)_

## Dataset JSON Example

Let's look at a typical dataset JSON file that you might create:

```json
[
  { "id": 1, "excel_path": "sales_data/january_sales.xlsx" },
  { "id": 2, "excel_path": "sales_data/february_sales.xlsx" }
]
```

In this example, you've specified paths to Excel files. DocETL will use these paths to locate and process the external files. However, without custom parsing, DocETL wouldn't know how to handle the contents of these files. This is where parsing tools come in handy.

## Custom Parsing in Action

#### 1. Configuration

To use custom parsing, you need to define parsing tools in your DocETL configuration file. Here's an example:

```yaml
parsing_tools:
  - name: top_products_report
    function_code: |
      def top_products_report(document: Dict) -> List[Dict]:
          import pandas as pd
          
          # Read the Excel file
          filename = document["excel_path"]
          df = pd.read_excel(filename)
          
          # Calculate total sales
          total_sales = df['Sales'].sum()
          
          # Find top 500 products by sales
          top_products = df.groupby('Product')['Sales'].sum().nlargest(500)
          
          # Calculate month-over-month growth
          df['Date'] = pd.to_datetime(df['Date'])
          monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()
          mom_growth = monthly_sales.pct_change().fillna(0)
          
          # Prepare the analysis report
          report = [
              f"Total Sales: ${total_sales:,.2f}",
              "\nTop 500 Products by Sales:",
              top_products.to_string(),
              "\nMonth-over-Month Growth:",
              mom_growth.to_string()
          ]
          
          # Return a list of dicts representing the output
          # The input document will be merged into each output doc,
          # so we can access all original fields from the input doc.
          return [{"sales_analysis": "\n".join(report)}]

datasets:
  sales_reports:
    type: file
    source: local
    path: "sales_data/sales_paths.json"
    parsing:
      - function: top_products_report

  receipts:
    type: file
    source: local
    path: "receipts/receipt_paths.json"
    parsing:
      - input_key: pdf_path
        function: paddleocr_pdf_to_string
        output_key: receipt_text
        ocr_enabled: true
        lang: "en"
```

In this configuration:

- We define a custom `top_products_report` function for Excel files.
- We use the built-in `paddleocr_pdf_to_string` parser for PDF files.
- We apply these parsing tools to the external files referenced in the respective datasets.

#### 2. Pipeline Integration

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

### How Data Gets Parsed and Formatted

When you run your DocETL pipeline, the parsing tools you've specified in your configuration file are applied to the external files referenced in your dataset JSONs. Here's what happens:

1. DocETL reads your dataset JSON file.
2. For each entry in the dataset, it looks at the parsing configuration you've specified.
3. It applies the appropriate parsing function to the file path provided in the dataset JSON.
4. The parsing function processes the file and returns the data in a format DocETL can work with (typically a list of strings).

Let's look at how this works for our earlier examples:

#### Excel Files (using top_products_report)

For an Excel file like "sales_data/january_sales.xlsx":

- The top_products_report function reads the Excel file.
- It processes the sales data and generates a report of top-selling products.
- The output might look like this:

```markdown
Top Products Report - January 2023

1. Widget A - 1500 units sold
2. Gadget B - 1200 units sold
3. Gizmo C - 950 units sold
4. Doohickey D - 800 units sold
5. Thingamajig E - 650 units sold
   ...

Total Revenue: $245,000
Best Selling Category: Electronics
```

#### PDF Files (using paddleocr_pdf_to_string)

For a PDF file like "receipts/receipt001.pdf":

- The paddleocr_pdf_to_string function reads the PDF file.
- It uses PaddleOCR to perform optical character recognition on each page.
- The function combines the extracted text from all pages into a single string.
  The output might look like this:

```markdown
RECEIPT
Store: Example Store
Date: 2023-05-15
Items:

1. Product A - $10.99
2. Product B - $15.50
3. Product C - $7.25
4. Product D - $22.00
   Subtotal: $55.74
   Tax (8%): $4.46
   Total: $60.20

Payment Method: Credit Card
Card Number: \***\* \*\*** \*\*\*\* 1234

Thank you for your purchase!
```

This parsed and formatted data is then passed to the respective operations in your pipeline for further processing.

### Running the Pipeline

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
    show_root_heading: true
    heading_level: 3

::: docetl.parsing_tools.txt_to_string
  options:
    show_root_heading: true
    heading_level: 3

::: docetl.parsing_tools.docx_to_string
  options:
    show_root_heading: true
    heading_level: 3

::: docetl.parsing_tools.whisper_speech_to_text
  options:
    show_root_heading: true
    heading_level: 3

::: docetl.parsing_tools.pptx_to_string
  options:
    show_root_heading: true
    heading_level: 3

::: docetl.parsing_tools.azure_di_read
  options:
    heading_level: 3
    show_root_heading: true

::: docetl.parsing_tools.paddleocr_pdf_to_string
  options:
    heading_level: 3
    show_root_heading: true

### Using Function Arguments with Parsing Tools

When using parsing tools in your DocETL configuration, you can pass additional arguments to the parsing functions.

For example, when using the xlsx_to_string parsing tool, you can specify options like the orientation of the data, the order of columns, or whether to process each sheet separately. Here's an example of how to use such kwargs in your configuration:

```yaml
datasets:
  my_sales:
    type: file
    source: local
    path: "sales_data/sales_paths.json"
    parsing_tools:
      - name: excel_parser
        function: xlsx_to_string
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

## Creating Custom Parsing Tools

If the built-in tools don't meet your needs, you can create your own custom parsing tools. Here's how:

1. Define your parsing function in the `parsing_tools` section of your configuration.
2. Ensure your function takes a item (dict) as input and returns a list of items (dicts).
3. Use your custom parser in the `parsing` section of your dataset configuration.

For example:

```yaml
parsing_tools:
  - name: my_custom_parser
    function_code: |
      def my_custom_parser(item: Dict) -> List[Dict]:
          # Your custom parsing logic here
          return [processed_data]

datasets:
  my_dataset:
    type: file
    source: local
    path: "data/paths.json"
    parsing:
      - function: my_custom_parser
```
