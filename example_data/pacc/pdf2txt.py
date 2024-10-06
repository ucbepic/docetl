import os

import fitz  # PyMuPDF

def pdf_to_text(pdf_path, txt_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    text = ""
    
    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    
    # Write the text to a file
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

def convert_all_pdfs_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            txt_path = os.path.splitext(pdf_path)[0] + '.txt'
            pdf_to_text(pdf_path, txt_path)
            print(f"Converted {pdf_path} to {txt_path}")

if __name__ == "__main__":
    directory = "./pdfs/2020"
    convert_all_pdfs_in_directory(directory)