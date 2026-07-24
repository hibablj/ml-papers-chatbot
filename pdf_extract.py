import fitz 

def extract_text_from_pdf(pdf_path):
    """ Opens a PDF and returns its stext, page by page . 
        Returns a list of strings. One per page."""
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)): 
        page = doc[page_num]
        text = page.get_text()
        pages_text.append(text)

    doc.close()
    return pages_text 

if __name__ == "__main__":
    
    pages = extract_text_from_pdf("/Users/hibabelhaj/Desktop/1706.03762v7.pdf")

    print(f"Extracted {len(pages)} pages.")
    print("First 500 characters of page 1 ")
    print(pages[0][:500])