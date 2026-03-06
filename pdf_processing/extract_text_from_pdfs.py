import pdfplumber
import re
import pytesseract
import pdf2image
import json


def extract_text_from_scanned_page(pdf_path, page_num):
    images = pdf2image.convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    text = pytesseract.image_to_string(images[0], lang='ukr')
    return text


def table_list_to_tuples(table):
    table_repr = ""

    # horizontal headers
    possible_headers = table[0]
    for row in table[1:]:
        if None in row:
            continue
        for ind, header in enumerate(possible_headers):
            table_repr += f"{header}: {row[ind]}\n\n"
        table_repr += "\n\n"

    # vertical headers
    for row in table:
        if None in row:
            continue
        table_repr += f"{row[0]}: {", ".join(row[1:])}\n\n"

    return table_repr

def process_table(table):
    table = table.extract()
    if table:
        return table_list_to_tuples(table)

def line_position_relative_to_bbox(line, bbox, tol=3):
    """
    Returns one of: "above", "inside", "below"
    based on vertical position relative to bbox.
    """
    _, bbox_top, _, bbox_bottom = bbox
    line_top = line["top"]
    line_bottom = line["bottom"]

    if line_bottom <= bbox_top + tol:
        return "above"

    if line_top >= bbox_bottom - tol:
        return "below"

    return "inside"

def append_line(line, prev_line, page_text):
    if prev_line == None:
        page_text += f"{line['text']}"

    else:
        # very simple heuristic
        this_line_first_char = line["text"][0]
        prev_line_last_char = prev_line["text"][-1]
        if this_line_first_char.islower() and (prev_line_last_char.islower() or prev_line_last_char in ",-:»–-"):
            page_text += f" {line['text']}"
        else:
            page_text += f"\n{line['text']}"

    return page_text

def process_pdfplumber_page(page, prev_line):
    final_page_text = ""

    tables = page.find_tables()
    lines = page.extract_text_lines()

    # delete page number at the top
    if re.match(r'\d{1,3}', lines[0]["text"]):
        lines = lines[1:]

    top_table = tables.pop(0) if tables else None
    top_table_processed = False

    for line in lines:
        skip_line = False

        # check if still have unprocessed tables
        if top_table:
            line_position = line_position_relative_to_bbox(line, top_table.bbox)
            if line_position == "inside":
                # if table not yet processed, get text tuples
                if not top_table_processed:
                    final_page_text += '\n\n'
                    final_page_text += process_table(top_table)
                    top_table_processed = True

                skip_line = True

            elif line_position == "below":
                top_table = tables.pop(0) if tables else None
                top_table_processed = False

                if top_table:
                    line_position = line_position_relative_to_bbox(line, top_table.bbox)
                    if line_position == "inside":
                        # if table not yet processed, get text tuples
                        if not top_table_processed:
                            final_page_text += '\n\n'
                            final_page_text += process_table(top_table)
                            top_table_processed = True

                        skip_line = True

        if not skip_line:
            # process line
            final_page_text = append_line(line, prev_line, final_page_text)

            prev_line = line

        # in case page ends with a table
        else:
            bbox_x0, bbox_top, bbox_x1, bbox_bottom = top_table.bbox
            prev_line = {'text': '\n',
                         'top': bbox_top,
                         'bottom': bbox_bottom,
                         'x0': bbox_x0,
                         'x1': bbox_x1}

    return final_page_text, prev_line

def process_pytesseract_page(page_string, prev_line):
    # can't identify table from a scanned page without CV
    # simple logic for now

    lines = page_string.split('\n')
    if re.match(r'\d{1,3}', lines[0]):
        lines = lines[1:]

    final_page_text = ""

    for line in lines:
        if not line:
            final_page_text += '\n'
            prev_line = {"text": '\n'}
        else:
            final_page_text = append_line({"text": line}, prev_line, final_page_text)
            prev_line = {"text": line}

    return final_page_text, prev_line


def process_document(pdf_path, output_path):
    with pdfplumber.open(pdf_path) as pdf:
        document_text = ""
        page_char_ranges = {ind+1: tuple() for ind in range(len(pdf.pages))}

        prev_line = None
        for page in pdf.pages:
            pdfplumber_found_text = page.extract_text()

            if len(pdfplumber_found_text) == 0:
                # if no text, check if page is scanned
                pytesseract_found_text = extract_text_from_scanned_page(pdf_path, page.page_number)
                if len(pytesseract_found_text) > 0:
                    page_text, prev_line = process_pytesseract_page(pytesseract_found_text, prev_line)
                # if pytesseract returned nothing: page is empty
                else:
                    page_text = ''

            else:
                page_text, prev_line = process_pdfplumber_page(page, prev_line)

            page_char_ranges[page.page_number] = (len(document_text), len(document_text) + len(page_text))
            document_text += page_text

    file_name = pdf_path.split('/')[-1].split('.')[0]

    with open(f"{output_path}/{file_name}_text.txt", 'w') as output_file:
        output_file.write(document_text)

    with open(f"{output_path}/{file_name}_page_ranges.json", 'w') as output_file:
        output_file.write(json.dumps(page_char_ranges))

def get_text_page(processed_files_path, pdf_name, snippet_range):
    page_char_ranges = json.load(open(f"{processed_files_path}/{pdf_name}_page_ranges.json"))
    for page_number, page_range in page_char_ranges.items():
        page_number = int(page_number)

        # snippet starts on this page
        if page_range[0] < snippet_range[0] < page_range[1]:
            # snippet ends on this page
            if snippet_range[1] < page_range[1]:
                return page_number
            # snippet ends on the next page
            else:
                this_page_snippet_length = page_range[1] - snippet_range[0]
                next_page_snippet_length = snippet_range[1] - snippet_range[0] - this_page_snippet_length
                if this_page_snippet_length > next_page_snippet_length:
                    return page_number
                else:
                    return page_number + 1

if __name__ == '__main__':
    import os

    pdf_path = '../data/domain_2/dev'
    output_path = '../data/output/domain_2'

    for pdf_name in os.listdir(pdf_path):
        if pdf_name.endswith('.pdf'):
            process_document(f"{pdf_path}/{pdf_name}", output_path)
            print(pdf_name)
