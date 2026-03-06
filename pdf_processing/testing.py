from zipfile import ZipFile
import pypdf
import pytesseract
from pdf2image import convert_from_path

# with ZipFile('../data/domain_2/dev.zip') as zf:
#     for file in zf.namelist():
#         print(f"{file} ##############################")
#         if not file.endswith('.pdf'):
#             continue
#         # domain_1
#         # if file in ['dev/759dfc8486c0f02391d7cfc1fed753b0608fc601.pdf',
#         #             'dev/41b8a1637eea9e8acb06eef16b1c2479e37e759c.pdf',
#         #             'dev/54a634188fe502d79e2653231c8ea286636142a0.pdf',
#         #             'dev/72e790ea927cb79b08838bcd0e290c83fd4f6e06.pdf',
#         #             'dev/b59848334218c38574465a72b75d6b4744bb985f.pdf',
#         #             'dev/61990b93694d93444332fc6fa6a0f837ee312bdb.pdf',
#         #             'dev/cadef3fc7594662353b7c94387b63a926b5d92d3.pdf',
#         #             'dev/a066d97ba4ff355ede19dabfdf01860cfa04a6de.pdf',
#         #             'dev/31e80927922c1ffb86a2fcf32c073097d8f4ac72.pdf',
#         #             'dev/0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.pdf',
#         #             'dev/ac85a12a4483579ba8f51197b5c8ace05f5397e0.pdf',
#         #             ]:
#         # domain_2
#         if file in [
#                     'dev/7db5015ac19c3cd7bb04118b38a228b3572a358c.pdf',
#                     'dev/2ee44110e9b8b4b6f0e41ef9c8cc553be744551b.pdf',
#                     'dev/828b7ab8d9ead3e6f25ce173cd6f03cce5c800c1.pdf',
#                     'dev/5c75a2eb4814242f35871112bc657e36f813674b.pdf',
#                     'dev/67bfbcbe3cfd8f845785a8664b9b73b6b4751d72.pdf',
#                     'dev/114aa3c7890f255d1134a31a22a29caae427a6f1.pdf',
#                     'dev/60f2d879c17e5e6967e40c2315b8d46a987bda47.pdf',
#                     'dev/4e779acee13fa6e0763fb33d1c83030b8e6ea33d.pdf',
#                     'dev/780fd55597cd81f1cf0d1492926ab69cfb115d89.pdf',
#                     'dev/cc340c81af3efa6ad4f864b38c4b1c497562df63.pdf',
#                     'dev/fb285f08989b559f0080c9dd096d3557860f919d.pdf',
#                     'dev/51fc038fd5ddeade0c2c261e2b56508b287bab98.pdf',
#                     'dev/556be4e553eab6942598e58cfa02b6aaa553a0e5.pdf',
#                     'dev/1026d4282a0fb8d22694301adc83729950c8498b.pdf',
#                     'dev/2827fbd42bc74ef48ac74ef40b90868b94b1da79.pdf',
#                     'dev/aba537442395c1a4ca13d05e378a49a47865cc46.pdf',
#                     'dev/9d2df2f02cd5d57fe401497c3e64e47a3994112d.pdf',
#                     'dev/20e086e7585d5e496d67ed2313ceb5aa53b409b5.pdf',
#                     'dev/a8bf09b126fb7c9072a5a54c3c0aaa5df7968ea4.pdf',
#                     'dev/f8e8ea2b7e82798862bc3d7b2753f2262642d889.pdf',
#                     'dev/5ae9814fac3ba1f5048ce716f15304d3ead03823.pdf',
#                     'dev/f548cde5ba56c4e1d8aeca95c3e190f2711d6fa1.pdf',
#                     'dev/3828dec85bc546bd064190a023c7c802d6a8ab0a.pdf',
#                     'dev/b0e35f26831baa753925da047ca36d318a100825.pdf',
#                     'dev/1caa31f28b0feabbe319d352bf58cb08adcc4dc8.pdf',
#                     'dev/d80178b24cc3a40d841ec790e575767afd0a3fd9.pdf',
#                     'dev/fc5f5d668b03d9b606779484d465efcb4436a6be.pdf',
#                     'dev/60d1bee90467207fbbac980b346b932229542c73.pdf',
#                     'dev/986dbec74e7ac870dfcf8b14337b301227c61238.pdf',
#                     'dev/f4539bc3a79016285524705e25fee5a8d523db3d.pdf',
#                     ]:
#             continue
#         with zf.open(file) as f:
#             reader = pypdf.PdfReader(f)
#             # print(reader.pages[1].extract_text())
#             for page in reader.pages:
#                 if len(page.extract_text()) < 100:
#                     print(f"WARNING: short text")
#                 print(f'page {page.page_number+1} ######################')
#         break

pdf = pypdf.PdfReader(open('../data/domain_1/dev/0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.pdf', 'rb'))
print(pdf.pages[25].extract_text())


def extract_text_from_scanned_page(pdf_path, page_num):
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    text = pytesseract.image_to_string(images[0], lang='ukr')
    print(text)

extract_text_from_scanned_page('../data/domain_1/dev/0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.pdf', 26)



# Keep pypdf as the main extractor.
#
# Run a per-page “is this empty?” check.
#
# For the 3 image-only pages:
#
# Use pdf2image + pytesseract with lang="ukr+eng".
#
# Store OCR text per page.

##############
# pdfplumber extracts tables into lists of rows
# convert tables to value tuples
# simple heuristics to figure out newlines:
    # should be left a separate lines
    # text should be in one line (replace with space)
    # lines are a list


# later:
# mark headers and lists using markdown -- hard to figure out
# use heuristics to figure out if table is vertical or horizontal

