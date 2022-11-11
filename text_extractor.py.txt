from tika import parser

def text_extractor(pdf_file):
  raw = parser.from_file(pdf_file)
  return raw['content']