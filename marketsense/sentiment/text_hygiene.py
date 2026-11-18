"""Remove publisher boilerplate from article text before scoring."""

import re


_COMMENT_FOOTER = re.compile(
    r"\bno\s+comments\s+yet\s*[.!]?\s*be\s+the\s+first\s+to\s+join\s+the\s+discussion\b"
    r".*?\bpowered\s+by\s+mindblaze\s+technologies\s*[.!]?\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Stored bodies may be flattened. Recognize a footer at a newline or after
# sentence-ending punctuation, as well as when the whole body is boilerplate.
_FOOTER_START = r"(?:\A|[\r\n]|(?<=[.!?])\s+)\s*"
_AGENCY = r"(?:reuters|associated press(?: of pakistan)?|afp|app|ap|bloomberg|news agencies|agencies)"
_FOOTER = re.compile(
    _FOOTER_START
    + r"(?:"
    r"(?:the\s+)?(?:writer|author)\s+(?:is|was|teaches|writes|works|can\s+be\s+reached)\b"
    r"[^\r\n.!?]*[.!?]?|"
    r"(?:about\s+the\s+author\s*:\s*)[^\r\n]+|"
    r"(?:(?:e-?mail|contact)\s*:\s*)?\[email\s+protected\][.!]?|"
    r"(?:reporting|additional reporting|writing|editing)\s+by\s+[^\r\n.!?]*[.!?]?|"
    rf"(?:(?:source|with\s+inputs\s+from|with\s+additional\s+reporting\s+from)\s*:?\s*)"
    rf"{_AGENCY}[.!]?|"
    rf"(?:©|copyright)\s*(?:\d{{4}}\s+)?{_AGENCY}\b\.?\s*(?:all rights reserved\.?)?"
    r")\s*\Z",
    re.IGNORECASE,
)
_AGENCY_CREDIT = re.compile(
    rf"(?:\A|\s*\n\s*|\s*[-–—]+\s*|\s+(?=\())\(?{_AGENCY}\)?[.!]?\s*\Z",
    re.IGNORECASE,
)


def clean_article_text(text: str) -> str:
    """Strip trailing publisher boilerplate while preserving other text verbatim."""
    while True:
        for pattern in (_COMMENT_FOOTER, _FOOTER, _AGENCY_CREDIT):
            match = pattern.search(text)
            if match:
                text = text[:match.start()].rstrip()
                break
        else:
            return text
