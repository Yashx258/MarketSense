import pytest

from marketsense.sentiment.text_hygiene import clean_article_text


@pytest.mark.parametrize("footer", [
    "No comments yet. Be the first to join the discussion! Powered by Mindblaze Technologies.",
    "No comments yet.\r\nBe the first to join the discussion!\r\n"
    "Leave a comment\r\nCopyright 2026\r\nPowered by Mindblaze Technologies.\r\n",
    "NO COMMENTS YET! Be the first to join the discussion! Powered by\xa0Mindblaze Technologies",
    "No comments yet Be the first to join the discussion! Powered by Mindblaze Technologies.",
])
@pytest.mark.parametrize("separator", [" ", "\n\n"])
def test_mindblaze_suffix(footer, separator):
    assert clean_article_text(f"Market turnover rose.{separator}{footer}") == "Market turnover rose."
    assert clean_article_text(footer) == ""


@pytest.mark.parametrize("footer", [
    "The writer is a Lahore-based economist.",
    "The author works at a research institute.",
    "About the author: Sara covers monetary policy.",
    "[email protected]",
    "Email: [EMAIL PROTECTED].",
    "The writer teaches economics. [email protected]",
    "— Reuters",
    "(AFP)",
    "Source: Associated Press of Pakistan",
    "Reporting by Sara Ahmed; Editing by Ali Khan.",
    "Copyright 2026 Reuters. All rights reserved.",
])
def test_author_and_agency_footers(footer):
    text = f"Market turnover rose.\n\n{footer}\n"
    assert clean_article_text(text) == "Market turnover rose."


def test_stacked_footers_are_removed():
    text = (
        "Market turnover rose. The writer is an economist. [email protected]\n"
        "— Reuters\nNo comments yet. Be the first to join the discussion! "
        "Powered by Mindblaze Technologies."
    )
    cleaned = clean_article_text(text)
    assert cleaned == "Market turnover rose."
    assert clean_article_text(cleaned) == cleaned


@pytest.mark.parametrize("article", [
    "",
    "  Market turnover rose.\r\n\r\nBanks  led the gains.\n",
    "Reuters reported that market turnover rose.",
    "Send submissions to [email protected] by Friday.",
    "The writer is an economist. Her study found that market turnover rose.",
    "No comments yet. Be the first to join the discussion! Trading resumes on Tuesday.",
])
def test_clean_article_passes_through_verbatim(article):
    assert clean_article_text(article) == article
