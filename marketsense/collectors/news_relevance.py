"""Versioned rules for annotating the relevance of MarketSense news."""
from __future__ import annotations

from dataclasses import dataclass
import re


RELEVANCE_VERSION = "psx-rules-v1"


@dataclass(frozen=True)
class RelevanceDecision:
    status: str  # "include", "review", or "reject"
    reason: str
    topic: str
    content_type: str = "news"
    version: str = RELEVANCE_VERSION

    def article_fields(self) -> dict[str, str]:
        return {
            "relevance_status": self.status,
            "relevance_topic": self.topic,
            "relevance_reason": self.reason,
            "relevance_version": self.version,
        }

# A hard reject is applied only to clearly unrelated headline types. Do not add broad words
# such as "Pakistan", "business", "market", or "economy" here.
_HARD_REJECT = re.compile(
    r"\b(?:obituary|funeral|dies? aged|dies? at the age|passes? away|death of|"
    r"army chief.{0,40}dies|general \(retd\).{0,30}dies|"
    r"gold prices?\b|gold per tola|silver prices?\b|"
    r"travel cost|tourism|holiday bookings|chromebooks?|schoolchildren.{0,30}devices|"
    r"sports?|cricket|football|celebrity|film star|entertainment)\b",
    re.IGNORECASE,
)

_FOREIGN_ONLY = re.compile(
    r"\b(?:india|indian|china|chinese|us|u\.s\.|united states|europe|european|"
    r"japan|japanese|uk|britain|british)\b.{0,100}\b(?:bonds?|bond yields?|stocks?|shares?|"
    r"equities|central bank|markets?)\b",
    re.IGNORECASE,
)

_GENERIC_BILATERAL = re.compile(
    r"\b(?:explore new avenues|explore avenues|bilateral ties|bilateral cooperation|"
    r"trade and investment cooperation|food security partnership|offers? .{0,35}partnership|"
    r"strengthen .*? relations|boost .*? ties|diplomatic relations)\b",
    re.IGNORECASE,
)

_PAKISTAN_CONTEXT = re.compile(
    r"\b(?:pakistan|pakistani|pakistan's|psx|kse\s*[- ]?100|sbp|state bank of pakistan|"
    r"fbr|secp|imf|pkr|pakistani rupee|rupee|sindh|punjab|balochistan|khyber pakhtunkhwa)\b",
    re.IGNORECASE,
)

_DIRECT_PSX = re.compile(
    r"\b(?:psx|kse\s*[- ]?100|kse\s*[- ]?30|pakistan stock exchange|"
    r"pakistan stock market|pakistani equities|stocks? at the bourse|"
    r"shares? at the bourse|selling at (?:the )?bourse|trading at (?:the )?bourse|"
    r"market breadth|market turnover|index points?)\b",
    re.IGNORECASE,
)

_SBP_POLICY = re.compile(
    r"\b(?:sbp|state bank of pakistan|mpc|monetary policy committee)\b.{0,100}\b"
    r"(?:policy rate|interest rates?|monetary policy|rate decision|rate cut|rate hike|"
    r"cut rates|hike rates|hold rates|monetary tightening|monetary easing)\b|"
    r"\b(?:policy rate|interest rate decision|rate cut|rate hike|monetary tightening|"
    r"monetary easing)\b.{0,100}\b(?:sbp|state bank of pakistan|mpc|monetary policy committee)\b",
    re.IGNORECASE,
)

# Macro signals that can be directly relevant when the headline clearly anchors them to Pakistan.
_PAKISTAN_MACRO = re.compile(
    r"\b(?:inflation|consumer price|cpi|gdp|policy rate|interest rates?|monetary tightening|"
    r"monetary easing|budget deficit|fiscal deficit|current account|trade deficit|trade gap|"
    r"foreign exchange reserves|forex reserves|exchange rate|devaluation|depreciation|"
    r"pakistani rupee|pkr|rupee|exports?|imports?|remittances?|circular debt|"
    r"power tariff|electricity tariff|petrol prices?|diesel prices?|fuel prices?)\b",
    re.IGNORECASE,
)

# FBR/tax/privatisation and sector-policy stories can matter, but headline-only evidence is
# often insufficient. Route these to manual review unless a direct PSX/company signal applies.
_REVIEW_POLICY = re.compile(
    r"\b(?:fbr reforms?|tax reforms?|tax policy|auto policy|privati[sz]ation|"
    r"privati[sz]e|privati[sz]ation entity|state[- ]owned enterprises?|soes?|"
    r"sme lending|small and medium enterprises?|credit conditions|farm size|"
    r"agriculture policy|oilseeds?|wheat price|food prices?|expats?|remittances?|"
    r"exports? matter|trade cooperation|investment cooperation|launches? .*? in pakistan|"
    r"stakeholders? see|voices? concern|concern over|editorial|opinion|why |how )\b",
    re.IGNORECASE,
)

_COMPANY_EVENT = re.compile(
    r"\b(?:acquisition|acquires?|acquired|merger|merges?|divestment|divests?|"
    r"financial results|corporate results|quarterly results|annual results|earnings|"
    r"profit rises?|profit falls?|loss widens?|dividend|bonus shares|buyback|"
    r"board approves?|shareholders approve?|capacity expansion|plant shutdown|"
    r"production halt|default|debt restructuring|contract worth|investment of|"
    r"capital expenditure|raises? funds|rights issue|revenue rises?|revenue falls?|"
    r"appoints? .*? ceo|ceo appointed|appoints? .{0,40}chief executive|posts? (?:a |an )?(?:(?:quarterly|annual|net) )?profit|material agreement|power purchase agreement)\b",
    re.IGNORECASE,
)

_EDITORIAL = re.compile(r"^\s*(?:editorial|opinion|analysis|commentary)\s*[:—-]", re.IGNORECASE)


def classify_news(headline: str, summary: str = "", categories: tuple[str, ...] | list[str] = (), *, has_company: bool = False) -> RelevanceDecision:
    """Return a conservative decision using the headline as the primary evidence.

    The summary parameter is retained for collector compatibility but is deliberately not
    used to promote a headline: RSS summaries can contain boilerplate or unrelated page text.
    Category metadata identifies opinion/editorial content. Company evidence comes from the stock index.
    """
    title = " ".join((headline or "").split()).strip()
    if not title:
        return RelevanceDecision("reject", "missing headline", "unknown")

    category_text = " ".join(str(x) for x in categories).lower()
    editorial = bool(_EDITORIAL.search(title) or re.search(r"\b(?:editorial|opinion|commentary)\b", category_text))

    if _HARD_REJECT.search(title):
        return RelevanceDecision("reject", "clearly non-market headline", "non_market", "editorial" if editorial else "news")

    if _FOREIGN_ONLY.search(title) and not _PAKISTAN_CONTEXT.search(title) and not _DIRECT_PSX.search(title):
        return RelevanceDecision("reject", "foreign-market story without Pakistan-market link", "foreign_market")

    if _GENERIC_BILATERAL.search(title) and not _DIRECT_PSX.search(title) and not has_company:
        return RelevanceDecision("reject", "generic diplomatic/trade announcement without material market evidence", "bilateral_announcement")

    if _DIRECT_PSX.search(title):
        if editorial:
            return RelevanceDecision("review", "editorial about PSX; separate opinion from reported news", "psx", "editorial")
        return RelevanceDecision("include", "explicit PSX/KSE market signal", "psx")

    if _SBP_POLICY.search(title):
        if editorial:
            return RelevanceDecision("review", "editorial on monetary policy", "monetary_policy", "editorial")
        return RelevanceDecision("include", "explicit SBP/MPC monetary-policy signal", "monetary_policy")

    if has_company and _COMPANY_EVENT.search(title):
        if editorial:
            return RelevanceDecision("review", "company-related editorial; verify factual event", "corporate", "editorial")
        return RelevanceDecision("include", "named PSX-listed company and material corporate event", "corporate")

    has_pakistan = bool(_PAKISTAN_CONTEXT.search(title))
    has_macro = bool(_PAKISTAN_MACRO.search(title))
    if has_pakistan and has_macro:
        if editorial:
            return RelevanceDecision("review", "macro opinion/editorial; verify underlying data and publication time", "macro", "editorial")
        # Trade/remittance/export headlines can be useful, but broad partner-country claims
        # are held for review unless the headline itself supplies a measurable figure. This
        # prevents a generic bilateral/trade headline from becoming an automatic model input.
        if re.search(r"\b(?:matter more|why |how |concern|stakeholders|explore|cooperation)\b", title, re.I):
            return RelevanceDecision("review", "macro topic appears, but headline is commentary or exploratory", "macro")
        if re.search(r"\b(?:trade gap|trade deficit|exports?|imports?|remittances?)\b", title, re.I):
            has_measure = bool(re.search(r"(?:\b\d+(?:[.,]\d+)?\s*(?:%|pc|percent|bn|billion|mn|million|trillion)\b|\b(?:record|all[- ]time high|surge[sd]?|slump(?:ed)?|soar(?:ed)?|plung(?:e|ed)|rise[sd]?|fell|fall[es]?|increase[sd]?|decreas(?:e|ed)|narrow(?:s|ed)?|widen(?:s|ed)?)\b)", title, re.I))
            if not has_measure:
                return RelevanceDecision("review", "trade/flow topic needs a measurable figure or article-level verification", "macro")
            if re.search(r"\b(?:with|between|among)\s+(?:the )?(?:gulf|china|qatar|saudi|uae|europe|india)\b", title, re.I) and not re.search(r"\d", title):
                return RelevanceDecision("review", "partner-specific trade claim needs article-level verification", "macro")
        return RelevanceDecision("include", "Pakistan-specific macroeconomic or monetary indicator", "macro")

    if _REVIEW_POLICY.search(title):
        return RelevanceDecision("review", "potential policy/sector relevance; direct equity-market impact not established", "policy_or_sector", "editorial" if editorial else "news")

    # Agriculture, bilateral investment, SME finance, and sector launches may matter in some
    # cases, but are not auto-admitted without a named listed company or measurable macro signal.
    if re.search(r"\b(?:agriculture|farm|wheat|oilseed|sme|small business|investment|trade|auto|automaker|automotive|privati[sz]ation)\b", title, re.I):
        return RelevanceDecision("review", "possible sector/economic relevance; listed-company or measurable market link needed", "sector_or_economy")

    return RelevanceDecision("reject", "no explicit PSX, listed-company, or Pakistan macro-market link", "general_news")
