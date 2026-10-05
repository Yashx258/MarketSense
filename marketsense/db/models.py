"""SQLAlchemy 2.0 models. Mirror sql/schema.sql (schema.sql is the source of truth for production)."""
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from sqlalchemy import (BigInteger, Boolean, CheckConstraint, Date, ForeignKey, Index, Integer, Numeric,
                        String, Text, TIMESTAMP, UniqueConstraint, func, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PK = BigInteger().with_variant(Integer, "sqlite")   # lets SQLite autoincrement in unit tests
TS = TIMESTAMP(timezone=True)


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"
    stock_id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    is_kse100: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", server_default="active")
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), onupdate=func.now(), nullable=False)
    __table_args__ = (CheckConstraint("status IN ('active','inactive','delisted')", name="ck_stock_status"),)


class PriceHistory(Base):
    __tablename__ = "price_history"
    price_id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.stock_id", ondelete="CASCADE"), nullable=False)
    trading_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    high: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    low: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    close: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="psxdata", server_default="psxdata")
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    __table_args__ = (UniqueConstraint("stock_id", "trading_date", name="uq_price_stock_date"),
                      Index("ix_price_date", "trading_date"))


class NewsArticle(Base):
    __tablename__ = "news_articles"
    news_id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    stock_id: Mapped[int | None] = mapped_column(ForeignKey("stocks.stock_id", ondelete="SET NULL"))
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    article_text: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(TS)
    retrieved_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en", server_default="en")
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    __table_args__ = (Index("uq_news_url", "url", unique=True,
                            postgresql_where=text("url IS NOT NULL"), sqlite_where=text("url IS NOT NULL")),)


class Fundamental(Base):
    __tablename__ = "fundamentals"
    fundamental_id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.stock_id", ondelete="CASCADE"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    pe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    eps: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    market_cap: Mapped[Decimal | None] = mapped_column(Numeric(24, 2))
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    __table_args__ = (UniqueConstraint("stock_id", "as_of_date", "source", name="uq_fund_stock_date_source"),)


class TechnicalFeature(Base):
    __tablename__ = "technical_features"
    technical_id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.stock_id", ondelete="CASCADE"), nullable=False)
    trading_date: Mapped[date] = mapped_column(Date, nullable=False)
    rsi: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    macd: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    macd_signal: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    macd_histogram: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    __table_args__ = (UniqueConstraint("stock_id", "trading_date", name="uq_tech_stock_date"),)


class SentimentFeature(Base):
    __tablename__ = "sentiment_features"
    sentiment_id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    news_id: Mapped[int] = mapped_column(ForeignKey("news_articles.news_id", ondelete="CASCADE"), nullable=False)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.stock_id", ondelete="CASCADE"), nullable=False)
    positive_score: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    neutral_score: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    negative_score: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(10), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(TS, server_default=func.now(), nullable=False)
    __table_args__ = (UniqueConstraint("news_id", "model_name", name="uq_sent_news_model"),)
