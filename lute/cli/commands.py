"""
Simple CLI commands.
"""

import click
from flask import Blueprint

from lute.cli.language_term_export import generate_language_file, generate_book_file
from lute.cli.korean_term_analysis import run as kta_run

bp = Blueprint("cli", __name__)


@bp.cli.command("hello")
def hello():
    "Say hello -- proof-of-concept CLI command only."
    msg = """
    Hello there!

    This is the Lute cli.

    There may be some experimental scripts here ...
    nothing that will change or damage your Lute data,
    but the CLI may change.

    Thanks for looking.
    """
    print(msg)


@bp.cli.command("language_export")
@click.argument("language")
@click.argument("output_path")
def language_export(language, output_path):
    """
    Get all terms from all books in the language, and write a
    data file of term frequencies and children.
    """
    generate_language_file(language, output_path)


@bp.cli.command("book_term_export")
@click.argument("bookid")
@click.argument("output_path")
def book_term_export(bookid, output_path):
    """
    Get all terms for the given book, and write a
    data file of term frequencies and children.
    """
    generate_book_file(bookid, output_path)


@bp.cli.command("korean_term_analysis")
@click.option("--commit", is_flag=True)
@click.option("--book", default=0)
@click.option("--include_books", default="")
@click.option("--exclude_books", default="")
@click.option("--trace", default="")
@click.argument("todo_path")
def korean_term_analysis(todo_path, book, include_books, exclude_books, commit, trace):
    """
    Analyzes undefined terms in all Korean books.
    """
    include_books = [int(id.strip()) for id in include_books.split(',')] if include_books else []
    exclude_books = [int(id.strip()) for id in exclude_books.split(',')] if exclude_books else []
    if book:
        book = int(book)
        if book not in include_books:
            include_books.append(book)
    kta_run(todo_path, include_books, exclude_books, commit, trace)
