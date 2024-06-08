"""
Simple CLI commands.
"""

import click
from flask import Blueprint

from lute.cli.language_term_export import generate_language_file, generate_book_file
from lute.cli.import_books import import_books_from_csv, import_books_from_yt
from lute.cli.korean_term_analysis import run as kta_run
from lute.cli.anki_export import generate_anki_package
from lute.cli.apply_tag import apply_tag, apply_min_status

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


@bp.cli.command("import_books_from_csv")
@click.option(
    "--commit",
    is_flag=True,
    help="""
    Commit the changes to the database. If not set, import in dry-run mode. A
    list of changes will be printed out but not applied.
""",
)
@click.option(
    "--tags",
    default="",
    help="""
    A comma-separated list of tags to apply to all books.
""",
)
@click.option(
    "--language",
    default="",
    help="""
    The name of the default language to apply to each book, as it appears in
    your language settings. If unset, the language must be indicated in the
    "language" column of the CSV file.
""",
)
@click.argument("file")
def import_books_from_csv_cmd(language, file, tags, commit):
    """
    Import books from a CSV file.

    The CSV file must have a header row with the following, case-sensitive,
    column names. The order of the columns does not matter. The CSV file may
    include additional columns, which will be ignored.

      - title: the title of the book

      - text: the text of the book

      - language: [optional] the name of the language of book, as it appears in
      your language settings. If unspecified, the language specified on the
      command line (using the --language option) will be used.

      - url: [optional] the source URL for the book

      - tags: [optional] a comma-separated list of tags to apply to the book
      (e.g., "audiobook,beginner")

      - audio: [optional] the path to the audio file of the book. This should
      either be an absolute path, or a path relative to the CSV file.

      - bookmarks: [optional] a semicolon-separated list of audio bookmark
      positions, in seconds (decimals permitted; e.g., "12.34;42.89;89.00").
    """
    tags = list(tags.split(",")) if tags else []
    import_books_from_csv(file, language, tags, commit)


@bp.cli.command("korean_term_analysis")
@click.option("--commit", is_flag=True)
@click.option("--book", default=0)
@click.option("--include_books", default="")
@click.option("--exclude_books", default="")
@click.option("--trace", default="")
@click.option("--files", default="")
def korean_term_analysis(book, include_books, exclude_books, files, commit, trace):
    """
    Analyzes undefined terms in all Korean books.
    """
    include_books = [int(id.strip()) for id in include_books.split(',')] if include_books else []
    exclude_books = [int(id.strip()) for id in exclude_books.split(',')] if exclude_books else []
    if book:
        book = int(book)
        if book not in include_books:
            include_books.append(book)
    files = files.split(',') if files else []
    kta_run(include_books, exclude_books, files, commit, trace)


@bp.cli.command("anki_export")
@click.option("--guids", default="")
@click.argument("output_path")
def anki_export(guids, output_path):
    """
    Export terms for a language as an Anki package.
    """
    generate_anki_package("foo", output_path, guids)


@bp.cli.command("apply_tag")
@click.option("--commit", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--remove", is_flag=True)
@click.argument("language")
@click.argument("tag")
@click.argument("words_file")
def apply_tag_cmd(language, words_file, tag, commit, verbose, remove):
    """
    Apply a tag to all words in a list.
    """
    apply_tag(language, words_file, tag, commit, verbose, remove)


@bp.cli.command("apply_min_status")
@click.option("--commit", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.argument("language")
@click.argument("status")
@click.argument("words_file")
def apply_min_status_cmd(language, words_file, status, commit, verbose):
    """
    Apply a minimum status to all words in a list.
    """
    apply_min_status(language, words_file, int(status), commit, verbose)


@bp.cli.command("import_books_from_yt")
@click.option("--commit", is_flag=True)
@click.option("--recursive", is_flag=True)
@click.option("--tags", default="")
@click.option("--sort", default="")
@click.option("--sort_numeric", is_flag=True)
@click.argument("language")
@click.argument("lc")
@click.argument("spec")
def import_books_from_yt_cmd(language, lc, spec, tags, sort, sort_numeric, recursive, commit):
    """
    Import books from a paths containing youtube downloads.
    """
    tags = [tag for tag in tags.split(',')] if tags else []
    import_books_from_yt(language, lc, spec, tags, sort, sort_numeric, recursive, commit)
