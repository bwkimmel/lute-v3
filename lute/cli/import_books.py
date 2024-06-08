"""
Bulk import books.
"""

import csv
import os
import re
import sys

from glob import glob
from lute.book import service
from lute.book.forms import NewBookForm
from lute.book.model import Book, Repository
from lute.db import db
from lute.models.language import Language
from werkzeug.datastructures import FileStorage
from flask_wtf.file import FileField


YOUTUBE_ID_RE = "\\[([A-Za-z0-9_-]{11})]"


def import_books_from_yt(language, lc, spec, tags, sort_re, sort_numeric, recursive, commit):
    if 'youtube' not in tags:
        tags.append('youtube')
    print("Will add tags: {}".format(tags))
    repo = Repository(db)
    vtt_ext = ".{}.vtt".format(lc)

    books = []
    lang = Language.find_by_name(language)
    for path in glob(spec, recursive=recursive):
        fn = os.path.basename(path)
        if not path.endswith(vtt_ext):
            continue
        matches = re.findall(YOUTUBE_ID_RE, fn)
        if not matches or len(matches) != 1:
            print("Cannot determine Youtube ID for: ".format(path))
            continue
        video_id = matches[0]
        url = "https://www.youtube.com/watch?v={}".format(video_id)

        title = fn[:-len(vtt_ext)]
        if repo.find_by_title(title, lang.id) is not None:
            print("Already exists: {}".format(title))
            continue

        mp3_fn = "{}.mp3".format(title)
        mp3_path = os.path.join(os.path.dirname(path), mp3_fn)

        text = ""
        with open(path, 'rb') as f:
            fs = FileStorage(name=fn, filename=fn, stream=f)
            text = service.get_file_content(fs)

        audio_file = None
        if commit and os.path.exists(mp3_path):
            with open(mp3_path, 'rb') as f:
                fs = FileStorage(name=mp3_fn, filename=mp3_fn, stream=f)
                audio_file = service.save_audio_file(fs)

        book = Book()
        book.language_name = language
        book.title = title
        book.text = text
        book.source_uri = url
        book.book_tags = tags
        book.audio_filename = audio_file
        books.append(book)
        print("Added book (text length: {}, url: {}): {}".format(len(book.text), book.source_uri, book.title))

    print()
    print("Added {} books".format(len(books)))
    print()

    if sort_re:
        def key(book):
            m = re.findall(sort_re, book.title)
            if m:
                if sort_numeric:
                    return int(m[0])
                else:
                    return str(m[0])
            else:
                if sort_numeric:
                    return 0
                else:
                    return ""
        books = sorted(books, key=key)

        print()
        print("Sort order:")
        for book in books:
            print("  {}".format(book.title))
    
    if not commit:
        print("Dry run, no changes made.")
        return

    for book in books:
        repo.add(book)

    print("Committing...")
    sys.stdout.flush()
    repo.commit()


def import_books_from_csv(file, language, tags, commit):
    """
    Bulk import books from a CSV file.

    Args:

      file:     the path to the CSV file to import (see lute/cli/commands.py
                for the requirements for this file).
      language: the name of the language to use by default, as it appears in
                your languages settings
      tags:     a list of tags to apply to all books
      commit:   a boolean value indicating whether to commit the changes to the
                database. If false, a list of books to be imported will be
                printed out, but no changes will be made.
    """
    repo = Repository(db)
    count = 0
    with open(file, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            book = Book()
            book.title = row["title"]
            book.language_name = row.get("language") or language
            if not book.language_name:
                print(f"Skipping book with unspecified language: {book.title}")
                continue
            lang = Language.find_by_name(book.language_name)
            if not lang:
                print(
                    f"Skipping book with unknown language ({book.language_name}): {book.title}"
                )
                continue
            if repo.find_by_title(book.title, lang.id) is not None:
                print(f"Already exists in {book.language_name}: {book.title}")
                continue
            count += 1
            all_tags = []
            if tags:
                all_tags.extend(tags)
            if "tags" in row and row["tags"]:
                for tag in row["tags"].split(","):
                    if tag and tag not in all_tags:
                        all_tags.append(tag)
            book.book_tags = all_tags
            text = row.get("text") or None
            text_file = row.get("text_file") or None
            if text and text_file:
                print(f"Skipping {book.language_name} book that has both text and text_file: {book.title}")
                continue
            if text_file:
                fn = os.path.join(os.path.dirname(file), text_file)
                with open(fn, 'rb') as f:
                    fs = FileStorage(name=fn, filename=fn, stream=f)
                    text = service.get_file_content(fs)
            book.text = text
            book.source_uri = row.get("url") or None
            if "audio" in row and row["audio"]:
                book.audio_filename = os.path.join(os.path.dirname(file), row["audio"])
            book.audio_bookmarks = row.get("bookmarks") or None
            repo.add(book)
            print(
                f"Added {book.language_name} book (tags={','.join(all_tags)}): {book.title}"
            )

    print()
    print(f"Added {count} books")
    print()

    if not commit:
        db.session.rollback()
        print("Dry run, no changes made.")
        return

    print("Committing...")
    sys.stdout.flush()
    repo.commit()
