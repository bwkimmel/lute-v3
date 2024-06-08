"""
Import books.
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
        if repo.find_by_title(title) is not None:
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


def import_books_from_csv(language, file, tags, commit):
    repo = Repository(db)
    count = 0
    with open(file, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            title = row['title']
            if repo.find_by_title(title) is not None:
                print("Already exists: {}".format(title))
                continue
            count += 1
            all_tags = []
            if tags:
                all_tags.extend(tags)
            if 'tags' in row:
                for tag in row['tags'].split(','):
                    if tag and tag not in all_tags:
                        all_tags.append(tag)
            book = Book()
            book.language_name = language
            book.title = row['title']
            book.source_uri = row['url']
            book.text = row['text']
            book.book_tags = all_tags
            if 'audio' in row and row['audio']:
                book.audio_filename = os.path.join(os.path.dirname(file), row['audio'])
            if 'bookmarks' in row and row['bookmarks']:
                book.audio_bookmarks = row['bookmarks']
            repo.add(book)
            print("Added book (tags={}): {}".format(','.join(all_tags), book.title))

    print()
    print("Added {} books".format(count))
    print()

    if not commit:
        print("Dry run, no changes made.")
        return

    print("Committing...")
    sys.stdout.flush()
    repo.commit()
