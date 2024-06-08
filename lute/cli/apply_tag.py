"""
Apply tags to existing terms.
"""

import sys

from lute.db import db
from lute.term.model import Repository


def apply_min_status(language_id, words_file, status, commit, verbose):
    updated = 0
    unchanged = 0
    not_found = 0
    skipped = 0

    repo = Repository(db)
    with open(words_file) as f:
        for line in f:
            word = line.rstrip()
            term = repo.find(language_id, word)
            if not term:
                not_found += 1
                continue
            if 'generated' in term.term_tags:
                skipped += 1
                continue
            if term.status >= status:
                unchanged += 1
                continue
            updated += 1
            if verbose:
                print("Updating status from {} => {} for {}".format(term.status, status, word))
            term.status = status
            repo.add(term)

    print("Updated  : {}".format(updated))
    print("Unchanged: {}".format(unchanged))
    print("Not found: {}".format(not_found))
    print("Skipped  : {}".format(skipped))
    print()

    if not commit:
        print("Dry run, no changes made.")
        return

    print("Committing...")
    sys.stdout.flush()
    repo.commit()
    


def apply_tag(language_id, words_file, tag, commit, verbose, remove):
    newly_tagged = 0
    already_tagged = 0
    not_found = 0
    skipped = 0

    repo = Repository(db)
    with open(words_file) as f:
        for line in f:
            word = line.rstrip()
            term = repo.find(language_id, word)
            if not term:
                not_found += 1
                continue
            if 'generated' in term.term_tags:
                skipped += 1
                continue
            if (tag in term.term_tags) != remove:
                already_tagged += 1
                continue
            newly_tagged += 1
            if remove:
                if verbose:
                    print("Removing tag from: {}".format(word))
                term.term_tags.remove(tag)
            else:
                if verbose:
                    print("Tagging: {}".format(word))
                term.term_tags.append(tag)
            repo.add(term)

    print("Newly tagged   : {}".format(newly_tagged))
    print("Already tagged : {}".format(already_tagged))
    print("Words not found: {}".format(not_found))
    print("Skipped        : {}".format(skipped))
    print()

    if not commit:
        print("Dry run, no changes made.")
        return

    print("Committing...")
    sys.stdout.flush()
    repo.commit()
