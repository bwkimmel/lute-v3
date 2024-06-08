import csv
import hashlib
import os

from flask import current_app
from lute.db import db
from lute.models.term import Status, Term
from genanki import Deck, Model, Note, Package, guid_for
from textwrap import dedent
from typing import Optional

HIDDEN_TAGS = ["loan", "anki", "generated", "fixme", "topik1", "topik2", "topik1v", "topik2v"]
EXCLUDE_TAGS = ["fixme", "generated", "interjection", "onomatopoeia", "transliteration", "???", "sic", "particle", "loan"]
INCLUDE_TAGS = ["topik1", "topik2", "topik1v", "topik2v", "anki"]
DEFAULT_MIN_STATUS = 4
MIN_STATUS_OVERRIDE = {
    "topik1":  1,
    "topik1v": 1,
    "topik2":  3,
    "topik2v": 3,
}
PREEXISTING_GUIDS = {}

ANKI_MODEL = Model(
    0x515AEEF3,
    "Lute Term (Korean)",
    fields = [
        {"name": "Term"},
        {"name": "Language"},
        {"name": "Pronunciation"},
        {"name": "Translation"},
        {"name": "Image"},
    ],
    css = dedent("""\
        .card {
            font-family: arial;
            font-size: 20px;
            text-align: center;
            color: black;
            background-color: white;
        }
        h1 {
            font-size: small;
            padding-bottom: 1em;
        }
        """),
    templates = [{
        "name": "Recognize",
        "qfmt": dedent("""\
            <h1>What does this {{Language}} word mean?</h1>
            {{Term}}
            {{tts ko_KR voices=Microsoft_InJoon:Term}}
            <br>
            {{#Pronunciation}}
                <br>
                Pronunciation: {{Pronunciation}}
            {{/Pronunciation}}
            """),
        "afmt": dedent("""\
            {{FrontSide}}
            
            <hr id="answer">
            
            {{Translation}}
            <br>
            {{#Tags}}
                <br>
                Tags: {{Tags}}
            {{/Tags}}
            {{#Image}}
                <br><br>
                {{Image}}
            {{/Image}}
            """),
    # }, {
    #     "name": "Recall",
    #     "qfmt": dedent("""\
    #         <h1>How do you say this in {{Language}}?</h1>
    #         {{Translation}}
    #         <br>
    #         {{#Tags}}
    #             <br>
    #             Tags: {{Tags}}
    #         {{/Tags}}
    #         {{#Image}}
    #             <br><br>
    #             {{Image}}
    #         {{/Image}}
    #         """),
    #     "afmt": dedent("""\
    #         {{FrontSide}}
    #
    #         <hr id="answer">
    #
    #         {{Term}}
    #         {{tts ko_KR voices=Microsoft_InJoon:Term}}
    #         <br>
    #         {{#Pronunciation}}
    #             <br>
    #             Pronunciation: {{Pronunciation}}
    #         {{/Pronunciation}}
    #         """),
    }],
)


class LuteNote(Note):
    @property
    def guid(self):
        if self.fields[0] in PREEXISTING_GUIDS:
            return PREEXISTING_GUIDS[self.fields[0]]
        return guid_for(self.fields[0])


def preprocess_translation(tr: str) -> str:
    lines = [line
             for line in tr.splitlines()
             if "[noanki]" not in line and "[loan]" not in line]
    return "<br>".join(lines)


def sanitize_tag(tag: str) -> str:
    return tag.replace(" ", "_")


def term_to_note(term: Term, media_files: [str]) -> Optional[LuteNote]:
    tags = [tag.text for tag in term.term_tags]
    if "noanki" in tags:
        return None
    if "anki" not in tags:
        if term.parents and 'root' not in term.parents:
            return None
    if len(set(tags) & set(INCLUDE_TAGS)) == 0:
        if len(set(tags) & set(EXCLUDE_TAGS)) > 0:
            return None
        if term.status == Status.IGNORED:
            return None
    min_status = DEFAULT_MIN_STATUS
    for tag in tags:
        if tag in MIN_STATUS_OVERRIDE:
            override = MIN_STATUS_OVERRIDE[tag]
            if override < min_status:
                min_status = override
    if term.status < min_status:
        return None
    if not term.translation:
        return None

    translation = preprocess_translation(term.translation)
    if not translation:
        return None

    image = term.get_current_image()
    if image:
        if image.startswith("/"):
            image = image[1:]
        image_path = os.path.join(current_app.config["DATAPATH"], image + ".jpeg")
        image_file = os.path.basename(image_path)
        media_files.append(image_path)

    return LuteNote(
        model = ANKI_MODEL,
        fields = [
            term.text.replace("\u200b", ""),
            "Korean",
            term.romanization or "",
            translation,
            """<img src="{image}">""".format(image=image_file) if image else "",
        ],
        tags = [sanitize_tag(tag) for tag in tags if tag not in HIDDEN_TAGS],
    )


def extract_prexisting_guids(guids_file):
    if not guids_file:
        return

    with open(guids_file) as f:
        reading_headers = True
        guid_col = None
        front_col = None
        special_cols = []
        r = csv.reader(f, delimiter="\t", quotechar='"')
        for row in r:
            if reading_headers and len(row) == 1 and row[0].startswith("#"):
                if row[0].startswith("#guid column:"):
                    guid_col = int(row[0].split(":", 1)[1]) - 1
                    special_cols.append(guid_col)
                elif row[0].startswith("#deck column:"):
                    special_cols.append(int(row[0].split(":", 1)[1]) - 1)
                elif row[0].startswith("#notetype column:"):
                    special_cols.append(int(row[0].split(":", 1)[1]) - 1)
                elif row[0].startswith("#tags column:"):
                    special_cols.append(int(row[0].split(":", 1)[1]) - 1)
                continue
            if reading_headers:
                reading_headers = False
                front_col = 0
                while front_col in special_cols:
                    front_col += 1
                if guid_col is None:
                    raise Error("no guid column in Anki export")
            if len(row) < guid_col or len(row) < front_col:
                # warn: invalid row
                continue
            if row[front_col] in PREEXISTING_GUIDS:
                # warn: duplicate
                continue
            PREEXISTING_GUIDS[row[front_col]] = row[guid_col]


def generate_anki_package(language, output_path, guids_file):
    extract_prexisting_guids(guids_file)

    deck = Deck(0x532D74B9, "Korean")
    media_files = []
    count = 0
    terms = db.session.query(Term).all()
    for term in terms:
        note = term_to_note(term, media_files)
        if note:
            deck.add_note(note)
            count += 1

    print("Notes: {count}".format(count=count))
    pkg = Package(deck)
    pkg.media_files = media_files
    pkg.write_to_file(output_path)
