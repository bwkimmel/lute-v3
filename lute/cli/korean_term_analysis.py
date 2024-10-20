"""
Add child terms for words found in Korean language books.

This script goes through *all* the books for the Korean language, parses it
using mecab-ko. For each word, it determines the root word. If the root word is
defined, and the structure of the word is found in a user-supplied list of
grammatical patterns, a child term is automatically created.

Root words that are not found are written to a CSV along with some basic stats
for each such word.
"""

import sys
import csv
from collections import Counter
from dataclasses import dataclass
from glob import glob
from importlib import resources
from lute.db import db
from lute.models.book import Book
from lute.term.model import Repository
from mecab import MeCab, Feature
from tqdm import tqdm
from . import static


_TRACE_WORDS=set()


def _trace(word, fmt, *args):
    if word not in _TRACE_WORDS:
        return
    print("TRACE[{}]: {}".format(word, fmt.format(*args)))


def normalize(features):
    if len(features) < 1:
        return features
    if any([f.reading is None for f in features]):
        return features

    noun_prefix_len = 0
    for f in features:
        if '+' in f.pos:
            break
        if not f.pos.startswith('NN') and not (f.pos == 'XSN' and not f.reading == '들'):
            break
        noun_prefix_len += 1
    if noun_prefix_len > 0:
        pos = 'NNG'
        if noun_prefix_len == 1:
            pos = features[0].pos
        noun_prefix = "".join([f.reading for f in features[:noun_prefix_len]])
        end_prefix = features[noun_prefix_len - 1]
        prefix_feature = Feature(
            pos=           pos,
            reading=       noun_prefix,
            has_jongseong= end_prefix.has_jongseong,
            start_pos=     features[0].start_pos,
            end_pos=       end_prefix.end_pos,
        )
        features = [prefix_feature] + features[noun_prefix_len:]

    nr_prefix_len = 0
    for f in features:
        if f.pos != 'NR':
            break
        nr_prefix_len += 1
    if nr_prefix_len > 1:
        nr_prefix = "".join([f.reading for f in features[:nr_prefix_len]])
        last = features[nr_prefix_len - 1]
        nr_feature = Feature(
            pos=          'NR',
            reading=       nr_prefix,
            has_jongseong= last.has_jongseong,
            start_pos=     features[0].start_pos,
            end_pos=       last.end_pos,
        )
        features = [nr_feature] + features[nr_prefix_len:]

    stem_pos = 'VV'
    stem_end = None
    for i, f in enumerate(features):
        start_pos = f.start_pos
        if start_pos is None:
            start_pos = f.pos
        if start_pos in ['VV', 'XSV']:
            stem_pos = 'VV'
            stem_end = i
        elif start_pos in ['VA', 'XSA']:
            stem_pos = 'VA'
            stem_end = i
        elif start_pos in ['VX']:  # should probably not be combined
            stem_end = i

    if stem_end is not None:
        stem_features = []
        if stem_end > 0:
            stem_features = features[:stem_end]
        pos = stem_pos
        if '+' in features[stem_end].pos:
            poses = features[stem_end].pos.split('+')
            pos = '+'.join([stem_pos] + poses[1:])
        stem = ''.join([f.reading for f in stem_features])
        reading = stem + features[stem_end].reading
        expr = None
        if features[stem_end].expression is not None:
            elems = features[stem_end].expression.split('+')
            stem += elems[0].split('/')[0]
            expr = '+'.join(["{}/{}/*".format(stem, stem_pos)] + elems[1:])
        else:
            stem = reading
        f = Feature(
            pos=           pos,
            reading=       reading,
            has_jongseong= features[stem_end].has_jongseong,
            type=          features[stem_end].type,
            start_pos=     features[0].start_pos,
            end_pos=       features[stem_end].end_pos,
            expression=    expr,
        )
        features = [f] + features[stem_end+1:]

    return features


def parse_elem(elem):
    parts = elem.split('/', 3)
    if len(parts) != 3:
        raise ValueError('invalid element: {}'.format(elem))
    # if parts[2] != '*':
        # print("### {}".format(parts[2]))
    return (parts[0], parts[1], parts[2])


def split_expr(expr):
    return [parse_elem(e) for e in expr.split('+')]


POS_TAGS = {
    'NNG':  ['noun'],
    'NNP':  ['noun', 'proper noun'],
    'NNB':  ['noun', 'bound noun'],
    'NNBC': ['noun', 'counter word'],
    'NR':   ['number'],
    'NP':   ['pronoun'],
    'VV':   ['verb'],
    'VA':   ['adj'],
    'VX':   [],
    'VCP':  [],
    'VCN':  [],
    'MM':   ['det'],
    'MAG':  ['adv'],
    'MAJ':  ['conjunction'],
    'IC':   ['interjection'],
    'JKS':  [],
    'JKC':  [],
    'JKG':  [],
    'JKO':  [],
    'JKB':  [],
    # 'JKV':  [],
    # 'JKQ': [],
    'JX':   [],
    'JC':   [],
    # 'EP':   [],
    'EF':   [],
    'EC':   [],
    'ETN':  ['noun'],
    'ETM':  ['prenoun'],
    # 'XPN':  [],
    # 'XSN':  [],
    'XSV':  ['verb'],
    'XSA':  ['adj'],
    # 'XR':   [],
    # 'SF':   [],
    # 'SE':   [],
    # 'SS':   [],
    # 'SP':   [],
    # 'SO':   [],
    # 'SW':   [],
    # 'SL':   [],
    # 'SH':   [],
    # 'SN':   [],
}

# HONORIFIC_ELEMS = (
#         ('으시', 'EP', '*'),
#         ('시', 'EP', '*'),
# )


LETTER_SUBS = [
    ('\u1107', 'ㅂ'),
    ('\u11b8', 'ㅂ'),
    ('\u1105', 'ㄹ'),
    ('\u11af', 'ㄹ'),
    ('\u1102', 'ㄴ'),
    ('\u11ab', 'ㄴ'),
    ('\u1106', 'ㅁ'),
    ('\u11b7', 'ㅁ'),
]


@dataclass
class WordStats:
    text:     str
    pos:      list
    refs:     int
    children: list
    term_id:  int


@dataclass
class PatternStats:
    pattern: str
    refs:    int
    roots:   str


@dataclass
class Pattern:
    signature:   str
    translation: str
    tags:        list
    parents:     list
    output:      list
    line:        int
    used:        bool


def reduce_pattern_once(elements, patterns, parents=None, tags=None, word=None):
    for n in reversed(range(1, len(elements))):
        for i in range(0, len(elements) - n + 1):
            sub_pattern = '+'.join(['/'.join(x) for x in elements[i:i+n]])
            if sub_pattern not in patterns:
                continue
            p = patterns[sub_pattern]
            if p.translation and not p.output:
                continue
            old_pattern = '+'.join(['/'.join(x) for x in elements])
            elements[i:i+n] = p.output
            new_pattern = '+'.join(['/'.join(x) for x in elements])
            if word is not None:
                _trace(sub_pattern, "term {} has subpattern: {} => {}", word, old_pattern, new_pattern)
                _trace(word, "applying production rule for {} => {}", sub_pattern, new_pattern)
            if parents is not None:
                for parent in p.parents:
                    if parent not in parents:
                        if word:
                            _trace(word, "adding parent {}", parent)
                            _trace(parent, "added as parent to word {}", word)
                        parents.append(parent)
            if tags is not None:
                for tag in p.tags:
                    if tag not in tags:
                        _trace(word, "adding tag {}", tag)
                        tags.append(tag)
            p.used = True
            return True
    return False


def reduce_pattern(elements, patterns, parents=None, tags=None, word=None):
    reduced = False
    while reduce_pattern_once(elements, patterns, parents, tags, word):
        reduced = True
    return reduced


def analyze_term(term, features, termrepo, language_id, patterns, undefined_roots, undefined_patterns):
    # print("Analyzing: {}".format(term.text))
    _trace(term.text, "before normalization")
    for f in features:
        _trace(term.text, "  > original feature: {}", f)
    features = normalize(features)
    elements = []
    _trace(term.text, "after normalization")
    for f in features:
        _trace(term.text, "  > feature: {}", f)
        if f.reading is None:
            #print('Invalid feature in word {}: {}'.format(term.text, f))
            _trace(term.text, "invalid feature in word: {}", f)
            return
        if f.expression is not None:
            elements.extend(split_expr(f.expression))
        else:
            elements.append((f.reading, f.pos, '*'))
    if len(elements) == 0:
        _trace(term.text, "word has no elements")
        return
    word_from_features = "".join([f.reading for f in features])
    if term.text != word_from_features:
        _trace(term.text, "features do not match word: {} != {}", term.text, word_from_features)
        return

    root = elements[0]
    root_word = root[0]
    if root[1].startswith('V'):
        _trace(term.text, "root word is a verb or adjective")
        root_word += '다'
    _trace(term.text, "root word: {}", root_word)
    _trace(root_word, "is the root word for {}", term.text)
    root_term = termrepo.find(language_id, root_word)
    if root_term is None or root_term.id is None or (root_term.translation is None and root_term.status != 98):
        # print("undefined root: {}/{}".format(root_word, root[1]))
        _trace(term.text, "root word is undefined")
        stats = undefined_roots.get(root_word)
        term_id = None
        if root_term is not None:
            term_id = root_term.id
        if stats is None:
            stats = WordStats(
                text     = root_word,
                pos      = [],
                refs     = 0,
                children = [],
                term_id  = term_id,
            )
        if root[1] not in stats.pos:
            stats.pos.append(root[1])
            stats.pos.sort()
        stats.refs += 1
        if len(elements) > 1 and term.text not in stats.children:
            stats.children.append(term.text)
            stats.children.sort()
        undefined_roots[root_word] = stats
        return

    pos = elements[-1][1]
    pos_tags = POS_TAGS.get(pos)
    tags = []
    if pos_tags:
        tags.extend(pos_tags)

    if elements[0][1].startswith('NN'):
        _trace(term.text, "word begins with a noun")
        elements[0] = (elements[0][0], 'NN', '*')

    # old_elements = elements
    # elements = []
    # for e in old_elements:
    #     if e in HONORIFIC_ELEMS:
    #         _trace(term.text, "dropping honorific element {} from pattern", e)
    #         tags.append('honorific')
    #         continue
    #     elements.append(e)

    if len(elements) < 2:
        _trace(term.text, "word has fewer than two elements")
        return

    elements[0] = ('*', elements[0][1], '*')
    for i, e in enumerate(elements):
        text = e[0]
        old_text = text
        for (x, y) in LETTER_SUBS:
            text = text.replace(x, y)
        elements[i] = (text, e[1], e[2])

    pattern = '+'.join(['/'.join(x) for x in elements])
    _trace(pattern, "term {} has this pattern", term.text)
    _trace(term.text, "has pattern {}", pattern)
    translation = None
    parents = [root_word]
    while True:
        p = patterns.get(pattern)
        if p and p.translation:
            p = patterns[pattern]
            p.used = True
            for parent in p.parents:
                if parent not in parents:
                    parent_term = termrepo.find(language_id, parent)
                    if parent_term is None or parent_term.id is None:
                        _trace(term.text, "parent term {} is undefined", parent)
                        return
                    parents.append(parent)
            translation = p.translation
            _trace(term.text, "existing translation: '{}'", term.translation)
            if term.translation is not None:
                _trace(term.text, "has existing translation; checking for conflicts...")
                if term.translation != translation:
                    _trace(term.text, "conflicting translations: '{}' vs. '{}'", term.translation, translation)
                    if 'conflict' not in term.term_tags:
                        term.term_tags.append('conflict')
                    return
                _trace(term.text, "original parents: {}, new parents: {}", term.parents, parents)
                if term.parents != parents:
                    _trace(term.text, "conflicting parents: '{}' vs. '{}'", term.parents, parents)
                    if 'conflict' not in term.term_tags:
                        term.term_tags.append('conflict')
                    return
            _trace(term.text, "updating term: status={}, parents={}, translation={}, tags={}", root_term.status, parents, translation, set(tags + p.tags))
            term.status = root_term.status
            term.parents = parents
            term.translation = translation
            for tag in tags + p.tags:
                if tag not in term.term_tags:
                    _trace(term.text, "adding tag: {}", tag)
                    term.term_tags.append(tag)
            break

        else:
            if not reduce_pattern_once(elements, patterns, parents, tags, term.text):
                break
            pattern = '+'.join(['/'.join(x) for x in elements])

    if not translation:
        _trace(term.text, "cannot translate word because there is no translation for pattern: {}", pattern)
        _trace(pattern, "cannot translate {} because pattern is not defined", term.text)
        stats = undefined_patterns.get(pattern)
        if stats is None:
            stats = PatternStats(pattern=pattern, refs=0, roots=[])
        stats.refs += 1
        if root_word not in stats.roots:
            stats.roots.append(root_word)
            stats.roots.sort()
        undefined_patterns[pattern] = stats


def analyze_word(word, is_bad, features, collector, termrepo, language_id, patterns, undefined_roots, undefined_patterns):
    if is_bad:
        _trace(word, "word has bad feature: {}", features)
        return
    _trace(word, "looking up word")
    term = termrepo.find_or_new(language_id, word)
    # print("  > Term: {}".format(term))
    if 'fixme' in term.term_tags and term.parents and 'generated' not in term.term_tags:
        _trace(word, "resetting word with fixme tag")
        collector[term.text] = term
        for p in term.parents:
            parent = termrepo.find(language_id, p)
            if parent and parent.translation and 'fixme' in parent.term_tags and 'generated' not in parent.term_tags:
                collector[parent.text] = parent
        term.status = 0
        term.term_tags = []
        term.translation = None
        term.parents = []

    if term.id is not None and 'generated' in term.term_tags:
        _trace(word, "generated word already exists, resetting: status={}, parents={}, tags={}, translation={}", term.status, term.parents, term.term_tags, term.translation)
        term.status = 0 
        term.translation = None
        term.term_tags = []
        term.parents = []

    if term.id is None or (term.translation is None and term.status != 98) or 'generated' in term.term_tags:
        _trace(word, "analyzing word")
        analyze_term(term, features, termrepo, language_id, patterns, undefined_roots, undefined_patterns)
        _trace(word, "translated word as '{}'", term.translation)
        if term.translation is not None:
            term.sync_status = True
            if 'generated' not in term.term_tags:
                term.term_tags.append('generated')
            collector[term.text] = term


def analyze_book(book, collector, termrepo, language_id, mecab, patterns, undefined_roots, undefined_patterns):  # pylint: disable=too-many-locals
    """
    Process a single book.
    """

    # print("Analysing {}".format(book.title))

    fulltext = "\n".join([t.text for t in book.texts])
    analyze_text(fulltext, collector, termrepo, language_id, mecab, patterns, undefined_roots, undefined_patterns)  # pylint: disable=too-many-locals


def analyze_book_file(path, collector, termrepo, language_id, mecab, patterns, undefined_roots, undefined_patterns):  # pylint: disable=too-many-locals
    text = ""
    title = None
    with open(path, 'r') as f:
        for line in f.readlines():
            if text == "" and line.startswith('#'):
                (k, v) = line.strip().split(':', 1)
                k = k[1:].strip().lower()
                v = v.strip()
                if k == "title":
                    title = v
                continue
            text += line
    # if title:
    #     print("Analyzing {}: {}".format(path, title))
    # else:
    #     print("Analyzing {}".format(path))
    analyze_text(text, collector, termrepo, language_id, mecab, patterns, undefined_roots, undefined_patterns)  # pylint: disable=too-many-locals


def analyze_text(text, collector, termrepo, language_id, mecab, patterns, undefined_roots, undefined_patterns):  # pylint: disable=too-many-locals
    morphemes = mecab.parse(text)

    pos = 0
    cur_word = ""
    cur_word_features = []
    bad_word = False
    for morpheme in morphemes:
        if morpheme.feature.pos[0] == 'S': # punctuation, letters, numbers
            continue

        if morpheme.feature.pos == 'UNKNOWN':
            # print("Bad feature: {}".format(morpheme))
            bad_word = True

        if morpheme.span.start > pos: # new word
            analyze_word(cur_word, bad_word, cur_word_features, collector, termrepo, language_id, patterns, undefined_roots, undefined_patterns)
            cur_word = ""
            cur_word_features = []
            bad_word = False

        cur_word += morpheme.surface
        cur_word_features.append(morpheme.feature)

        pos = morpheme.span.end

    if cur_word != "":
        analyze_word(cur_word, bad_word, cur_word_features, collector, termrepo, language_id, patterns, undefined_roots, undefined_patterns)


def run(include_books, exclude_books, book_globs, commit, trace):
    global _TRACE_WORDS
    if trace:
        _TRACE_WORDS=set([word.strip() for word in trace.split(",")])
        for word in sorted(_TRACE_WORDS):
            _trace(word, "tracing")
    language_name = 'Korean'
    books = db.session.query(Book).all()
    langid = books[0].language.id
    books = [
        b
        for b in books
        if b.language.name == language_name
            and not b.archived
            and b.id not in exclude_books
            and ((not include_books and not book_globs) or b.id in include_books)
    ]
    if len(books) == 0 and len(book_globs) == 0:
        print(f"No books for given language {language_name}, quitting.")
        sys.exit(0)

    patterns = {}
    pattern_file = resources.files(static) / 'korean_term_patterns.csv'
    with pattern_file.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            line_no = r.line_num
            signature = row['pattern']
            translation = row.get('translation') or None
            tags = []
            if 'tags' in row and row['tags']:
                tags = [s.strip() for s in row['tags'].split(',')]
            parents = []
            if 'parents' in row and row['parents']:
                parents = [s.strip() for s in row['parents'].split(',')]
            output = []
            if 'output' in row and row['output']:
                output = split_expr(row['output'])
            for (x, _) in LETTER_SUBS:
                if x in signature:
                    print(f"WARNING: Line {line_no}: invalid character '{x}' in rule for pattern {signature}")
                s = row.get('output') or ''
                if x in s:
                    print(f"WARNING: Line {line_no}: invalid character '{x}' in output '{s}' in rule for pattern {signature}")
            if signature in patterns:
                print(f"WARNING: Line {line_no}: redefining pattern on line {patterns[signature].line}: {signature}")
                continue
            _trace(signature, "defining pattern: translation={}, tags={}, parents={}", translation, tags, parents)
            patterns[signature] = Pattern(
                signature=signature,
                translation=translation,
                tags=tags,
                parents=parents,
                output=output,
                line=line_no,
                used=False,
            )

    for sig, pat in patterns.items():
        e = split_expr(sig)
        if reduce_pattern(e, patterns):
            new_sig = '+'.join(['/'.join(x) for x in e])
            new_pat = patterns.get(new_sig)
            print(f"WARNING: Line {pat.line}: pattern {sig} reduces to {new_sig}")
            if new_pat:
                print(f"  >> Reduced pattern defined on line {new_pat.line}")
                if new_pat.translation == pat.translation:
                    print(f"  >> Reduced pattern has same translation")
                else:
                    print(f"  >> Reduced pattern has different translation:")
                    print(f"    >> Original translation: {pat.translation}")
                    print(f"    >> Reduced translation : {new_pat.translation}")
            else:
                print("  >> Reduced pattern not defined")

    mecab = MeCab()
    repo = Repository(db)
    terms = {}
    undefined_roots = {}
    undefined_patterns = {}
    for b in tqdm(books, desc="processing books"):
        analyze_book(b, terms, repo, langid, mecab, patterns, undefined_roots, undefined_patterns)

    for spec in book_globs:
        print("Looking for files matching '{}'".format(spec))
        for path in glob(spec):
            analyze_book_file(path, terms, repo, langid, mecab, patterns, undefined_roots, undefined_patterns)

    num_conflicts = 0
    num_added = 0
    num_updated = 0
    num_invalid_parent_count = 0
    num_untranslatable = 0
    num_deleted = 0
    num_patched_parents = 0
    for term in tqdm(terms.values(), desc="processing terms"):
        if 'fixme' in term.term_tags:
            if not term.translation:
                _trace(term.text, "deleting term with fixme tag and no translation")
                repo.delete(term)
                num_deleted += 1
                continue
            _trace(term.text, "removing fixme tag")
            term.term_tags.remove('fixme')
            if 'generated' not in term.term_tags:
                _trace(term.text, "parent term with fixme tag updated")
                num_patched_parents += 1
                repo.add(term)
                continue
        if 'conflict' in term.term_tags:
            print("Conflicting definitions for: {}".format(term.text))
            _trace(term.text, "conflicting definitions")
            num_conflicts += 1
            continue
        if len(term.parents) == 0:
            _trace(term.text, "no parents")
            num_invalid_parent_count += 1
            continue
        if not term.translation:
            _trace(term.text, "could not translate")
            num_untranslatable += 1
            continue
        op = ""
        if term.id:
            op = "updated"
            num_updated += 1
        else:
            op = "added"
            num_added += 1
        _trace(term.text, "{} term with parents={}, translation={}, tags={}", op, term.parents, term.translation, term.term_tags)
        repo.add(term)

    with open("korean_unknown_terms.csv", 'w', newline='') as f:
        fieldnames = ['word', 'term_id', 'pos', 'refs', 'num_children', 'children']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for stats in undefined_roots.values():
            w.writerow({
                'term_id':      stats.term_id,
                'word':         stats.text,
                'pos':          ", ".join(stats.pos),
                'refs':         stats.refs,
                'num_children': len(stats.children),
                'children':     ", ".join(stats.children),
            })

    with open("korean_undefined_patterns.csv", 'w', newline='') as f:
        fieldnames = ['pattern', 'refs', 'num_roots', 'roots']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for stats in undefined_patterns.values():
            w.writerow({
                'pattern':   stats.pattern,
                'refs':      stats.refs,
                'num_roots': len(stats.roots),
                'roots':     ", ".join(stats.roots),
            })

    if commit:
        print("Committing...")
        repo.commit()
    else:
        print("Dry-run mode, no changes made.")

    print("Done.")
    print("Stats: conflicts={}, invalid={}, untranslatable={}, unknown roots={}, added={}, updated={}, deleted={}, removed_fixme_tag={}".format(
        num_conflicts, num_invalid_parent_count, num_untranslatable, len(undefined_roots), num_added, num_updated, num_deleted, num_patched_parents,
    ))

    # for sig, pat in patterns.items():
    #     if pat.used:
    #         continue
    #     print(f"UNUSED PATTERN: Line {pat.line}: {sig}")
