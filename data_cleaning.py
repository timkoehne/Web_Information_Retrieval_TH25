import re
import pkg_resources
import unidecode
import html
from symspellpy import SymSpell, Verbosity

max_edit_distance_dictionary = 2
prefix_length = 3
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def clean_document(text: str) -> str:
    # text = text[int(len(text) * 0.3) :]
    return text

def remove_urls(text: str) -> str:
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    return text


def replace_unicode_encoding(text: str) -> str:
    return unidecode.unidecode(text)


def replace_html_encoding(text: str) -> str:
    return html.unescape(text)


def replace_spelling_mistakes(text: str) -> str:
    words = text.split()
    corrected_text = " ".join(
        [
            sym_spell.lookup(
                word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
            )[0].term
            for word in words
        ]
    )
    return corrected_text
