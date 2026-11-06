"""Tests for comic page -> chapter transcript text cleanup.

The OCR recogniser drops inter-word spaces on Latin text, and a dictionary
segmenter recovers them -- but that same segmenter shreds proper nouns
("SAPIN" -> "sap in"). These tests pin down the guards that keep names intact.
"""

from comic_text import (
    clean_ocr_text,
    has_cjk,
    is_meaningful,
    looks_concatenated,
    resegment,
    words_in,
)


# -- cleaning ---------------------------------------------------------------


def test_clean_strips_leading_junk_but_keeps_dialogue_punctuation():
    assert clean_ocr_text(" 5 SCRATCH  ") == "SCRATCH"


def test_clean_drops_cjk_that_leaks_into_latin_text():
    # PP-OCR hallucinated this from an English bubble during testing.
    assert clean_ocr_text("IHAVE\u513fTOADMIT") == "IHAVETOADMIT"


def test_clean_keeps_apostrophes_and_inner_punctuation():
    assert clean_ocr_text("HE DOESN'T HIT") == "HE DOESN'T HIT"
    assert clean_ocr_text("WAIT... WHAT?") == "WAIT... WHAT?"


def test_has_cjk_detects_hallucinated_glyphs():
    assert has_cjk("IHAVE\u513fTOADMIT")
    assert not has_cjk("I HAVE TO ADMIT")


# -- deciding whether to resegment -----------------------------------------


def test_a_fused_run_anywhere_in_a_block_is_detected():
    # Real OCR: the space before the fused word survives, which is why looking
    # only at single-token blocks missed half the cases.
    assert looks_concatenated("HISHEAD ON")
    assert looks_concatenated("REY,MAKESURE")
    assert looks_concatenated("NOWTHATALICE THINKSI'VE")


def test_already_spaced_text_is_not_resegmented():
    assert not looks_concatenated("HE DOESN'T HIT")
    assert not looks_concatenated("PREPARE TO BE")


def test_a_contraction_is_not_a_fused_run():
    assert not looks_concatenated("I DON'T")
    assert not looks_concatenated("IT ISN'T")


def test_a_fused_run_ending_in_a_contraction_is_still_detected():
    # "MONTHSANDI'M" ends in "I'M" but is plainly three words.
    assert looks_concatenated("MONTHSANDI'M")


def test_short_noise_is_not_resegmented():
    # OCR noise must not be turned into plausible-looking words.
    assert not looks_concatenated("dWHHL")
    assert not looks_concatenated("HL")
    assert not looks_concatenated("KWNI")


def test_a_real_word_is_not_split_into_single_letters():
    # Wordninja offers "color is t" for this; a stray "t" means the split is
    # wrong, not that the OCR lost spaces.
    assert not looks_concatenated("Colorist")


def test_names_containing_single_letter_fragments_are_left_alone():
    # Wordninja offers "e lenoir" and "ze s tier"; both are damage.
    for name in ("ELENOIR", "Zestier", "Valden"):
        assert not looks_concatenated(name), f"{name} was split"


def test_capital_letters_inside_a_word_mean_it_is_not_concatenated():
    # Mixed case marks a name: fused uppercase dialogue has none.
    for name in ("TurtleMe", "Fuyuki", "Maybur", "TelmoreCity"):
        assert not looks_concatenated(name), f"{name} was split"


def test_a_lowercase_name_can_still_be_split_without_a_glossary():
    # Documents the real limit of a dictionary-only approach: "Ashbern" reads
    # as two English words. The per-series glossary is what fixes these, and
    # why it is carried in config rather than guessed.
    assert looks_concatenated("Ashbern")


def test_repeated_noise_is_not_resegmented():
    assert not looks_concatenated("HL HL HL")


def test_a_real_word_is_not_resegmented():
    assert not looks_concatenated("PREVIOUS")
    assert not looks_concatenated("RUMBLE")


# -- resegmentation ---------------------------------------------------------


def test_resegment_recovers_words_from_concatenated_text():
    assert resegment("HEDOESN'THIT") == "he doesn't hit"
    assert resegment("BLOCKHIMFROM") == "block him from"
    assert resegment("STILLNOTUSEDTO") == "still not used to"


def test_resegment_handles_a_fused_run_inside_a_spaced_block():
    # Only the fused run is split; "ON" is a real word and keeps its case.
    assert resegment("HISHEAD ON") == "his head ON"
    assert resegment("NOWTHATALICE THINKSI'VE FALLEN ASLEEP,") == (
        "now that alice thinks i've FALLEN ASLEEP,"
    )


def test_resegment_handles_punctuation_inside_a_fused_block():
    # Only the fused run is lowercased; the short name keeps its capitals.
    assert resegment("REY,MAKESURE") == "REY,make sure"


def test_resegment_leaves_a_word_that_needs_no_splitting_untouched():
    # HEARTBROKEN is one dictionary word, so it is not lowercased and its
    # exclamation mark stays attached.
    assert resegment("HEARTBROKEN!") == "HEARTBROKEN!"


def test_resegment_splits_a_run_that_ends_in_a_contraction():
    assert resegment("MONTHSANDI'M") == "months and i'm"


def test_resegment_leaves_already_spaced_text_alone():
    assert resegment("HE DOESN'T HIT") == "HE DOESN'T HIT"


def test_resegment_preserves_trailing_sentence_punctuation():
    assert resegment("SWEETDREAMS,") == "sweet dreams,"
    assert resegment("THATSHELF!") == "that shelf!"


def test_resegment_preserves_case_when_it_does_not_split():
    assert resegment("PREVIOUS") == "PREVIOUS"
    assert resegment("RUMBLE") == "RUMBLE"


def test_resegment_does_not_shred_known_proper_nouns():
    # Names that appear in the chapter are protected, including ones a
    # dictionary would happily bisect.
    names = ["SAPIN", "XYRUS", "DARV", "TURTLEME", "ELENOIR", "DICATHEN"]
    for name in names:
        out = resegment(name, known=names)
        assert out.upper() == name, f"{name} was mangled into {out!r}"


def test_short_blocks_are_never_split():
    # Five letters is not enough evidence of a lost space, so short names
    # survive even without a glossary.
    assert resegment("SAPIN") == "SAPIN"
    assert resegment("DARV") == "DARV"


def test_short_fused_dialogue_is_repaired_without_a_glossary():
    # Regression: the 7-letter floor meant no run shorter than "BLOCKHIMFROM"
    # was ever repaired, so these survived into transcripts as single tokens.
    # 40+ occurrences across seven real chapters. Every part here is a common
    # word, which is what distinguishes them from names.
    assert resegment("HEHAS YOUR HAIR!") == "he has YOUR HAIR!"
    assert resegment("IWAS") == "i was"
    assert resegment("TOFORM") == "to form"
    assert resegment("AREYOU") == "are you"
    assert resegment("WELL,ASMUCH as i want") == "WELL,as much as i want"


def test_a_short_proper_noun_is_still_not_split():
    # The same length window as the repaired cases above. These parts are rare
    # or unknown words, so the cost gate refuses them -- no glossary needed.
    # (Seven-letter names such as ETISTIN take the older long-run rule and still
    # split; that is a pre-existing limit the glossary covers, not a regression.)
    for name in ("SAPIN", "XYRUS", "DARV", "BURIM", "VALDEN", "ASHBER"):
        assert resegment(name).upper() == name, f"{name} was split"


def test_a_word_already_in_the_dictionary_is_never_split():
    # "MANA" is a real dictionary word, but the segmenter offers "man a".
    # Splitting an existing word can only lose information.
    for word in ("MANA", "ALICE", "VIRION", "CONJURER", "ART", "REYNOLDS"):
        assert resegment(word) == word, f"{word} was split"


def test_a_contraction_tail_is_not_invented():
    # "they re" and "what re" put a token on the page that was never there.
    assert resegment("THEYRE") == "THEYRE"
    assert resegment("WHATRE") == "WHATRE"


def test_a_suffix_is_not_shaved_off_a_stem():
    # AUGMENTER is a core term of the setting. "augment er" is damage, and it
    # was already happening under the old 7-letter rule because the run is 9.
    for term in ("AUGMENTER", "AUGMENTERS"):
        assert resegment(term).upper() == term, f"{term} was split"


def test_unprotected_long_name_splits_when_no_glossary_is_given():
    # Documents the limit honestly: without outside knowledge the segmenter has
    # only a dictionary, and "sap in" and "block him from" are the same shape.
    # This is why the glossary is carried in config rather than guessed.
    assert resegment("Ashbern") == "ash bern"


def test_resegment_accepts_a_glossary_to_protect():
    # A glossary name is never split, even when it is long enough to split.
    assert resegment("ZESTIER IS NICE", known=["ZESTIER"]) == "ZESTIER IS NICE"


def test_resegment_accepts_a_domain_word_to_protect():
    # "mana" is a story term the English dictionary splits into "man a"; the
    # glossary keeps it whole inside a fused run.
    assert resegment("TOCONTROLMANAIS", known=["MANA"]) == "to control MANA is"


def test_resegment_leaves_credit_names_alone():
    # From the real credits page.
    for line in ("Original Novel and Script: TurtleMe", "Comic Artist: Fuyuki23"):
        assert resegment(line) == line


def test_resegment_does_not_lose_characters():
    for block in ("REY,MAKESURE", "HISHEAD ON", "\u201cART\u201dISALITTLE", "dWHHL"):
        before = [c for c in block if c.isalpha()]
        after = [c for c in resegment(block) if c.isalpha()]
        assert len(before) == len(after), f"{block!r} lost characters"


# -- counting ---------------------------------------------------------------


def test_words_in_counts_split_words_not_character_runs():
    assert words_in("TO THE OUTSIDE WORLD,") == 4
    assert words_in("HE DOESN'T HIT") == 3
    assert words_in("HISHEAD ON") == 3


def test_words_in_ignores_pure_punctuation():
    assert words_in("...") == 0
    assert words_in("+") == 0


# -- deciding whether a block carries any real text ------------------------


def test_a_sentence_of_real_words_is_meaningful():
    assert is_meaningful("he doesn't hit that corner")
    assert is_meaningful("A KING.")


def test_a_glossary_name_alone_is_meaningful():
    # The map page is mostly place names, which no English dictionary knows.
    assert is_meaningful("ELENOIR", known=["ELENOIR"])
    assert is_meaningful("SAPIN", known=["SAPIN"])


def test_fused_dialogue_is_meaningful():
    assert is_meaningful("HEDOESN'THIT")


def test_detector_noise_is_not_meaningful():
    # Specular highlights and page furniture, all observed in real output.
    for junk in ("dWHHL", "CNHL", "213.", "+", "/", "7"):
        assert not is_meaningful(junk), f"{junk!r} was kept"


def test_some_noise_survives_and_that_is_accepted():
    # The word list contains abbreviations like "nih" and "lii", so this block
    # is kept. Dropping it would need a stricter rule that starts discarding
    # real text, which costs a summary more than one stray line.
    assert is_meaningful("NIHLII")


def test_a_single_stray_letter_is_not_meaningful():
    assert not is_meaningful("m")
    assert not is_meaningful("l")


def test_empty_text_is_not_meaningful():
    assert not is_meaningful("")
    assert not is_meaningful("   ")
