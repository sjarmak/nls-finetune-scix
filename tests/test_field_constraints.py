"""Enum values must be values ADS actually indexes.

The snapshot is the ADS bibgroup_facet list for q=*:* (48 values, fetched
2026-09-25). A value missing from it matches no records, so offering it to
the extractor produces empty result sets.
"""

from finetune.domains.scix.field_constraints import BIBGROUPS, PROPERTIES, PROPERTY_DOCTYPES
from finetune.domains.scix.jev_option_text import BIBGROUP_DESCRIPTIONS
from finetune.domains.scix.ner import BIBGROUP_SYNONYMS

ADS_BIBGROUPS_2026_09_25 = frozenset(
    {
        "PhysEd", "USGS", "NTRS", "NASA SLSL", "CfA", "Historical Literature",
        "NASA PubSpace", "Spitzer", "NRAO", "HST", "ESO/Telescopes", "Chandra",
        "NOIRLab", "Leiden Observatory", "Solar Dynamics Observatory", "XMM",
        "NOAO", "SOHO", "Keck", "IUE", "USNO", "ALMA", "NASA Astrobiology",
        "Gemini", "SETI", "Herschel", "GALEX", "HCOWAC", "JWST", "WHT", "STEREO",
        "JCMT", "INT", "IRTF", "Subaru", "Kepler", "Swift", "Pan-STARRS", "TESS",
        "GTC", "SMA", "FUSE", "K2", "CFHT", "Chandra/Technical", "EUVE",
        "Chandra/CSC", "Copernicus",
    }
)  # fmt: skip

# property_facet values for q=*:* on the same date. ADS also accepts each
# with underscores (property:pub_openaccess), the form PROPERTIES uses.
ADS_PROPERTIES_2026_09_25 = frozenset(
    {
        "esource", "article", "refereed", "notrefereed", "openaccess",
        "pubopenaccess", "eprintopenaccess", "nonarticle", "pmcopenaccess", "toc",
        "associated", "adsopenaccess", "data", "inspire", "authoropenaccess",
        "private", "presentation", "release", "ocrabstract", "librarycatalog",
    }
)  # fmt: skip


def test_every_bibgroup_is_an_ads_bibgroup():
    assert BIBGROUPS <= ADS_BIBGROUPS_2026_09_25


def test_every_synonym_names_an_offered_bibgroup():
    assert set(BIBGROUP_SYNONYMS.values()) <= BIBGROUPS


def test_every_bibgroup_has_a_description():
    assert set(BIBGROUP_DESCRIPTIONS) == {"none", *BIBGROUPS}


def test_properties_ads_lacks_are_exactly_the_record_kinds():
    lacking = {
        value for value in PROPERTIES if value.replace("_", "") not in ADS_PROPERTIES_2026_09_25
    }
    assert lacking == PROPERTY_DOCTYPES
