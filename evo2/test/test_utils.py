from unittest import TestCase
from unittest.mock import Mock, patch

from evo2.utils import make_phylotag_from_gbif


class MakePhyloTagFromGbifTest(TestCase):
    @patch("requests.get")
    def test_non_success_response_returns_none(self, get):
        get.return_value = Mock(status_code=503)

        self.assertIsNone(make_phylotag_from_gbif("Synthetic taxon"))

    @patch("requests.get")
    def test_missing_taxonomy_returns_none(self, get):
        get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "kingdom": None,
                "phylum": None,
                "class": None,
                "order": None,
                "family": None,
                "genus": None,
                "species": None,
            },
        )

        self.assertIsNone(make_phylotag_from_gbif("Synthetic taxon"))

    @patch("requests.get")
    def test_resolved_taxonomy_returns_tag(self, get):
        get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "kingdom": "Synthetic kingdom",
                "phylum": "Synthetic phylum",
                "class": "Synthetic class",
                "order": "Synthetic order",
                "family": "Synthetic family",
                "genus": "Synthetic genus",
                "species": "Synthetic species",
            },
        )

        self.assertEqual(
            make_phylotag_from_gbif("Synthetic species"),
            "|D__SYNTHETIC KINGDOM;P__SYNTHETIC PHYLUM;C__SYNTHETIC CLASS;"
            "O__SYNTHETIC ORDER;F__SYNTHETIC FAMILY;G__SYNTHETIC GENUS;"
            "S__SYNTHETIC SPECIES|",
        )
