from unittest import TestCase
import os
import orcasong
import orcasong.from_toml as from_toml

EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(orcasong.__file__)), "examples"
)


def _test_extr(infile):
    return infile + "_extr"


orcasong.from_toml.EXTRACTORS["nu_chain_neutrino"] = _test_extr


class TestSetupProcessorExampleConfig(TestCase):
    def setUp(self):
        self.processor = from_toml.setup_processor(
            infile="test_in",
            toml_file=os.path.join(EXAMPLES, "orcasong_example.toml"),
            detx_file="test_det",
        )

    def test_time_window(self):
        self.assertEqual(self.processor.time_window, [-100, 5000])

    def test_max_n_hits(self):
        self.assertEqual(self.processor.max_n_hits, None)

    def test_chunksize(self):
        self.assertEqual(self.processor.chunksize, 16)

    def test_extractor_is_dummy_extractor(self):
        self.assertEqual(self.processor.extractor, "test_in_extr")
