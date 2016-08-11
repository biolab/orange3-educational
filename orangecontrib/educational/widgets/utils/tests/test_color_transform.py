import unittest
from orangecontrib.educational.widgets.utils.color_transform import (
    rgb_hash_brighter,
    hex_to_rgb, rgb_to_hex)


class TestColorTransform(unittest.TestCase):

    def setUp(self):
        pass

    def test_rgb_to_hex(self):
        # provided by http://en.labelpartners.com/pantone_coated_table.html
        matches = [[(254, 221, 0), "#FEDD00"],
                   [(0, 20, 137), "#001489"],
                   [(175, 152, 0), "#AF9800"],
                   [(255, 127, 50), "#FF7F32"]]

        for rgb, hex_code in matches:
            self.assertEqual(rgb_to_hex(rgb).lower(), hex_code.lower())

    def test_hex_to_rgb(self):
        # provided by http://en.labelpartners.com/pantone_coated_table.html
        matches = [[(254, 221, 0), "#FEDD00"],
                   [(0, 20, 137), "#001489"],
                   [(175, 152, 0), "#AF9800"],
                   [(255, 127, 50), "#FF7F32"]]

        for rgb, hex_code in matches:
            self.assertSequenceEqual(
                hex_to_rgb(hex_code), list(map(lambda x: x / 255, rgb)))

    def test_rgb_hash_brighter(self):
        # calculated manually
        matches = [["#1F7ECA", "#56a5e5"],
                   ["#CCCC33", "#dbdb70"],
                   ["#993322", "#d55a46"],
                   ["#663366", "#ab58ab"]]

        for hash1, hash2 in matches:
            self.assertEqual(
                rgb_hash_brighter(hash1, 0.3).lower(), hash2.lower())
