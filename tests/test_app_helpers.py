import unittest

from app import color_distance, hex_to_rgb, recolor_pixel, rgb_to_hex
from matrix_calculator import MatrixError


class AppHelperTests(unittest.TestCase):
    def test_hex_rgb_conversion(self):
        self.assertEqual(hex_to_rgb("#ff0000"), (255, 0, 0))
        self.assertEqual(rgb_to_hex((47, 128, 237)), "#2f80ed")

    def test_invalid_hex_color(self):
        with self.assertRaises(MatrixError):
            hex_to_rgb("#123")

    def test_color_distance(self):
        self.assertEqual(color_distance((10, 20, 30), (10, 20, 30)), 0)

    def test_recolor_pixel_can_preserve_shades(self):
        dark_red = (120, 0, 0)
        target_red = (255, 0, 0)
        replacement_blue = (0, 0, 255)
        shaded = recolor_pixel(dark_red, target_red, replacement_blue, preserve_shades=True)
        flat = recolor_pixel(dark_red, target_red, replacement_blue, preserve_shades=False)

        self.assertNotEqual(shaded, flat)
        self.assertLess(shaded[2], flat[2])
        self.assertGreater(shaded[2], shaded[0])
        self.assertEqual(flat, replacement_blue)


if __name__ == "__main__":
    unittest.main()
