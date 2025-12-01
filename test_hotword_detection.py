import unittest
from aural import Aural, config

class TestHotwordDetection(unittest.TestCase):
    def setUp(self):
        self.aural = Aural()

    def test_model_selection(self):
        test_cases = [
            ("dolphin mistral are", "dolphin-mistral"),
            ("hey dolphin", "dolphin-mistral"),
            ("deepseek are you there", "deepseek-r1:14b"),
            ("llama", "llama3.2"),
            ("unknown phrase", "deepseek-r1:14b")  # Default model
        ]

        for phrase, expected_model in test_cases:
            with self.subTest(phrase=phrase):
                selected_model = self.aural.select_model_for_phrase(phrase)
                self.assertEqual(selected_model, expected_model)

if __name__ == '__main__':
    unittest.main()
