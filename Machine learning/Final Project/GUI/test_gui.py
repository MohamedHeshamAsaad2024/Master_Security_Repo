import tkinter as tk
import unittest
from main_gui import FakeNewsGUI
import os

class TestFakeNewsGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.gui = FakeNewsGUI(cls.root)
        # Mock features directory for initialization testing
        cls.test_features_dir = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\Output\features_out"
        cls.gui.features_dir.set(cls.test_features_dir)

    def test_ui_initialization(self):
        """Check if all tabs and main controls are created."""
        self.assertEqual(len(self.gui.tabControl.tabs()), 4, "Should have 4 main tabs")
        self.assertTrue(hasattr(self.gui, 'nb_wrapper'), "GUI should have a placeholder for nb_wrapper")
        self.assertEqual(self.gui.model_type.get(), "Best NB", "Default model type should be 'Best NB'")

    def test_model_selection_values(self):
        """Verify the available models in the dropdown."""
        expected_models = [
            "Best NB", "BNB", "CNB", "MNB", 
            "SVM", "XG Boost", "Logistic Regression"
        ]
        self.assertListEqual(self.gui.available_models, expected_models)

    def test_wrapper_initialization_logic(self):
        """Test the logic of initialization (without actually loading if possible, or just checking the call)."""
        # We can't easily mock the entire wrapper loading without deep patching, 
        # but we can check if the method exists.
        self.assertTrue(callable(self.gui._init_wrapper))

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

if __name__ == "__main__":
    # We use a short timeout or just run it. 
    # Note: This requires a display environment.
    unittest.main()
