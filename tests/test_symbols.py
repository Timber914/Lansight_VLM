import pathlib, re, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]

class TestSymbols(unittest.TestCase):
    def test_model_symbols_present(self):
        p = ROOT/'lansight'/'model'/'lansight_vlm.py'
        s = p.read_text(encoding='utf-8', errors='ignore')
        self.assertIn('class VLMConfig', s)
        self.assertIn('class LanSightVLM', s)
    def test_eval_default_load(self):
        p = ROOT/'lansight'/'scripts'/'eval.py'
        s = p.read_text(encoding='utf-8', errors='ignore')
        self.assertIn("default=0", s)

if __name__ == '__main__':
    unittest.main()
