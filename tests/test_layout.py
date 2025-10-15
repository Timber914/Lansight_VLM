import pathlib, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class TestLayout(unittest.TestCase):
    def test_directories_exist(self):
        for p in [ROOT/'lansight', ROOT/'lansight'/'model', ROOT/'lansight'/'dataset', ROOT/'lansight'/'trainer', ROOT/'assets', ROOT/'assets'/'datasets']:
            self.assertTrue(p.exists(), f"缺少目录: {p}")

    def test_min_sample_data(self):
        self.assertTrue((ROOT/'assets'/'datasets'/'pretrain_data.jsonl').exists())
        self.assertTrue((ROOT/'assets'/'datasets'/'sft_data.jsonl').exists())

if __name__ == '__main__':
    unittest.main()
