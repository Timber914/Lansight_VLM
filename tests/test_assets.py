import pathlib, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]

class TestAssets(unittest.TestCase):
    def test_vision_model_present(self):
        p = ROOT/'lansight'/'model'/'vision_model'/'clip-vit-base-patch16'
        self.assertTrue(p.exists(), '缺少 CLIP 视觉模型文件夹')
    def test_tokenizer_present(self):
        self.assertTrue((ROOT/'lansight'/'model'/'tokenizer.json').exists() or (ROOT/'lansight'/'model'/'tokenizer_config.json').exists(), '缺少分词器文件')
    def test_weights_present(self):
        # 至少满足一种：
        # 1) 项目 out/ 中的原生 .pth 权重
        # 2) 项目 out/transformers/LanSight_Model 中的 transformers 权重
        # 3) 兼容旧路径：assets/{torch_weights,transformers}
        out_pth = list((ROOT/'out').glob('*.pth'))
        out_trf = ROOT/'out'/'transformers'/'LanSight_Model'/'pytorch_model.bin'
        assets_pth = list((ROOT/'assets'/'torch_weights').glob('*.pth'))
        assets_trf = ROOT/'assets'/'transformers'/'LanSight_Model'/'pytorch_model.bin'
        ok = (len(out_pth) > 0) or out_trf.exists() or (len(assets_pth) > 0) or assets_trf.exists()
        self.assertTrue(ok, '缺少任意一种可用权重文件（out 或 assets 任一位置均可）')

if __name__ == '__main__':
    unittest.main()
