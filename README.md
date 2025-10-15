### 第0步

```bash
# 克隆代码仓库
git clone https://github.com/Timber914/Lansight.git
```

```bash
# 下载clip模型到 ./model/vision_model 目录下
git clone https://huggingface.co/openai/clip-vit-base-patch16
# or
git clone https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16
```
```bash
# 下载纯语言模型权重到 ./out 目录下（作为训练VLM的基座语言模型）
https://huggingface.co/Timber0914/Lansight/resolve/main/llm_512.pth
#or
https://huggingface.co/Timber0914/Lansight/resolve/main/llm_768.pth
```
或者
```
bash Lansight_VLM/lansight/scripts/bootstrap_assets.sh
```
自动安装数据集、clip模型和语言模型权重。执行该命令后可以跳过数据集下载，直接进行训练。
## 第1步 开始训练

### 1.环境准备

```bash
conda env create -f environment.yml
```
<details style="color:rgb(128,128,128)">
<summary>注：提前测试Torch是否可用cuda</summary>

```bash
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下载whl文件安装。参考[链接](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2.数据下载
下载需要的数据文件（创建`./dataset`目录）并放到`./dataset`下。
`*.jsonl`为问答数据集，`*images`为配套的图片数据，下载完成后需要解压图像数据。

### 3.开始训练

**3.1 预训练（学图像描述）**

```bash
python -m lansight.trainer.train_pretrain --device cuda:0 --epochs 1 --batch_size 16
```
**3.2 监督微调（学看图对话方式）**

```bash
python -m lansight.trainer.train_sft --device cuda:0 --epochs 4 --batch_size 4
```
### 4.数据集
数据集构成
本模型的训练数据由两个核心部分组成：用于视觉-语言特征对齐的预训练数据集，以及用于提升指令遵循能力的微调数据集。

预训练数据：[Chinese-LLaVA-Vision](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)

此阶段的数据集包含约57万张图像，源自公开数据集 CC-3M 与 COCO 2014。本阶段的目标是让模型学习图像特征与文本描述之间的深层对齐关系，为后续的指令理解能力构建坚实基础。

指令微调数据：[llava-en-zh-300k](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)

此阶段的数据集包含30万条指令微调数据及15万张关联图像。它源于英文视觉指令数据，我们对其进行了高质量翻译、数据清洗与格式优化。这些处理显著增强了模型对中文指令的精确理解与执行能力。

数据格式说明：LLaVA

项目原生支持 LLaVA 数据格式。这是一种专为视觉指令微调任务设计的结构化 JSON 格式。在该格式中，每个数据样本均将一张图像与一段多轮对话相关联，并明确定义了 `用户` 与 `模型` 等角色。这种结构使模型能够高效地学习如何在视觉语境下，进行符合逻辑的多轮问答与指令遵循。
