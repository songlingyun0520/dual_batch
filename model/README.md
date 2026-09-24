# 模型元数据，无权重

- `config.json`：本 demo 使用的 DeepSeek V3.2 结构配置；关闭 MTP，去除 FP8 quantization_config。
- `config.original.json`：迁移源保存的未适配结构配置，仅供对照，不用于启动。
- `tokenizer.json`：分词词表与规则，不是神经网络权重。
- `tokenizer_config.json`：迁移源保存的精简 tokenizer 配置。

这些文件从原实验目录原样迁移。上游为
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)。
随附上游 [MIT 许可证](LICENSE.deepseek)；精简 tokenizer 配置和 demo 结构配置属于实验适配。
本仓库未确定最初下载的上游 revision，不将它们声称为某个上游 commit 的逐字副本。

启动固定 `--load-format dummy`，不会寻找或下载真实权重。不要替换为真实 checkpoint 后
仍使用专家共享和 KV 截断补丁来做模型精度评估。
