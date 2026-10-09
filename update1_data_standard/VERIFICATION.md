# 已完成的验证

验证日期：2026-10-08。环境：Python 3.13.7；实际依赖见 requirements.lock.txt。

## 真实文件提取

对原始 French_eng.csv 只读运行提取程序：8,070 条原始记录，23,563 个文本条目；无提取 error；44 条疑似引号或转义残留 warning。嵌套 instances 的记录也成功提取。没有把 warning 自动当成错误修正，也没有执行全量 LID。

## tokenizer 实测

用 mBERT tokenizer 对一条虚构教学文本运行真实切分和对齐：

Je / pense / que / the / referee / is / un / ##fair / .

示范语言标签为未核查的教学标签。采用 inherit_previous + observed 时，句号继承英语并计入长度，所以 que 后的英语片段是 6 个 subword，duration 类别 1。若改变标点政策，统计可能变化，三个人必须统一。

完整示例输入和实际输出在 examples/worked_example。该目录不是 SwitchLingua 数据，不用于报告模型成绩。tokenizer 提交版本保存在 tokenizer_manifest.json；可用 prepare-tokenizer 的 --revision 参数复现该版本。没有下载模型权重，也没有训练 baseline。

## 规则验证

8 项 unittest 通过：

1. 长度 1、2、3、6、7、9 的类别及相邻位置边界。
2. UNKNOWN 与尾部被截断情形。
3. LID 改写或漏空格被拒绝。
4. Unicode 字符位置、中性字符、跨语言 token 和未知 token。
5. CSV 内换行、instances 嵌套和无法解析记录。
6. 同记录、重复文本保持同组；current 和 prefix 输入均不包含下一 token。
7. 错误标签被验收程序识别。
8. pilot 记录及其完全重复关联组固定为 train。

只做了上述范围的验证；未验证其他 11 对语言的 LID 准确性、XLM-R 分词结果或真实模型训练。
