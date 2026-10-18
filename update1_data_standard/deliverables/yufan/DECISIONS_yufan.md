# Yufan 数据方法与未核验交付说明

当前版本为每对从筛查保留集中固定种子随机抽 500 条原始 record 的子集；抽样前完整数据及全部问题清单留在本地。标签原样保留，仍未核验。

## 实际使用的方法

四对原文均来自原始 SwitchLingua CSV 的 data_generation_result。先完整提取，随后跳过 extraction issues 中命中的 sequence_id 或 record_id（warning/error 均跳过），再按完全相同的文本摘要去重，进行 Lingua 2.2.0 混合语言检测。提取 warning 只表示需要检查，不等于无效；被跳过的原文仍在本地。

法英使用 FRENCH/ENGLISH，德英使用 GERMAN/ENGLISH，意英使用 ITALIAN/ENGLISH，菲律宾–英语使用 TAGALOG/ENGLISH。菲律宾的来源标识仍保留 philippines；选择 Tagalog 检测器的依据是 SwitchLingua 官方 core/prompt.py 中明确使用 Filipino (Tagalog) 的命名，这不是逐条语言准确性认证。

早期 French 对话模型标注 116 条未并入本包。它们未独立核验，只作为开发对照。PinoyBot 只进行了示例运行检查，最终数据没有使用 PinoyBot。

## 筛选和质量局限

采用整条序列隔离，未自动修正语言标签：

- 短英语候选 you know、right?、for sure、don't you think、honestly、trust me、you see、anyway 的字母未全部标为 en。
- LID 或 subword 为 UNKNOWN、对齐歧义、切换位置 duration 无效，或没有有效预测位置。
- French 与早期未核验模型预标注存在语言分歧的样本。
- 菲律宾中被标为英语的独立 at，作为同形词待核查。英语本身也有 at，因此这不是“所有 at 均错误”的断言。

德英、意英此前还做过一次四类英语短语筛查；同一模型重跑结果相同，没有因此获得更高准确性的证据。各对最终筛选范围以 annotation/screening_manifest_yufan.json 为准。

保留集不代表完全无错误。所有标签保持 unreviewed，没有独立人工核验或可报告的真实 LID 准确率。整条隔离 UNKNOWN 或缺失 duration 是保守筛选；原程序本来允许通过 mask 只忽略相关位置，并非所有被隔离序列都不具备使用价值。筛选会改变短切换、duration、专名等样本分布；不能把筛选后分布当作原始语料分布。

仅凭单句测试不能证明 Lingua 整体优于 PinoyBot。菲律宾的 at 示例已作为具体边界问题记录，但尚未证明针对性训练能提升效果。

## 输出边界

完整文本只用于制作标签；提供给预测器的输入应保持 prefix-only。逐条目处理会丢失条目之间的切换。observed 规则仅表示当前提取条目内可观察到的片段长度，不能认定真实对话在此结束。标点继承并计入 subword 长度会影响 duration 分布。

cs_type 完全沿用原始 CSV 的 record 元数据；不保证每个拆分条目本身都仍体现该切换类型，例如来自混合记录的单语条目。这是来源切换类型，不是新人工核验的逐条目类型。

## 已做与未做的核查

已逐条核对最终原文与原 CSV 的 source_row/source_path、ID 和摘要；验证语言片段还原原文、标签规则、mask 和统计；cs_type 与最终标签按 ID 完整对应。每对有一个真实隔离案例，见 annotation/case_study_yufan.json。

提取 warnings/errors 的原因和原清单随包保留，但没有据此完成独立人工语言核验，也没有将自动筛查清单伪装成人工 review。共同脚本未修改；实际代码 commit、摘要、依赖与 tokenizer 固定信息见 runtime_yufan.json。

原始 CSV、tokenizer、模型、环境和被隔离的完整预测标签不提交。本包仅保留最新筛选集及其来源和问题清单。
