# CS6140 本地 baseline + next step

日期：2026-10-16。数据来自 [PR #1](https://github.com/Ihatestudy75/cs6140-streaming-code-switching/pull/1)，分支 `data/shaoyu`，固定提交 `440cbb032b57b010f909ac30d6deb223f7e642dc`。

**结论：四组交付通过结构、对齐、标签和划分检查，可以用于当前伪标签 baseline。已完成本地训练和评分；未 push、未 merge、未评论 GitHub。** 此结论不等于人工语言标注质量合格。

## 1. README 与数据状态

已读取根目录 README、`update1_data_standard/README.md`、`deliverables/shaoyu/README.md` 和 `DECISIONS.md`。用户已确认最初的“汉语”指“韩语”。四组为粤英、韩英、俄英、马英，每组抽样 500 条原始 record；这不是全量数据，也不是团队全部 12 对语言的最终实验。

| 语言对 | 有效 record | 序列 | 有效切换位置 | 真实标签切换数 | UNKNOWN subword | [UNK] |
|---|---:|---:|---:|---:|---:|---:|
| yue_en | 499 | 2560 | 54848 | 3030 | 1950 | 1671 |
| ko_en | 500 | 1529 | 41863 | 2237 | 172 | 122 |
| ru_en | 500 | 1564 | 40585 | 2082 | 109 | 106 |
| ms_en | 500 | 1568 | 36634 | 2929 | 1512 | 81 |

全部 7,221 个序列通过以下检查：JSONL ID 集合一致、正文 SHA256 与 LID 逐字重建一致、manifest 输入摘要一致、所有 token 数组等长、offset 合法、按现有 offset 和 LID 重新对齐得到相同语言与备注、重算 switch/duration/valid masks 得到相同结果、统计与原交付一致、tokenizer/标点/末尾策略一致、cs_type 一对一关联。

Tokenizer 为 mBERT，revision `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`；资产摘要 `78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f`。策略为 `inherit_previous + observed`。manifest 中脚本摘要按 LF/CRLF 两种形式验证，未修改共同代码。冻结 tokenizer 资产和原始 CSV 未交付，因此没有从原始 CSV 重提取，也没有从 tokenizer 资产重新切词。

### 已知数据问题

- **粤语 1 条解析失败**：`yue_en:cb9f144e3464:108148`，日志为 `invalid character '《' (U+300A)`。提取程序依次尝试 JSON 和 Python literal，均失败。`《` 在正确引用的字符串里本来合法，具体引号或容器错误需回查原始 CSV；目前不能从交付文件还原具体坏行。该条未进入 baseline，499 条有效 record 保留。上游同学已在 DECISIONS 中披露，可反馈复查，非 baseline 引入。
- **粤语缺字**：1,671 个 `[UNK]`，包括常用粤语字；被规则标为 UNKNOWN，周边部分切换与 duration 被 mask。
- **标注性质**：粤、韩、俄主要按文字系统打标签；马来语使用 DeepSeek。每组 20 条 record 的 reviewed 是模型辅助核查，不是独立人工真值。验证和测试均为未核查伪标签，F1 表示与伪标签的一致性。
- 粤语中的书面中文也标为 yue；专名的归属规则在不同语言对中不完全相同。
- 逐条目处理缺少条目间切换；当前实验预测记录内下一 subword，不代表真实键盘字符流。
- 抽样中的 warning：粤语 16、韩语 13、俄语 0、马来语 7。正文保留原样；原始 CSV 缺失，未判定这些来源警告是否都可修复。

## 2. 固定划分与信息隔离

沿用共同 `pipeline.py split`，seed=6140。train / validation / test = **5876 / 683 / 662 个序列**。80 条 pilot record 及其重复关联组固定在 train；record、重复组、NFC+空白标准化的完全重复文本不跨集合。近重复和生成模板泄漏仍未自动解决。

| 语言对 | Train 序列 | Validation 序列 | Test 序列 |
|---|---:|---:|---:|
| ko_en | 1237 | 145 | 147 |
| ms_en | 1279 | 152 | 137 |
| ru_en | 1256 | 160 | 148 |
| yue_en | 2104 | 226 | 230 |

拟合只用训练集；阈值和前缀超参数只用验证集；选择冻结后才计算测试分数。模型输入仅 token ID，不能使用 LID、cs_type、标签、完整正文、未来 token 或语言对元数据。cs_type 只用于分组评分和验证集目标。

## 3. 已完成的模型

**Current-token baseline**：训练集统计 `P(switch at t | token_id[t])`。已见 token 使用经验频率，未见 token 回退训练集总体频率。duration 在有效切换上统计三类频数，以多数类预测；无有效样本时回退训练集总体 duration 多数类。duration 定义为下一新语言片段长度，0=1–2、1=3–6、2=7+ subword。

**Next-step：因果前缀频率模型**：使用截至 t 的最近 N 个 token，最多 N=5；稀疏计数逐级回退至较短历史，再回退全局先验。验证集比较 N∈{2,3,5}、最小支持数∈{1,5,20}，联合选择切换阈值。选中 N=5、最小支持数=5、threshold=0.35。当前 token baseline 的 threshold 也是 0.35。duration 独立按有效 duration 样本数回退。没有额外下载模型、没有付费 API、无需 GPU。

阈值遍历验证集全部不同概率，概率相同的位置同时移动；目标为 12 个“语言对 × cs_type”单元的正类 switch F1 等权平均。并列时选更高阈值；前缀超参数并列选更短历史。这个 Macro-Average 定义在本次实验中明确固定；仓库没有完整课程 spec，最终作业若指定其他聚合方式需对齐。额外提供 pooled、逐语言对、逐类型及 12 单元结果。

## 4. 测试结果

| 模型 | 12 单元 Macro switch F1 | Pooled switch F1 | Precision | Recall | Duration Macro F1（真实切换位置） | 联合 Duration Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| always_no_switch | 0.00% | 0.00% | 0.00% | 0.00% | 21.00% | 0.00% |
| always_switch | 11.11% | 10.89% | 5.76% | 100.00% | 21.00% | 1.61% |
| current_token | 52.81% | 51.84% | 52.20% | 51.48% | 48.49% | 24.43% |
| causal_prefix | 54.61% | 53.72% | 57.33% | 50.53% | 44.11% | 25.97% |

测试集共有 16,387 个有效位置，944 个切换，其中 881 个有有效 duration。真实切换位置上的 duration 指标不依赖 switch 预测是否正确，测量单独分类能力。联合 duration 指标将漏检计为 FN、错误长度类别计为相应 FP/FN、非切换误报计为 FP；真实切换但 duration 未知的位置从联合 duration 评价排除，仍参与 switch 评价。三种 duration 类别等权，零分母类计 0。

| 语言对 | Current-token switch F1 | Causal-prefix switch F1 |
|---|---:|---:|
| yue_en | 41.84% | 42.76% |
| ko_en | 69.90% | 70.39% |
| ru_en | 50.21% | 53.94% |
| ms_en | 48.13% | 49.79% |

前缀模型的主要 switch 指标提高 **1.79 个百分点**。误报由 445 降到 355，漏检由 458 增到 467，体现 precision/recall 权衡。真实切换上的 duration Macro F1 从 48.49% 降到 44.11%；联合 duration Macro F1 从 24.43% 升到 25.97%。因此当前 next step 主要改善切换检测，不代表 duration 预测全面改善。只做单一固定划分，未声称统计显著或已达到课程最终模型要求。

## 5. 接下来做什么

1. **数据验收反馈**：请 Shaoyu 检查粤语原始 CSV 第 108148 条；保持当前 v1 结果不变，修复数据另存版本。若重新纳入样本，记录输入 SHA、split manifest 和可比范围，不悄悄改变已冻结测试集。
2. **建立独立人工核查集**：由能判断各语言的人抽查未用于调规则的 record；特别关注专名、借词、书面中文/粤语以及马来语共用词。模型二次核查不作为人工真值。
3. **改善粤语 UNKNOWN**：比较冻结的 XLM-R tokenizer，或设计从字符 LID 恢复 `[UNK]` 标签的规则。更换 tokenizer 会改变输入 ID 和 duration 标签，必须重新生成全套数据并作为独立实验，不能混入 mBERT 当前结果。
4. **修复句间上下文缺失**：探索按 record 拼接，保留原始条目边界与 cs_type 映射；重新定义边界、duration 和 streaming 输入。先用 train/validation 验证，不能根据当前 test 案例调参后继续把它叫未见测试集。
5. **学习型前缀模型**：在相同冻结划分上实现 causal GRU 或带因果 mask 的模型，双头预测 switch 和 duration，处理不平衡及各自 mask。模型设计和超参数依赖 train/validation；未来 token 不可输入。当前前缀计数模型已提供无需 GPU 的参考。

## 6. 复现与文件

在本目录运行（标准库，实测 Python 3.9.6）：

```bash
python3 -m unittest test_baseline -v
cd source/update1_data_standard
python3 -m unittest discover -s tests -v
cd ../..
python3 run.py --out runs/v2
```

输出目录必须为空；使用新的 runs/v2 保留 runs/v1。现有验证：共同流程 8/8 测试通过，新 baseline 5/5 测试通过；包括长度边界、UNKNOWN、分组划分、pilot 训练隔离、前缀回退、阈值遍历与穷举一致、duration mask、联合误报/漏检、输入与标签隔离。

- `source/`：PR 提交的完整文件快照，保留共同代码。
- `run.py` / `test_baseline.py`：本地实验代码与新增测试。
- `runs/v1/audit.json`：检查和每组原始问题日志。
- `runs/v1/combined/`：固定数据、split 与 manifest。
- `runs/v1/results.json` / `metrics.csv`：完整分数。
- `runs/v1/validation_search.json`：验证集候选搜索，不含测试集选择。
- `runs/v1/*_model.json`：拟合计数表、先验与阈值。
- `runs/v1/*_test_predictions.jsonl`：逐位置预测。
- `runs/v1/error_examples.json`：测试错误样例，仅作报告分析，不用于本轮调整。
- `runs/v1/run_manifest.json`：代码、数据、split SHA256、环境与未验证范围。

所有修改与产物只在本地。
