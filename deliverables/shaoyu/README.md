# shaoyu 的 Update 1 数据交付

交付日期：2026-10-15。分支：`data/shaoyu`。语言对：粤语–英语、韩语–英语、俄语–英语、马来语–英语。

**共用代码没有修改。** 所有结果都用共用的 `pipeline.py` 生成（原来在 `update1_data_standard/`，现在改名为 `project_update_1/`，内容不变），设置为 mBERT（tokenizer_sha256 `78b44c32…`）、逐条目处理、`inherit_previous`、`observed`。入口脚本是 `project_update_1/run_unicode_deepseek.py`，标注实现是 `project_update_1/annotate_unicode.py` 和 `project_update_1/annotate_deepseek.py`。

4 组都通过了 `pipeline.py validate`，可以和其他成员的结果一起运行 `split`。做法和理由见 [DECISIONS.md](DECISIONS.md)。

**本目录不含数据集原文。** SwitchLingua 的许可协议禁止转发数据集（全部或部分）。所以 Git 中只保留统计、manifest、ID、`cs_type`、核查记录，以及马来语标注的字符位置（`ms_en/annotation/lid_offsets_shaoyu.jsonl`）。含原文的文件（texts、lid.template、lid、labeled_tokens）要在本地用入口脚本重建，输出到 `deliverables/shaoyu_full/`。这个文件夹被 Git 忽略，**不要提交**。重建方法见第 6 节。

## 1. 交给 Baseline 的文件

先按第 6 节重建 `deliverables/shaoyu_full/`。

| 文件 | 内容 |
|---|---|
| `deliverables/shaoyu_full/<pair>/tokenized/labeled_tokens_shaoyu.jsonl` | 正式的带标签数据（本地重建，不在 Git 中） |
| `<pair>/cs_type_shaoyu.jsonl` | `sequence_id` → `cs_type`，用于按切换类型计算 Macro-Average |
| `<pair>/annotation/pilot_record_ids_shaoyu.txt` | 20 条核查过的 record，划分时必须固定进 train |

`<pair>` 为 `yue_en`、`ko_en`、`ru_en`、`ms_en`。

划分命令（在仓库根目录运行）。先把 4 个 `pilot_record_ids_shaoyu.txt` 合并成 `all_pilot_record_ids.txt`：

```bash
python project_update_1/pipeline.py split --inputs deliverables/shaoyu_full/yue_en/tokenized/labeled_tokens_shaoyu.jsonl deliverables/shaoyu_full/ko_en/tokenized/labeled_tokens_shaoyu.jsonl deliverables/shaoyu_full/ru_en/tokenized/labeled_tokens_shaoyu.jsonl deliverables/shaoyu_full/ms_en/tokenized/labeled_tokens_shaoyu.jsonl --seed 6140 --train-records all_pilot_record_ids.txt --out project_update_1/data/combined
```

我测试过这条命令：train / validation / test 为 5,876 / 683 / 662 个序列，没有 record 跨越集合，80 条核查 record 都在 train。`loader.py` 能正常读取结果。

**`cs_type` 的用法：** `loader.py` 的 `provenance` 中有 `sequence_id`。用它查 `cs_type_shaoyu.jsonl`，就能按切换类型汇总分数。**不要**把 `cs_type` 作为模型输入。

## 2. 目录结构

```text
deliverables/shaoyu/       在 Git 中，不含原文
  README.md                本文件
  DECISIONS.md             做法、理由、已知问题、对共用代码的建议
  <pair>/
    cs_type_shaoyu.jsonl
    extraction/            sequence_index（texts 去掉 text 字段）、extract_issues、extract_manifest（含抽样信息）、sampled_record_ids
    annotation/            review.csv、pilot_record_ids、lid_manifest、build_manifest；ms_en 另有 lid_offsets、failures
    tokenized/             token_issues、stats、manifest
deliverables/shaoyu_full/  本地重建，含原文，不提交
  <pair>/                  上面的文件，加上 texts、lid.template、lid、labeled_tokens
```

所有结果文件名都加了 `_shaoyu`。`deliverables/shaoyu/` 中的 manifest 来自 2026-10-15/16 的原始运行。

## 3. 统计

| 语言对 | record | 序列 | 有效位置 | 切换 | 切换率 | Small / Medium / Large | 切换但 duration 无效 | UNKNOWN subword |
|---|---|---|---|---|---|---|---|---|
| yue_en | 499 | 2,560 | 54,848 | 3,030 | 5.5% | 732 / 846 / 1,130 | 322 | 1,950 |
| ko_en | 500 | 1,529 | 41,863 | 2,237 | 5.3% | 487 / 592 / 1,084 | 74 | 172 |
| ru_en | 500 | 1,564 | 40,585 | 2,082 | 5.1% | 361 / 696 / 966 | 59 | 109 |
| ms_en | 500 | 1,568 | 36,634 | 2,929 | 8.0% | 915 / 878 / 953 | 183 | 1,512 |

按切换类型的切换率：

| 语言对 | Intersentential | Intrasentential | Extra-sentential / Tag |
|---|---|---|---|
| yue_en | 4.9% | 6.0% | 5.7% |
| ko_en | 4.1% | 6.1% | 6.0% |
| ru_en | 3.4% | 5.8% | 6.3% |
| ms_en | 4.8% | 9.8% | 9.7% |

## 4. LID 方法

| | 粤语、韩语、俄语 | 马来语 |
|---|---|---|
| 方法 | `project_update_1/annotate_unicode.py`：按 Unicode 文字系统标注 | `project_update_1/annotate_deepseek.py`：DeepSeek `deepseek-flash`，思考模式，共用提示词 v1.0 |
| 专有名词 | 拉丁字母写的都标 `en` | 由模型按上下文判断，无法判断的标 `UNKNOWN`（v1.0 规则） |
| 核查修正 | 粤语 1 处，韩语、俄语 0 处 | 0 处 |

## 5. 使用前要知道的

1. **核查是模型辅助的。** 每组 20 条 record 由 Claude 核查。报告中不能称为独立的人工核查。
2. **其余条目是伪标签**（`unreviewed`）。在这些数据上得到的 F1，不能称为对人工真值的准确率。
3. **粤语 `[UNK]` 问题。** mBERT 词表缺少 `哋`、`啲`、`咗` 等常用粤语字（55 个字，1,671 次），它们的语言标为 `UNKNOWN`。这可以作为检查表第 5 项（Next Steps）的例子。
4. **逐条目处理会丢失句间切换。** 2,560 个粤语序列中有 903 个没有切换。改为按 record 拼接后，切换数会多 31%–41%。见 DECISIONS.md 第 4 节。
5. **马来语专有名词常标为 `UNKNOWN`**（v1.0 规则），旁边的位置无效。
6. 其他问题见 DECISIONS.md 第 3 节。

## 6. 复现

原始 CSV 不在仓库中。从 Hugging Face 数据集 `Shelton1013/SwitchLingua_text` 下载 `Can_Eng.csv`、`Korean_eng.csv`、`Russian_eng.csv`、`Malay_eng.csv`，放到仓库根目录的 `datasets/`（要自己新建这个文件夹，Git 会忽略它）。然后在仓库根目录运行：

```bash
python project_update_1/run_unicode_deepseek.py --setup-only
```

```bash
python project_update_1/run_unicode_deepseek.py --skip-install
```

入口脚本依次运行：提取全文件、抽样、LID、加入核查结果、tokenize、生成 `cs_type`。结果写到 `deliverables/shaoyu_full/`。最后它把不含原文的文件和 Git 中的 `deliverables/shaoyu/` 对比，一致时输出 `CHECK PASS`。详细说明见 [`project_update_1/README.md`](../../project_update_1/README.md#yue-ko-ru-ms-en)。

大模型输出每次可能略有不同，所以马来语默认使用本目录中的 `ms_en/annotation/lid_offsets_shaoyu.jsonl`：它只保存每个片段的字符位置和语言，脚本从 CSV 取出原文后还原标注。要重新调用 DeepSeek，加 `--ms-lid deepseek --key-file <key 文件路径> --check-against ""`（key 放在仓库外的文件中）。

我在干净的副本中运行过上面的命令。复现结果和本目录的数据一致：标签、统计和 `cs_type` 都相同。粤语、韩语、俄语只有 `annotation_source` 不同，因为它包含脚本名和运行日期。
