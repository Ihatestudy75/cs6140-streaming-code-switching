# Yufan：四对语言的随机抽样交付包

仅准备待提交文件，未暂存、提交或推送。每对从已筛查保留集随机抽 **500 条原始 record**，保留抽中 record 的全部现有条目，合计 **4,908 个序列**。所有标签仍为 unreviewed，没有独立语言准确性核验。

| 语言对 | 抽中 record | 交付序列 | 抽样前可选 record |
|---|---:|---:|---:|
| fr_en | 500 | 1,143 | 6,880 |
| de_en | 500 | 1,145 | 6,037 |
| it_en | 500 | 1,299 | 7,210 |
| philippines_en | 500 | 1,321 | 7,904 |

## 抽样规则

固定种子：`random.Random(f"6140:{pair}").sample(sorted(record_ids), 500)`。对 record 抽样，不独立抽 token 或拆散同一 record 的现有条目。抽样母体是本人的筛查保留集，不是原始 CSV 全部 record；此前隔离的条目不恢复。各对完整选择清单见 `extraction/sampling_manifest_yufan.json`。

这是缩小交付量，没有重新做 LID、修改语言判断、重切 subwords 或更改核查状态。完整 68,396 条交付包保留在本地 `data/archive/2026-10-17/full_submission_yufan/`。

## 目录和读取入口

```text
<pair>/
  cs_type_yufan.jsonl
  extraction/      对应原文、LID 模板、提取 issues、manifest、抽样清单
  annotation/      LID、筛查说明、待查 CSV、隔离原因、开发记录、案例
  tokenized/       labeled_tokens、token_issues、stats、manifest
```

- 预测标签：`<pair>/tokenized/labeled_tokens_yufan.jsonl`。
- `cs_type_yufan.jsonl` 与这些标签按 sequence_id 一一对应，取自原 CSV 元数据；只用于分组统计，不作为预测输入。
- 原文、LID、模板和标签按 ID 对应。原文曾逐条按原 CSV 的 source_row/source_path 核对，本次只原样筛选。
- `extract_issues_yufan.jsonl` 保留原全量提取的问题范围，不表示这些问题仍在抽样数据中。
- `review_yufan.csv`、隔离原因文件仅保留抽中 record 相关的待查条目；reviewer 为空，自动筛查不代表人工核查。案例可能来自抽样前的开发数据，不计入正式样本数量。
- `pilot_record_ids_yufan.txt` 只包含当前交付中存在的开发 record；不是人工 reviewed 清单。

## 方法与局限

Lingua 2.2.0 语言片段预标注，mBERT tokenizer；`inherit_previous + observed`；逐提取条目独立处理。完整文本仅用于标签制作，预测输入只允许当前 token 或历史前缀，不能使用未来文本、LID、cs_type 或答案。固定 revision、tokenizer 摘要和实际版本见 `runtime_yufan.json`。

保留集未命中筛查规则不代表语言准确。筛查会影响短切换、duration 和专名等分布；随机抽样不会消除这些偏差。菲律宾 at 同形词问题和其他已知局限见 `DECISIONS_yufan.md`。

## 只读检查

在 `update1_data_standard/` 中运行：

```bash
python deliverables/yufan/scripts/verify_delivery.py
```

检查文件摘要、ID、原文还原、LID、标签规则、统计、cs_type、500-record 抽样清单及开发记录。此检查不验证语言准确率。

抽样脚本为 `scripts/sample_records.py`。在同一工作目录下，可从本地完整包复现到新的输出目录：

```bash
python deliverables/yufan/scripts/sample_records.py --source data/archive/2026-10-17/full_submission_yufan --out data/sample_reproduction_yufan --records 500 --seed 6140
```

## 提交范围

本目录是唯一待提交包；不包括原始 CSV、全量中间数据、完整隔离数据、虚拟环境、模型或 tokenizer 缓存。输出文件保留 `_yufan` 后缀，遵循共享 README 的目录格式。

抽样后已无超过 50 MiB 的单个文件。文件大小及 SHA-256 见 `file_inventory_yufan.json`。本次没有安装或配置 Git LFS，没有执行 Git 暂存、提交或推送。
