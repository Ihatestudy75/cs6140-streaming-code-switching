# shaoyu 的数据处理记录

记录日期：2026-10-15。范围：粤语–英语、韩语–英语、俄语–英语、马来语–英语。

本文件只记录我对这 4 组的做法。为遵守 SwitchLingua 的许可协议，本文件和 Git 中都不引用数据集原文，只用 sequence_id 和字符位置。**共用代码 `pipeline.py` 没有修改**，和 `main`（提交 `367d6ec`）完全一致。共用文件夹 `update1_data_standard/` 后来改名为 `project_update_1/`，`pipeline.py` 和 `loader.py` 内容不变。第 4 节列出对共用代码的建议，由团队决定是否采用。

## 1. 共用设置（和 README 一致）

| 项目 | 设置 |
|---|---|
| 代码 | 共用的 `pipeline.py` 原样（提交 `367d6ec`，现在在 `project_update_1/`） |
| Tokenizer | mBERT `google-bert/bert-base-multilingual-cased`，revision `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`，tokenizer_sha256 `78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f` |
| 序列 | 原来的逐条目处理：每个提取条目一个序列 |
| 标签规则 | `--neutral-policy inherit_previous`，`--end-policy observed` |
| 未核查数据 | `--allow-unreviewed` |

**Windows 注意：** 在 Windows 上运行 `prepare-tokenizer` 会用 CRLF 保存部分 tokenizer 文件，哈希值会变成 `28108c3b…`，`split` 会拒绝合并。我把这些文件转成 LF 后，原来的 `tree_hash` 算出 `78b44c32…`，和 README 的值一致。

## 2. 我的做法

### 2.1 抽样

每组随机抽 500 条原始 record。抽样规则：`random.Random(f'6140:{pair}').sample(range(1, total+1), 500)`，见 `project_update_1/run_unicode_deepseek.py` 的 `sample()`。

做法：先用原来的 `pipeline.py extract` 提取整个 CSV，再筛选抽中的 record。所以 `record_id` 和 `source_row` 都是原文件的行号。

粤语抽中的 500 条中，有 1 条无法解析（`extract_issues` 中 severity = error），所以只有 499 条 record 有条目。

### 2.2 语言代码

`yue`、`ko`、`ru`、`ms`、`en`。

### 2.3 LID：粤语、韩语、俄语

`project_update_1/annotate_unicode.py`（原名 `lid_script.py`）按 Unicode 文字系统逐字符标注：

| 字符 | 标签 |
|---|---|
| 母语文字（粤语：汉字；韩语：韩文和汉字；俄语：西里尔字母） | 母语代码 |
| 拉丁字母 | `en` |
| 数字、标点、符号 | `NEUTRAL` |
| 其他文字 | `UNKNOWN` |
| 空白 | 跟随前一个字符 |

**拉丁字母写的专有名词标 `en`。** 例如韩语中的 `WWE`、俄语中的 `Marvel`。理由：实时预测器看到的是文字从韩文变成拉丁字母。标 `UNKNOWN` 会使这些位置的 switch 和 duration 无效。报告中要说明：这是按文字形式的标注，不是语言学归属。

### 2.4 LID：马来语

马来语和英语都用拉丁字母，文字规则不能用。我用 `project_update_1/annotate_deepseek.py`（原名 `lid_llm.py`）调用 DeepSeek。

**最终方法：** `deepseek-flash`（DeepSeek-V4.1-Flash），**思考模式开启**（reasoning_effort = high），temperature 0，JSON 输出，**提示词 v1.0**。v1.0 和共用的 `LID_PROMPT.md` 内容相同，只去掉了 JSON 示例外面的代码围栏，并填入语言代码 `ms（马来语）` 和 `en（英语）`。输入是逐条目的 `texts.jsonl`，和其他成员一样。1,568 个条目全部成功，0 个失败。

**为什么开启思考模式：** 选定方法前，我做了以下试验：

1. 用 20 条 record 比较 `deepseek-flash` 和 `deepseek-v4-pro`（都关闭思考）。Pro 把很多明显的英语标成 `ms`，所以选 Flash。
2. 关闭思考的 Flash 有时把整条记录标成 `ms`：500 条中有 89 条没有任何 `en` 片段。
3. 在 20 条 record（797 个词）上，和核查过的参考答案比较，不计 `UNKNOWN` 的错误词数从 78 个减少到 13 个。
4. 我也试过修改提示词（v1.1），但为了和团队的共用提示词保持一致，**没有采用**。

**最终结果的检查：**

- 只有 1 条 record 没有 `en`：`ms_en:e7c9e2acc79a:000186`，它的原文确实全是马来语。
- 没有「含 2 个以上常见英语词却没有 `en`」的条目。

**专有名词：** 按 v1.0 提示词，由模型根据上下文判断。无法判断的标 `UNKNOWN`（484 个片段，几乎全是人名、地名、品牌）。英语名字（例如 `Manchester United`）常被标为 `en`。

**为什么马来语和其他 3 组的专有名词处理不同：** 粤语、韩语、俄语用文字规则，无法识别名字，所以拉丁字母的名字都标 `en`（见 2.3）。马来语用大模型，按共用提示词处理名字。

**费用：** 最终运行约 $3.4（51.4 万输入 token，274 万输出 token，DeepSeek 高峰价格）。加上试验，马来语共花费约 $4.4。

### 2.5 核查

- 每组随机选 20 条 record（种子 `6140:<pair>:pilot`），这些 record 的所有条目标为 `reviewed`。
- record ID 在 `<pair>/annotation/pilot_record_ids_shaoyu.txt`。划分时用 `--train-records` 固定进 train。
- 粤语、韩语、俄语：检查所有标成 `en` 的片段。韩语、俄语无错误。粤语有 1 处：`yue_en:cb9f144e3464:085084:007` 第 10–11 个字符的 `d` 是罗马化的「啲」，改为 `yue`。在全部粤语中查找夹在汉字之间的短拉丁词，没有其他罗马化粤语。
- 马来语：和之前逐词核查过的参考答案比较，797 个词中，非专有名词的语言错误为 0。差别只在专有名词上：18 个词被标为 `en`（例如 `Manchester United`），39 个词被标为 `UNKNOWN`。按 v1.0 规则，这两种都可以接受，所以没有修正。
- **核查是模型辅助的。** 由 Claude 完成，我委托。`review_shaoyu.csv` 的 reviewer 一栏写明了这一点。报告中不能称为独立的人工核查。
- 其余条目都是 `unreviewed`，是伪标签。

## 3. 已知问题

1. **mBERT 词表缺少常用粤语字。** `哋`、`啲`、`咗`、`嗰`、`喺` 等 55 个字被切成 `[UNK]`，共 1,671 次。按 README，`[UNK]` 的语言为 `UNKNOWN`，所以旁边的切换位置和 duration 无效。模型输入也会丢失这些字的信息。
2. **Large 类别比例高**（35%–50%），因为 duration 按 subword 计数。
3. **很多粤语序列没有切换。** 2,560 个粤语序列中有 903 个（35%）没有切换，因为逐条目处理后，很多条目只有一种语言。
4. **粤语有 7 条 record 完全没有英语**，这是数据集本身的问题，没有删除。
5. **粤语文件中有书面中文**，也标为 `yue`。
6. **马来语中标为 `UNKNOWN` 的专有名词使旁边的位置无效。** 共 1,283 个 subword 因此标为 `UNKNOWN`，183 个切换的 duration 无效。

## 4. 对共用代码的建议（未实施，由团队决定）

我曾在本地试过下面这些修改，结果用来说明影响。现在交付的数据**没有使用**这些修改。

| 建议 | 影响 |
|---|---|
| 把同一条 record 的条目拼接成一个序列 | 逐条目处理会丢失条目之间的切换。在粤语、韩语、俄语中（标注方法相同），拼接后切换数多 31%–41%（逐条目：yue 3,030 / ko 2,237 / ru 2,082；拼接：3,968 / 3,159 / 2,933）。句间切换受影响最大 |
| 在 `texts.jsonl` 中保留 `cs_type` | Spec V.A.3 要求按切换类型算 Macro-Average。现在用单独的 `cs_type_shaoyu.jsonl` 代替 |
| `extract` 增加随机抽样选项 | 现在用 `run_unicode_deepseek.py` 的 `sample()` 代替 |
| `tree_hash` 计算前把 CRLF 当作 LF | Windows 用户不用手动转换 tokenizer 文件 |
| `[UNK]` 按字符的 LID 标注，或者改用 XLM-R | 解决粤语的 `[UNK]` 问题 |
