# Update 1 数据处理标准与操作指南

这套文件供三个数据成员各处理四对语言使用。目标是把原始 CSV 变成能够交给 baseline 的 subword 序列和正确答案。先用几条文本学习流程，再扩展。它不代替语言标注，也不训练模型。

## 先读这一页

你要做的事：

0. 协调人先将共同文件放入团队仓库的 main；各成员同步 main 后创建自己的分支，后续代码、标注和结果整理都在个人分支中进行；正式交付的结果文件名加上自己的姓名标识。
1. 运行提取程序，得到 `texts.jsonl` 和 `lid.template.jsonl`。
2. 人工或用团队选定的工具填写语言片段，保存为 `lid.jsonl`。程序不会自动判断语言。
3. 运行 tokenizer 和标签程序，得到 `labeled_tokens.jsonl`。
4. 检查样本和统计，将整个语言对目录交给baseline成员。
5. baseline成员统一划分 train / validation / test，再通过 `loader.py` 向 baseline 提供样本。

**目前的建议与决策状态**
- LID 优先建议采用“大模型预标注＋人工核查”，本地逐词语言识别模型作为备选，词典与 Unicode 字符规则作为辅助。
- 具体大模型、版本和调用方式仍需通过小样本试跑选定；**(这不表示已证明其对全部 12 对语言可靠。)**
- tokenizer 也未最终选定，下面用 mBERT tokenizer 演示。
- 暂行实现把每个提取条目作为独立序列；标点和末尾规则须明确传参。

**大家先验证这些规则，确认后使用相同命令和资产文件，不能各选一套。**

## 首次复制到团队 GitHub clone

本文件夹已整理为可以直接复制的公共文件包。复制整个 `update1_data_standard/` 文件夹到你电脑上已经 clone 的团队仓库根目录，保留目录结构和隐藏文件 `.gitignore`。不需要重新下载或解压 ZIP，也不要把文件分散到不同目录。

复制后应为：

```text
团队仓库/
  其他已有文件...
  update1_data_standard/
    README.md
    LID_PROMPT.md
    pipeline.py
    loader.py
    requirements.txt
    requirements.lock.txt
    VERIFICATION.md
    .gitignore
    tests/
    examples/
```

这份公共包只包含代码、说明、依赖和教学样本，不包含原始 CSV、全量个人输出、虚拟环境或模型缓存。`examples/worked_example/` 是已运行的虚构教学例子，可随公共代码提交，不是个人实验结果。

建议首次操作顺序：

1. 在团队仓库中查看 `git status`，先保存已有工作，切到更新后的 main。
2. 创建准备分支，例如 `setup/data-pipeline`；如果已有该分支则直接切换。
3. 将公共文件夹复制到仓库根目录。若已有同名文件夹，先比较，不直接覆盖。
4. 在该文件夹内按“工具与环境”安装依赖并运行检查。
5. 回到仓库根目录，仅暂存这批公共文件，检查后提交并推送准备分支。
6. 在 GitHub 创建指向 main 的 Pull Request，检查并合并。之后通知队友更新 main，再创建各自的数据分支。

第 5 步可使用以下命令；这只是操作示例，本地准备公共包不会自动执行提交或上传：

```bash
git add update1_data_standard/
git diff --cached --stat
git diff --cached
git commit -m "Add shared Update 1 data pipeline and guidelines"
git push -u origin setup/data-pipeline
```

提交前确认暂存区没有其他无关文件。共同文件不加姓名；后续个人交付结果才按姓名规则命名。子目录中的 `.gitignore` 只保护其适用范围，不能代替检查整个仓库是否包含密钥或缓存。

## 开始前 个人 Git 分支与姓名命名规则

### 协调人先把共同文件放进 main

**首次开始时先完成这一步，再让三个数据成员创建工作分支。** 按上面的复制流程，把整个 `update1_data_standard/` 目录放到团队仓库根目录，保留内部相对路径。如果取得的是 ZIP 版本则先解压；不要仅上传 ZIP 而不展开，否则队友不方便阅读、运行和检查代码变更。已有同名目录时先比较内容，不覆盖他人的工作。

应放入 main 的共同文件如下，不加个人姓名后缀：

| 共同文件 | 用途 |
|---|---|
| README.md | 团队处理标准、运行步骤与交付规则 |
| LID_PROMPT.md | 共同的辅助标注提示词 |
| pipeline.py | 文本提取、subword 对齐、标签生成、验收和划分 |
| loader.py | 向 baseline 或前缀实验提供样本批次 |
| requirements.txt、requirements.lock.txt | 依赖范围与已验证环境版本 |
| tests/ | 共同的规则检查 |
| examples/ | 教学样本、实际示例输出和核查表模板 |
| VERIFICATION.md | 已做检查及未验证范围 |
| .gitignore | 排除工作数据、环境与缓存；这是隐藏文件，复制时不要遗漏 |

推荐由协调人在 `setup/data-pipeline` 分支加入这些文件，检查后通过 Pull Request 合并到 main。这里的“放进 main”指共同版本最终保存在 main，并不要求绕过团队的分支保护直接推送 main。以上分支名为示例；如果同名分支已存在，不要重复创建。

原始 CSV、全量中间结果、虚拟环境、缓存和 API 密钥不随这批共同代码提交。个人交付放在自己的 `deliverables/姓名/` 中，通过 PR 检查后也可合入 main；因此 main 不仅能保存代码，也能保存已经接受的交付成果。

`assets/` 当前被 .gitignore 排除，不能以为本地下载的 tokenizer 会自动随代码上传。选定 tokenizer 后，协调人应将模型名称、固定 revision、tokenizer 文件摘要和共同参数记录进 README 或一个单独跟踪的配置文件，并约定冻结资产的共享位置。现有 examples 中的 mBERT manifest 是已验证的演示资产记录，不代表团队已最终选择它。

**现在就可以先提交现有共同代码作为初始版本。** 但说明中的待确认项仍然有效：LID 调用没有接入，tokenizer、标点和末尾策略也需要团队确认。完成这些选择后再更新共同版本，三个人使用同一版本批量处理。

### 先建立自己的分支

每人先从已经包含共同文件的 main 创建个人分支，再开始修改代码、标注或整理数据。个人分支会包含 main 当时的所有共同文件，不需要分别复制 pipeline.py。不要直接在共同主分支上处理个人任务。建议命名为 `data/自己的name`，例如 `data/alice`。姓名标识使用团队内唯一的英文名或拼音，小写字母、数字和短横线，不含空格；同名成员使用不同标识。

如果已经把团队仓库下载到电脑，先检查当前状态，再执行下面的示例。将 `alice` 换成自己的姓名标识；若实际默认分支不叫 main，统一替换为团队实际名称：

```bash
git status
git switch main
git pull --ff-only
git switch -c data/alice
git push -u origin data/alice
```

若已有未提交修改，先确认并保存所属工作，不用丢弃修改来强行切分支。若个人分支已经存在，用 `git switch data/alice`，无需再次创建。每次开始工作可用 `git branch --show-current` 确认所在分支。提交后推送自己的分支，通过 Pull Request 交给团队检查和合并。

### 共同代码更新后怎样同步

发现共同代码或标准需要修改时，在个人分支修改并提交 PR，说明原因和对输出的影响。团队确认后合入 main；不要各自长期维护不兼容的脚本，也不把共同文件改名为 pipeline_alice.py。

其他成员先保存当前工作，确认处于自己的分支，再同步 main：

```bash
git switch data/alice
git fetch origin
git merge origin/main
```

同步完成后，在 `update1_data_standard/` 目录运行 `python -m unittest discover -s tests -v`。如果出现冲突，检查双方修改再解决，不直接用一方覆盖另一方。更新涉及标签规则、tokenizer 或对齐逻辑时，重新核查并生成受影响结果，使用新的输出目录，避免新旧规则混用。

Git 分支是代码与文件的协作方式，和后面的 train / validation / test 数据划分是两回事。换 Git 分支不会自动划分训练和测试数据。

### 正式输出文件名加自己的 name

统一规则：`原文件基本名称_自己的name.扩展名`。例如 Alice 的结果为：

| 工作文件名 | 正式交付文件名 |
|---|---|
| texts.jsonl | texts_alice.jsonl |
| lid.template.jsonl | lid.template_alice.jsonl |
| extract_issues.jsonl | extract_issues_alice.jsonl |
| extract_manifest.json | extract_manifest_alice.json |
| lid.jsonl | lid_alice.jsonl |
| review.csv | review_alice.csv |
| labeled_tokens.jsonl | labeled_tokens_alice.jsonl |
| token_issues.jsonl | token_issues_alice.jsonl |
| stats.json | stats_alice.json |
| manifest.json | manifest_alice.json |
| pilot_record_ids.txt（如有） | pilot_record_ids_alice.txt |

语言对继续由上层目录区分，例如 `deliverables/alice/fr_en/tokenized/labeled_tokens_alice.jsonl`。同一成员负责多对语言时，每对放在各自目录，避免相同文件名互相覆盖。

**与现有代码的衔接：**当前程序仍按下面示例生成固定名称的工作文件，不会自动加姓名。先完成处理与验收，再把工作结果复制到 `deliverables/自己的name/语言对/`，保留 extraction、annotation、tokenized 子目录，并将每个结果文件按上表加后缀。原工作文件可保留，方便继续运行现有命令；正式提交或共享的是带姓名后缀的版本。

姓名只改变文件名，不改 sequence_id、record_id、text_sha256 或正文。读取个人交付时，将 `--texts`、`--lid`、`--input`、`--inputs` 等参数中的路径改成实际带姓名的文件名。比如验证 Alice 的输出：

```bash
python pipeline.py validate --input deliverables/alice/fr_en/tokenized/labeled_tokens_alice.jsonl
```

原始 CSV、共同代码（如 pipeline.py）、依赖文件、共享 tokenizer 资产不加姓名后缀，也不各自改词表。汇总成员生成的最终交付结果同样加其姓名后缀，如 `splits_bob.jsonl`；工作阶段仍可使用原固定名称。

现有 `.gitignore` 忽略了 data 工作目录，所以仅执行 git add 不会提交其中的数据。使用未被忽略的 deliverables 目录整理个人交付；提交前检查实际暂存文件。大文件按团队约定使用 Git LFS 或共享存储，不把虚拟环境、缓存和 API 密钥提交到 GitHub。

## 工具与环境

| 工具 | 用途 | 是否训练模型 |
|---|---|---|
| Python 3.13 | 读取 CSV、处理 JSON、规则计算、批次供给 | 否 |
| Python 内置 csv、json、ast | 读取原始数据和列表单元格 | 否 |
| Hugging Face Transformers / Tokenizers | 加载现成 tokenizer，获取 subword、ID、原文位置 | 否 |
| 团队选定的 LID 方法 | 给词或片段判断语言；人工或工具辅助 | 本包不包含该模型 |
| unittest | 检查标签边界、数据划分和未来信息隔离 | 否 |

编辑器不限。当前数据流程不需要 PyTorch、GPU 或 Excel，不需要下载 Transformer 权重。`loader.py` 是纯 Python 的样本迭代器，不是 PyTorch DataLoader；第四位同学可以在此基础上对接自己的 baseline。

`requirements.txt` 是依赖范围；`requirements.lock.txt` 是此次验证环境冻结版本。三个人优先采用同一 Python 小版本和冻结依赖。不同操作系统如安装失败，共同解决并更新版本，不分别静默升级。

在本目录打开终端，macOS / Linux 执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m unittest discover -s tests -v
```

Windows 的环境激活命令为 `.venv\Scripts\Activate.ps1`。后续命令中的输入路径换成个人电脑上的实际 CSV 路径。使用引号包围带空格的路径。

## 文件目录和统一格式

```text
update1_data_standard/
  README.md
  LID_PROMPT.md
  pipeline.py
  loader.py
  requirements.txt
  requirements.lock.txt
  tests/
  examples/
  assets/
    chosen_tokenizer/
      tokenizer_manifest.json
      tokenizer/                 # 冻结的实际 tokenizer 文件
  data/
    fr_en/
      extraction/
        texts.jsonl
        lid.template.jsonl
        extract_issues.jsonl
        extract_manifest.json
      annotation/
        lid.jsonl
        review.csv
      tokenized/
        labeled_tokens.jsonl
        token_issues.jsonl
        stats.json
        manifest.json
    ...
  combined/
    labeled_tokens.jsonl
    splits.jsonl
    split_manifest.json
```

所有 JSON / JSONL：UTF-8、无 BOM、标准 JSON、每行一个完整对象，不加 Markdown 围栏。原文中的换行以 JSON 的 `\n` 保存，读取后仍恢复成换行。CSV 核查表使用 UTF-8 和表头。上述目录展示程序运行时的工作文件名；正式交付时按前面的规则加个人姓名后缀。

语言对目录如 `fr_en`。语言代码先共同确认；`fil`、`tl` 或其他代码不能仅凭 Philippines 文件名自行决定。原始文件语言名称与代码对应关系由团队检查。

`sequence_id` 是关联键，**不使用行号连接不同输出文件**。`source_row` 指 CSV 表头后的第几条数据记录，从 1 开始；不等于文本文件的物理行号。一个原始记录内的所有条目共享 `record_id`。

输出目录必须为空，程序拒绝覆盖已有结果。修改规则时用新版本目录，避免交付文件混杂。

## 第 0 步 提取正文

先试前三条原始记录：

```bash
python pipeline.py extract --input "/你的路径/French_eng.csv" --languages fr,en --limit 3 --out data/fr_en/extraction
```

全部处理时去掉 `--limit 3` 并使用新目录。提取程序只读取 `data_generation_result`，支持字符串列表及嵌套 `instances`，不会读入 `summary` 或评分文字。

程序保留原文，不做 Unicode 归一化、不删除引号或反斜杠、不翻译、不调整标点。疑似外层引号和转义残留会记录 warning。无法解析的记录记录 error 并继续处理其他记录；交付前必须解释 error 的处置，不能只看“程序结束”。

安全解析使用 JSON 或 `ast.literal_eval`，不执行 `eval`。限制单元格长度并记录异常；仍只处理已知来源的课程文件，不把它视为任意恶意文件的安全沙箱。

`texts.jsonl` 每行主要字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| schema_version | 字符串 | 当前为 1.0 |
| sequence_id | 字符串 | 语言对、文件摘要、原始记录号和条目号组成的唯一 ID |
| record_id | 字符串 | 分组划分的基本单位 |
| item_id | 整数 | 原始记录内第几个提取条目，从 1 开始 |
| language_pair / languages | 字符串 / 字符串列表 | 语言对和允许的两个语言代码 |
| source_file / source_row / source_path | 字符串 / 整数 / 列表 | 原始文件及嵌套位置 |
| text | 字符串 | 原始混合语言文本 |
| text_sha256 | 字符串 | 检查原文是否变化的摘要 |

原始 CSV 一行可能含多个条目。当前实现不拼接条目，但仍将同一记录分在同一个数据集合。若要提供前文对话上下文，需要另行扩展序列构造，不能直接把各条目拼在一起而不处理边界。

## 第 1 步 填写 LID

### 建议方法及优先顺序

**优先建议：用大模型结合完整文本的上下文生成语言标注初稿，再人工核查。** 这是项目实施建议，不是课程硬性要求。它不需要 agent；可以先在聊天界面处理少量样本，后续用普通 Python 程序调用模型 API 批量处理。

| 优先顺序 | 方法 | 怎样使用 | 主要局限 |
|---|---|---|---|
| 1 优先试跑 | 大模型预标注＋人工核查 | 使用统一提示词，输入原文及语言对，输出连续语言片段；检查原文重建并核查语言判断 | 可能误判、改写或漏字；API 可能收费；不能假设对全部语言对同样可靠 |
| 2 备选 | 本地预训练的逐词语言识别模型 | Python 加载现成模型，再转换成统一 LID 格式 | 需要核实支持语言和混合文本能力；输出整句语言的模型不能直接当作逐词 LID 工具 |
| 3 辅助 | 词典规则 | 查词在各语言词典中的归属，对无法唯一判断的词保留 UNKNOWN | 共用词、借词、词形变化和专有名词容易产生歧义 |
| 4 辅助 | Unicode 文字系统规则 | 根据字符所属文字系统提供线索或检查明显冲突 | 文字系统不等于语言；法英都主要使用拉丁字母，不能靠此可靠地区分 |

人工标注也可以用于制作小规模参考样本，但不要求三位成员手工标完全部数据。人工核查应由能判断相应语言的人完成；不能让同一个大模型再次确认，就把结果称为人工真值。

### 优先方案的执行方式

1. 每个语言对先选约 20 条作为试跑，覆盖单次切换、多次切换、专名和歧义词。这个数量用于发现流程问题，不是统计意义上的质量保证。
2. 三位成员采用同一大模型版本和 `LID_PROMPT.md` 提示词，让模型返回 `segments`。具体模型选定后记录名称、版本、提示词版本、运行日期和可用的生成参数。
3. 将返回结果填入对应的 `lid.jsonl`，保留模板中的 sequence_id 和 text_sha256，初始状态为 unreviewed。程序检查结构和原文是否完整保留；人工检查语言判断及切换边界。
4. 记录错误类型和修正。若某语言对质量不足，调整提示词或试用本地逐词 LID 模型，再核查；不要直接扩大处理量。比较多种方法时保留各自来源和版本。
5. 试跑后根据实际处理速度、输出长度和费用估算全量工作，再决定批量调用方式。使用 API 时需要相应账号和凭据，密钥不能写入代码、提交文件或共享给队友。
6. 批量结果仍属于自动伪标签；未逐条人工核查的记录不能统一标为 reviewed。后续报告说明各语言对实际核查范围和剩余不确定性。

**现有代码的范围：**本包提供标注提示词、统一 LID 接口、结构检查及后续对齐代码；没有实现大模型 API 调用或本地 LID 模型推理。仅运行现有 Python 文件不会自动得到语言判断。以上是供团队选择与试跑的方法建议，不代表这些工具已全部接入或验证。

### LID 文件填写格式

把 `lid.template.jsonl` 复制为 `annotation/lid.jsonl`，保留 ID 和摘要。每条文本按语言切成若干连续片段，不必手工计算字符位置。

下面只展示结构，ID 与摘要必须复制真实值：

```json
{
  "sequence_id": "复制模板中的值",
  "text_sha256": "复制模板中的值",
  "annotation_source": "manual 或具体工具及版本和提示词版本",
  "review_status": "unreviewed",
  "reviewer": "",
  "segments": [
    {"text": "Je pense que ", "lang": "fr"},
    {"text": "the referee is unfair", "lang": "en"},
    {"text": ".", "lang": "NEUTRAL"}
  ]
}
```

实际 JSONL 每个对象写成一行。把所有 `segments[].text` 直接拼起来，必须逐字符等于原文，包含空格、换行、引号和重音符号。程序会检查这一点，再自动计算 `[start,end)` 字符位置。

**标注约定：**

- 用上下文给词或语言片段标注，不逐个 subword 猜语言。
- `NEUTRAL` 用于独立标点、纯数字、表情等无法单独表示语言的内容。词内部的撇号可随词保留；不要机械删除。
- 借词或专名根据上下文判断，不能判断时标 `UNKNOWN`。不因人名来自某国就归属该国语言。
- 空格必须保留，可与相邻片段放在一起；对齐时忽略空白字符的语言标签。
- `review_status` 只允许 `unreviewed`、`reviewed`、`needs_review`。真正核查后才改为 `reviewed` 并填写 reviewer。
- 使用语言模型时，可参考 `LID_PROMPT.md`；它只是可复用的标注请求，尚未经过 12 对语言的准确率验证。不要自动把结果标成 reviewed。
- Unicode 字符规则可作为辅助，但不是法英逐词语言识别的替代方法。

代码不上传文本、不调用付费 API。要采用具体 LID 服务，需要团队先决定工具、预算与覆盖语言，然后填写实际 annotation_source。

**建议的试跑核查量：**每对先核查约 20 条，包含短片段、多次切换和不确定词；这是流程检查起点，不是课程规定或质量合格证明。不会判断的语言应找能判断的人核查。用于开发标注规则的试跑记录，正式评价时应留在训练侧或排除，不能再视为未见测试数据。把这些原始 record_id 每行一个写入 `pilot_record_ids.txt`，协调人划分时使用 `--train-records pilot_record_ids.txt`，程序会将整个关联重复组固定到 train。

将核查过程写进 `review.csv`，表头见 examples。未经人工核查的标签即使被用于实验，也必须说明是伪标签；在伪标签上得到的 F1 不能被描述为对人工真值的准确率。

## 第 2 步 准备共同 tokenizer

协调人只准备一次，三位成员共用同一份资产目录。以下是 mBERT **示例**：

```bash
python pipeline.py prepare-tokenizer --model google-bert/bert-base-multilingual-cased --out assets/chosen_tokenizer
```

命令需要网络，只下载 tokenizer。它先将 main 解析为具体提交版本，再保存 tokenizer 文件及哈希。三个人不要分别在不同时间下载 main；共享生成的整个 `chosen_tokenizer` 目录。

若改用 XLM-R，模型标识可改为 `FacebookAI/xlm-roberta-base`，另建资产和输出目录，并先试跑。此次代码验收只保证实测过的 tokenizer；其他 tokenizer 需要验证字符位置映射，尤其重叠或跨语言 token。

切分时统一：`use_fast=True`、`add_special_tokens=False`、`truncation=False`、`padding=False`。不先截短全文，否则会把 duration 数短。之后构造模型输入时才加入需要的特殊 token 和 padding，并相应扩展忽略 mask。

**不同 tokenizer 的 subword 数可能不同，因此 duration 类别也可能不同。** 每套 tokenizer 独立产生标签和统计，不混合进一个 baseline；比较时说明 token 化差异。不能把 mBERT token ID 直接交给 XLM-R 模型。

## 第 3 步 对齐并生成标签

下面明确选择两条暂行规则：中性 token 继承前一个已知语言并计入长度；序列结束视为记录内片段的结束。

```bash
python pipeline.py tokenize --texts data/fr_en/extraction/texts.jsonl --lid data/fr_en/annotation/lid.jsonl --tokenizer-dir assets/chosen_tokenizer --neutral-policy inherit_previous --end-policy observed --out data/fr_en/tokenized
```

默认要求所有 LID 记录已核查。若团队明确要使用尚未核查的自动伪标签，增加 `--allow-unreviewed`；此选择和核查状态会保留在输出中。它是允许生成研究数据的开关，不是质量认证。

### 两项必须共同选择的规则

| 参数 | 行为 | 影响 |
|---|---|---|
| neutral-policy=inherit_previous | 独立中性 token 只继承前一个已解析 token 的语言；序列开头或 UNKNOWN 后仍未知 | 保留标点前缀，切换可落在标点之后；继承的 token 计入 duration |
| neutral-policy=mask | 独立中性 token 标为 UNKNOWN | 相邻切换位置不评价；遇到它的新语言片段长度也可能无法确定，会损失边界样本 |
| end-policy=observed | 将条目末尾作为片段结束 | 预测“本记录内的片段长度”，不能宣称真实对话此处结束 |
| end-policy=censored | 到末尾仍未见下一语言则忽略 duration | 切换答案可保留，但尾部长度答案不使用 |

这不是已核实的老师指定规则。建议先用 `inherit_previous + observed` 跑演示，讨论后冻结为同一规则版本。如果采用 mask，需关注它是否大量排除了句间切换，不能只为了方便而忽略这个偏差。

特殊情况：

- subword 跨越两个已标注语言的字符时标 UNKNOWN；不通过多数投票随意决定。
- `[UNK]` 标 UNKNOWN，避免把未知词当成可靠 subword。
- 空白范围不提供语言信息。含一个真实语言及中性字符的 token 归于该语言。
- 某个 switch 已确定，但新语言片段在 UNKNOWN 处中断时，保留 switch=1，duration=-1。
- 非切换 duration=-1。最后位置没有下一 token，switch=-1 且 mask=false。
- 原始总长度大于模型最大输入长度也不会在此阶段裁掉；后续可以裁历史前缀，但不能重算并截短答案。

### labeled_tokens.jsonl 字段

它保留提取字段，并增加：

| 字段 | 类型 | 含义 |
|---|---|---|
| input_ids | 整数列表 | tokenizer 的 subword 编号 |
| tokens | 字符串列表 | 便于查看的 token；不据此重新拼正文 |
| offsets | 二元整数列表 | 对应原文的字符区间 `[start,end)`，从 0 开始 |
| language_spans | 对象列表 | 由 LID 片段计算的原文语言区间 |
| language_ids | 字符串列表 | subword 语言；无法可靠判断用 UNKNOWN |
| switch_labels | 整数列表 | 0、1；无效为 -1 |
| switch_valid_mask | 布尔列表 | 是否参与 switch 训练和评价 |
| duration_lengths | 整数或 null 的列表 | 下一新语言片段的原始 subword 长度 |
| duration_labels | 整数列表 | 0=1–2，1=3–6，2=7+；忽略为 -1 |
| duration_valid_mask | 布尔列表 | 是否参与 duration 训练和评价 |
| alignment_notes / label_notes | 字符串列表 | 对齐说明、无效标签原因 |
| tokenizer_sha256 | 字符串 | 冻结资产标识 |
| neutral_policy / end_policy | 字符串 | 实际使用的规则 |
| annotation_source / review_status / reviewer | 字符串 | 标签来源与核查状态 |

逐 token 数组长度必须一致。位置 t 的标签预测 t+1。两类任务各有自己的 mask，不能用 switch 是否预测正确来决定真实 duration 标签是否存在。

`manifest.json` 保存输入文件摘要、脚本摘要、实际库版本和处理参数。`stats.json` 的三类比例分母为 duration 有效的切换位置数；切换率分母为 switch 有效位置数；无法计算的比例为 null，不是 0。

## 第 4 步 验收并交付

```bash
python pipeline.py validate --input data/fr_en/tokenized/labeled_tokens.jsonl
```

自动检查通过只代表结构和规则自洽，不证明法语或其他语言的判断正确。还要核查：

1. 提取条目能追回原始记录，无正文改写。
2. 抽样确认语言与 subword 对齐正确。
3. 手算长度 1、2、3、6、7、无切换、多次切换、标点、末尾和 UNKNOWN 示例。
4. `extract_issues.jsonl` 中的 error 已解释，warning 已抽查。
5. `token_issues.jsonl` 中 UNKNOWN 的数量和来源已报告。
6. 每人记录一个真实案例：原样本、问题、处理方法、验证和剩余局限。没有独特 bug 也可展示标注核查和统计差异。

每位同学在自己的分支提交四个语言对的个人交付目录，所有结果文件名加自己的姓名标识，同时说明共同脚本版本、实际依赖版本和个人分支名称。不要交虚拟环境、API 密钥或电脑缓存。原始 CSV 单独保存，不改动。

## 第 5 步 协调人统一划分

三个人交付后只由一套程序统一划分，例如：

```bash
python pipeline.py split --inputs data/fr_en/tokenized/labeled_tokens.jsonl data/ar_en/tokenized/labeled_tokens.jsonl --seed 6140 --out combined
```

传入全部选定语言对文件。程序拒绝混合不同 tokenizer 或标点/末尾规则。

- 先按 `record_id` 分组，再将共享标准化完全相同文本的记录并入同一组。文本归一化仅用于查重，不修改原文和 offsets。
- 大致按 80/10/10 分训练、验证、测试；比例是建议，不是课程硬性要求。每个语言组合按组随机划分，实际序列和位置比例不保证正好相同。
- 某组别不足三个独立组时全部进入 train，清单中会体现，不能据此声称有可用测试分数。
- 这里只检查完全重复文本；近重复、共同生成模板、共同文章来源仍需抽查，未自动解决。
- 固定 `splits.jsonl` 后，各次实验复用。不增加文件后重新随机划分，再把结果与旧划分混比。
- 若有用来开发规则的 pilot 记录，在命令中增加 `--train-records pilot_record_ids.txt`；记录 ID 未在输入中找到会报错，避免无效清单被忽略。
- 文本统计可以描述全数据；阈值、超参数、规则调整依据训练/验证数据，不能依据测试结果。

`splits.jsonl` 保存 sequence_id、record_id、group_id、split。`split_manifest.json` 保存种子、输入摘要及各语言对的实际划分数量。

## 第 6 步 baseline 怎样读取

查看“当前 token”基线收到的一批样本：

```bash
python loader.py --data combined/labeled_tokens.jsonl --splits combined/splits.jsonl --split train --context current --batch-size 4
```

每个样本分开保存：

```json
{
  "inputs": {"input_ids": [1234]},
  "targets": {"switch": 1, "duration": 0},
  "provenance": {"sequence_id": "示例ID", "position": 5, "language_pair": "fr_en"}
}
```

这些数值仅展示格式。**只有 inputs 可以交给预测器**。targets 是答案，provenance 用来追踪与按语言对汇总。LID、整句比例、完整文本、未来 token 都没有出现在 inputs 中。

完整前缀实验可改为 `--context prefix`；输入严格切到 t，不含 t+1。需要限制历史长度时加 `--max-context 128`，只裁左侧历史，标签仍来自完整序列。输出是变长 Python 列表，尚未加模型特殊 token、padding 或张量；后续接 Transformer 时需要相应 collator 和 mask。

当前流程的 streaming 指“已经分好的 subword 序列逐位置到达”，不等于已实现字符实时输入和增量 tokenizer；词内部切分可能依赖完整词的字符，不能夸大为真实键盘流系统。

第四位成员基于该接口统计训练集中 token 的后续切换频率、用验证集选阈值、在固定评价集合算 F1。**本包没有代做 baseline 训练与评分**，也没有最终模型比较结果。

## 不碰真实数据也能跑的小示例

`examples/make_demo.py` 生成一条虚构的法英示范文本和示范 LID，明确标为未核查，不能作为课程实验数据。

```bash
python examples/make_demo.py data/demo
python pipeline.py tokenize --texts data/demo/texts.jsonl --lid data/demo/lid.jsonl --tokenizer-dir assets/chosen_tokenizer --neutral-policy inherit_previous --end-policy observed --allow-unreviewed --out data/demo_tokenized
python pipeline.py validate --input data/demo_tokenized/labeled_tokens.jsonl
```

先读它的实际 tokens、offsets 和 labels，再去处理真实 CSV。这条示例只有一条记录，不能用来计算有意义的泛化 F1。

## 常见报错

| 提示 | 应该检查什么 |
|---|---|
| Output directory must be empty | 使用新版本目录，不覆盖旧结果 |
| Segments must reproduce original text exactly | 空格、换行、引号或文字是否被标注工具改写 |
| Text hash mismatch | 是否修改了提取文本，或把其他样本标签粘过来 |
| LID IDs must exactly match texts IDs | 是否漏标、重复或只提交了部分模板 |
| not reviewed | 完成核查；或明确接受伪标签后使用 allow-unreviewed |
| Frozen tokenizer files changed | 三个人是否使用了相同资产；不要手动改词表 |
| Mixed tokenizers or label policies | 统一配置，不能直接合并不同设置的结果 |

## 参考文档

- [Hugging Face tokenizer 接口与原文位置映射](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [mBERT tokenizer 所属模型说明](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [Python literal_eval](https://docs.python.org/3/library/ast.html#ast.literal_eval)

这些资料支持工具用法；具体标注、标点和末尾策略是本标准的暂行项目选择，不是资料或课程已经验证的结论。
