# KGTS — 知识图谱引导的任务合成

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/kgts-project/kgts/actions/workflows/ci.yml/badge.svg)](https://github.com/kgts-project/kgts/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)

KGTS 用来给大模型造训练数据，不同之处在于它从一张知识图谱出发，而不是从一堆文档出发。它面向需要"有据可查、可被验证"的训练任务（用于 SFT 或 RL）的 ML 工程师和研究者，每条数据都能一路追溯到原始材料。它不是又一个 GraphRAG，也不是又一个 doc2qa 工具：那些工具是从文本里抽图、或从文档生成问答对；KGTS 则是用图谱来**规划**要造什么数据，再按这个规划去检索材料、合成任务。

## 它做了什么

KGTS 是 Kimi K3 技术报告（Moonshot AI）中"知识图谱引导的任务合成"（task synthesis）流程的开源实现。整个流程分五个阶段：

1. 一个 agent 从少量人工给定的种子概念出发，逐步扩展出一张概念 DAG（有向无环图，一种树状结构：从粗粒度主题一路细分到子主题）。
2. 采样器（sampler）在这张图上挑选节点和节点组合，由你控制概念空间的哪些部分被变成数据。
3. 检索器（retriever）为每个被选中的节点找真实材料（网页、本地文件、GitHub、arXiv）。检索时的查询语句会带上该节点从根节点下来的路径，这样遇到有歧义的术语时能用上下文消歧。
4. 可插拔的任务类型（task type）把"节点 + 材料"变成训练任务（问答、摘要、对比分析等）。
5. 验证器（verifier）检查任务质量，最后生成一份数据集级别的报告，覆盖覆盖率、重复度、多样性和血缘（provenance，即每条数据从哪来的完整记录）。

KGTS 产品化的核心思想是**把知识图谱当作数据合成的控制面**（control plane）——"控制面"指的是负责决定*做什么*的部分，而不是*怎么做*。图谱决定合成什么、检索什么；具体每个任务怎么生成，交给插件处理。

## 为什么选 KGTS

现有的开源项目各自只覆盖了流程的一段，没有一个覆盖全流程：

| 项目 | 它做的事 | 相比 KGTS 缺什么 |
|---|---|---|
| GraphRAG / KGGen / RAKG / AutoSchemaKG | 文本 → 扁平实体图谱 | 只有扁平图；没有由粗到细的 DAG，没有合成调度 |
| InstructLab | 基于分类体系（taxonomy）的合成数据生成 | 分类体系靠手工维护，不是 agent 自动扩展的 |
| GraphGen | 图谱引导的 QA 合成 + 盲点（ECE）加权 | 没有层级结构，没有网络检索，任务类型与图谱没有解耦 |
| HippoRAG 2 / LightRAG | 子图锚定的检索 | 为问答设计，不是为数据调度设计 |
| AgentInstruct | agent 化的合成流程 | 只有论文，没有开源代码 |

KGTS 的差异化能力：

- **递归式 agent 建图** —— agent 在一个工作队列上循环（原子性检查 → 探索 → 对齐 → 提交），直到 LLM 调用预算用完。
- **可调度的采样** —— 广度 / 深度 / 联合三种采样算子，支持长尾加权（出现频率越低的节点被采到的概率越高，即逆频率加权），作用于层级化的 DAG。
- **agent 化的输入兼容** —— CorpusAdapterAgent 会采样检查任意本地语料，
  自动推断出声明式的提取规则（在真实记录上验证、可缓存、可人工修改），
  新数据格式不需要改代码适配。
- **祖先路径消歧的检索** —— 查询语句带节点从根出发的路径来消歧，按材料来源（web / local / GitHub / arXiv）分别适配。
- **任务类型插件** —— 合成逻辑与图谱解耦；每种任务类型自己声明需要什么材料、怎么生成、怎么验证。
- **完整血缘** —— 每条导出的样本都能沿 `Task → SampleBundle → Node → Material → Run → config_hash` 一路追溯，你随时能回答"这条训练数据是从哪来的"。

## 架构

```text
                 种子配置（粗粒度种子 + 领域声明）
                                  │
   ┌──────────────────────────────▼───────────────────────────────┐
   │ 阶段 A  build/      Explorer → Aligner → Commit → Atomicity  │
   │                     队列驱动的扩展，直到预算耗尽               │
   └──────────────────────────────┬───────────────────────────────┘
                          知识 DAG（NetworkX + SQLite）
   ┌──────────────────────────────▼───────────────────────────────┐
   │ 阶段 B  sample/     breadth | depth | joint 采样算子          │
   │                     逆频率加权 → bundles                     │
   └──────────────────────────────┬───────────────────────────────┘
   ┌──────────────────────────────▼───────────────────────────────┐
   │ 阶段 C  retrieve/   QueryBuilder（祖先路径查询）              │
   │                     来源： local | web | github | arxiv      │
   │                     后处理： license → dedup → rerank        │
   └──────────────────────────────┬───────────────────────────────┘
   ┌──────────────────────────────▼───────────────────────────────┐
   │ 阶段 D  synthesize/ TaskType 插件注册表                       │
   │                      atomic_qa | aggregated_qa | multihop_qa │
   │                      grounded_summary | comparative_analysis │
   └──────────────────────────────┬───────────────────────────────┘
   ┌──────────────────────────────▼───────────────────────────────┐
   │ 阶段 E  verify/ + eval/ + orchestrate/                       │
   │          verifiers → JSONL (SFT/RL) + manifest.json          │
   │          + 覆盖率/重复度/多样性/血缘报告                      │
   └──────────────────────────────────────────────────────────────┘
   横切模块： orchestrate/（checkpoint、断点续跑、预算）
              llm.py（LiteLLM / MockLLM，预算 + 缓存 + 限流）
              ui/（Gradio 工作目录查看器）
```

导出的两种格式分别面向 SFT（监督微调）和 RL（强化学习）两种训练方式。

## 安装

```bash
pip install kgts                # core (no LLM/UI deps)
pip install "kgts[llm]"         # + LiteLLM for real endpoints
pip install "kgts[ui]"          # + Gradio viewer
pip install "kgts[all]"         # everything incl. dev tools
```

从源码安装：

```bash
git clone https://github.com/kgts-project/kgts.git
cd kgts
pip install -e ".[all]"
```

需要 Python 3.11+。

## 快速上手

### 离线 demo（不需要 API key）

`examples/quickstart_offline.py` 会跑完整条流水线——种子 → 图谱 → bundles → 材料 → 任务 → 报告——用的是 `MockLLM`（测试用的假 LLM）和一个本地的小型示例语料，全程离线、结果可复现：

```bash
python examples/quickstart_offline.py
```

### 真实运行

1. 定义粗粒度种子。种子是人工给定的坐标系：每个种子下面的第一层概念由你来写，不是自动生成的。示例种子配置：
   `configs/seeds/cs_small.yaml`、`configs/seeds/medical_small.yaml`、
   `configs/seeds/legal_small.yaml`。
2. 把配置里的 `llm.model` 指向任意 OpenAI 兼容的端点（通过 LiteLLM），例如：

   ```yaml
   llm:
     model: gpt-4o-mini      # or any LiteLLM model string
     api_base: null
   budget:
     max_llm_calls: 2000
   retrieve:
     sources: [local, web]   # web needs TAVILY_API_KEY
   ```

   建议从 `configs/default.yaml` 开始改，里面每个字段都有注释说明。
3. 运行流水线。每个阶段都会保存 checkpoint，重新运行会从中断处继续，而不是从头再来：

   ```bash
   kgts run --config configs/seeds/cs_small.yaml
   kgts report --config configs/seeds/cs_small.yaml
   kgts serve --config configs/seeds/cs_small.yaml   # Gradio 查看器
   ```

导出产物落在 `export.out_dir`：`tasks_sft.jsonl`、`tasks_rl.jsonl`、
`manifest.json`、`report.md` / `report.json`。

## 用你自己的合成 agent 消费 KGTS

KGTS 刻意把"图谱创建"与"数据集生成"解耦：如果你有自己的数据合成 agent，
可以直接消费知识 DAG、采样束和锚定材料——见 [docs/consuming.md](docs/consuming.md)
和现成的 skill `skills/kgts-graph-consumption/SKILL.md`。

## CLI

所有命令都需要 `--config/-c PATH` 参数。

| 命令 | 参数 | 作用 |
|---|---|---|
| `kgts build` | `--cheap-mode`, `--resume/--no-resume` | 阶段 A：扩展 DAG → `graph.db` |
| `kgts sample` | `-n N`, `--resume/--no-resume` | 阶段 B：采样 bundles → `bundles.json` |
| `kgts retrieve` | `--resume/--no-resume` | 阶段 C：bundle → 材料 → `materials.json` |
| `kgts synth` | `--types a,b`, `--resume/--no-resume` | 阶段 D+E：合成任务，然后验证 |
| `kgts export` | `--format sft\|rl`, `--out DIR` | 按格式写 JSONL + `manifest.json` |
| `kgts report` | `--run ID` | 覆盖率/重复度/多样性/质量/血缘报告 |
| `kgts run` | `--resume/--no-resume` | 端到端跑完整条流水线，按阶段断点续跑 |
| `kgts graph` | `--stats` | 查看 `graph.db`（节点/边数、层级直方图） |
| `kgts review` | — | 打印人工审核队列（软约束违规项） |
| `kgts serve` | `--port 7860` | 启动 Gradio 工作目录查看器 |

任何单阶段命令都可以基于工作目录里已有的 checkpoint 重跑（比如 `kgts build` 生成 `graph.db` 之后，就能直接跑 `kgts sample`）。

## 仓库结构

```text
kgts/
├── kgts/
│   ├── models.py       # Node/Edge/Material/Task/SampleBundle/AlignDecision/Run
│   ├── llm.py          # LLMClient protocol, LiteLLMClient, MockLLM, ManagedLLM
│   ├── config.py       # YAML -> typed pydantic settings
│   ├── cli.py          # typer CLI (see table above)
│   ├── graph/          # GraphStore: NetworkX DAG + SQLite, cycle/level invariants
│   ├── build/          # ExplorerAgent, Aligner, AtomicityJudge, expansion loop
│   ├── sample/         # breadth/depth/joint operators, Prioritizer interface
│   ├── retrieve/       # QueryBuilder, MaterialSource plugins, postprocess
│   ├── synthesize/     # TaskType registry + built-in task types
│   ├── verify/         # Verifier protocol, answer_match, rubric_judge
│   ├── eval/           # dataset-level report (coverage/dup/diversity/...)
│   ├── orchestrate/    # stage runner, checkpoints, exporter, artifact store
│   └── ui/             # Gradio workdir viewer (optional extra)
├── configs/            # default.yaml + example seeds (cs/medical/legal)
├── examples/           # quickstart_offline.py (MockLLM, no API key)
├── tests/              # offline test suite (pytest, MockLLM only)
└── docs/               # architecture, fidelity-to-K3, plugins, cost model
```

## 路线图

| 版本 | 范围 | 状态 |
|---|---|---|
| v0.1 | 阶段 A 扩展循环、DAG 存储、Gradio 图谱浏览 | 已在 0.1.0 实现 |
| v0.2 | 阶段 B+C：三种采样算子、祖先路径查询、web+local 检索、血缘 | 已在 0.1.0 实现 |
| v0.3 | 阶段 D QA 类任务类型 + 摘要/对比、rubric 验证器、SFT 导出、自动报告 | 已在 0.1.0 实现 |
| v0.4 | `coding_task` / `data_analysis` 任务类型、沙箱验证器、ECE 长尾插件、RL 格式加固 | 计划中 |
| v0.5 | `agentic_workflow` 任务、下游 SFT 消融实验（KG 引导 vs 随机采样）、Neo4j/Graphiti 后端、人工审核工作流 | 计划中 |

GitHub 和 arXiv 来源已经可以通过 `retrieve.sources` 启用；
ECE 优先级插件挂接在现有的 `Prioritizer` 协议上
（见 `docs/plugins.md`）。

## 与 K3 报告的关系

KGTS 是对 Kimi K3 技术报告中公开描述的机制的**独立开源实现**。报告没有披露细节的地方——原子性判据、对齐阈值、任务类型选择映射、所有默认参数——KGTS 采用自己的工程选择，并在
[docs/fidelity.md](docs/fidelity.md) 中如实标注。KGTS 与 Moonshot AI 没有任何隶属、背书关系，也未使用其代码。

RL 训练本身不在 KGTS 范围内：KGTS 产出的是可验证的任务和 RL 导出行（`prompt + rubric + verifier hook`）；训练循环交给外部框架（LLaMA-Factory、verl 等）。

## 语料合规

你要为自己合成的训练数据和检索到的材料的许可问题负责。KGTS 会在每条材料上记录 `license` 字段；web 检索默认使用白名单过滤
（`retrieve.postprocess.license_mode: whitelist`），但这只是启发式手段，不构成法律意见。本地语料来源（`retrieve.sources: [local]`）是合规优先的路径：把它指向你有权使用的材料。

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。最欢迎的贡献是新的任务类型插件和材料来源——见
[docs/plugins.md](docs/plugins.md)。

## 许可证

[Apache License 2.0](LICENSE)
