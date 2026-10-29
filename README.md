# 基于 RAG + ReAct 的多模态电商导购 Agent

> 竞赛项目 | 2026-05-11 完成基础版本 | 2026-10-29 补质量保障层 | 单人开发
>
> 改了什么、为什么改，见 [IMPROVEMENTS.md](./IMPROVEMENTS.md)。

## 一句话概述

构建了一个具备**自主决策能力**的智能导购 Agent，用户通过文字或图片描述需求后，系统从 190 件真实数码产品中检索匹配商品，并由 LLM 自主调用工具链完成搜索、对比、追问、下单，最终输出结构化的多方案决策建议。

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| UI | Streamlit 1.57 | 纯 Python 单文件应用，含流式输出、Toast 弹窗、可展开面板 |
| 向量存储 | ChromaDB | 轻量级嵌入式向量库，cosine 相似度检索 |
| 稀疏检索 | TF-IDF (sklearn) + BM25 (rank_bm25) | char n-gram 1-3 分词，双路召回 + RRF 融合 |
| LLM | DeepSeek API (deepseek-chat + vision) | chat 负责 Agent 推理，vision 负责图片识别 |
| 数据 | 189 件真实产品，12 品类 | 覆盖手机/电脑/耳机/平板/手表/键鼠/显示器/音箱/相机/游戏主机/存储 |

## 核心架构

```
用户输入（文字/图片）
       │
       ▼
┌─────────────────────────────────────┐
│         ReAct Agent 引擎            │
│   Thought → Action → Observation    │
│         最多 6 轮迭代               │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│           4 大工具                  │
│  search_products / compare_products │
│  ask_user / simulate_order          │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│        混合检索引擎                  │
│  Query 改写(3版本) → TF-IDF + BM25  │
│       → RRF 融合 → 智能打分          │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│       DeepSeek 生成决策辅助          │
│  2-3 方案 / 📌场景 / ✅优点 / ⚠️缺点  │
└─────────────────────────────────────┘
```

## 功能清单

### 已完成（8 大功能模块）

1. **ReAct Agent 自主决策** — 4 工具、6 轮迭代上限，LLM 自主规划调用序列，支持 JSON 参数解析及容错降级
2. **混合检索** — TF-IDF(字符级 char n-gram 1-3) + BM25 双路召回，RRF(K=60, weights 0.5/0.5) 融合排序
3. **Query 改写 + 多路召回** — LLM 生成 3 个改写版本，每版本取 top10，合并去重后输出 top5
4. **多模态图片识别** — DeepSeek Vision 识别品类/颜色/款式，API 失败自动降级为模拟识别
5. **长期记忆用户画像** — JSON 持久化，自动提取品牌偏好/预算区间/拒绝商品，注入 System Prompt
6. **主动推荐** — 基于用户画像生成"猜你喜欢"，30s 无交互自动 Toast 弹窗推荐
7. **流式打字机输出** — API stream=True + st.write_stream 增量渲染，模拟 ChatGPT 打字效果
8. **决策辅助格式** — 强制输出 2-3 方案，每方案含 📌 适用场景 / ✅ 优点 / ⚠️ 缺点，末位追问引导

### 质量保障层（2026-10-29 新增）

9. **事实校验** — `verify.py` 把 Final Answer 逐条回溯到本轮检索结果，检测编造型号 / 跨轮幻觉 / 价格不符 / 已拒绝商品复现，输出可溯源率
10. **降级留痕** — `agent_trace.py` 从 steps 反推本轮所有兜底事件（JSON 解析降级、未知工具、工具异常、跑满轮次），不再静默
11. **评测集与打分** — 24 条用例 + `evaluate.py`，输出命中率 / 可溯源率 / 幻觉率，支持与基线批次对比

> 设计取舍：校验层**全部是纯规则、无 LLM 调用**。用一个 LLM 判断另一个 LLM 有没有说错，校验结果本身也不可信。
> 这套校验的能力边界（能抓什么、抓不了什么）写在 [IMPROVEMENTS.md](./IMPROVEMENTS.md) 第三节。

### 智能打分策略

检索结果经过二次排序：品类完全匹配 +5 分，品类部分匹配 +3 分，关键词重叠每个词 +2 分，叠加 RRF 原始分数，确保品类和关键词相关性优先。

### 容错设计

- Action Input 非 JSON → 降级为纯文本参数解析（逗号/顿号分词等）
- search_products query 为空 → 回退使用用户原始输入
- 品类过滤后结果不足 → 自动放宽品类限制
- 图片识别 API 失败 → 降级为模拟识别
- ChromaDB 集合名含维度+数量，数据变更自动建新集合

## 关键代码指标

| 指标 | 数值 |
|------|------|
| 代码量 | 950 行 (app.py) + 789 行 (质量保障层) |
| Agent 工具数 | 4 |
| 产品库规模 | 189 件 / 12 品类 |
| ReAct 最大迭代 | 6 轮 |
| RRF 参数 | K=60, weights 0.5/0.5 |
| Query 改写数 | 3 版本 |
| 缓存版本 | chroma_db_v4 |
| 用户画像字段 | 6 维度 |
| 校验别名条目 | 1047 条（含歧义剔除） |
| 评测用例 | 24 条 / 4 类意图 |

## 项目结构

```
shopping-guide/
├── app.py                    # 主程序（Agent + UI，950 行）
├── verify.py                 # 事实校验层（纯规则，384 行）
├── agent_trace.py            # 降级留痕（144 行）
├── evaluate.py               # 评测打分（261 行）
├── evals/
│   └── testset.json          # 24 条评测用例
├── data/
│   ├── products.json         # 189件产品数据
│   └── user_profile.json     # 用户画像持久化（本地生成）
├── chroma_db_v4/             # ChromaDB 向量存储（本地生成）
├── requirements.txt          # 依赖清单
└── IMPROVEMENTS.md           # 改版记录：改了什么、为什么改
```

## 快速验证

```bash
# 三个模块都有自测，不需要 API Key
python verify.py --selftest        # 4/4
python agent_trace.py --selftest   # 3/3
python evaluate.py --selftest      # 2/2
```

## 已知优化方向（按性价比排序）

### 高优先
- **语义检索升级** — TF-IDF 字符级语义弱，换 BGE-M3 或 text2vec 语义向量模型可大幅提升召回质量
- **CrossEncoder Rerank** — 检索后加重排序模型，结果更精准
- **品类关键词同步** — CATEGORY_KEYWORDS 需与子品类归并保持一致

### 中优先
- **Agent Self-Reflection** — 工具调用失败后自动换策略重试
- **多模态检索** — CLIP 向量直接参与检索而非仅文字中转
- ~~**评测体系** — 建立测试集，引入 MRR/NDCG 指标~~ → 已部分完成（24 条用例 + 命中率/可溯源率），检索侧的 MRR/NDCG 仍未做
- **语义校验作为第二道闸门** — 规则层管事实，再用独立 prompt 管「推荐是否真的满足需求」

### 低优先
- 实时数据接入（当前 190 件静态库够用）
- 真实支付/订单 API 对接

## 设计亮点

1. **单文件主体 + 外挂保障层** — Agent 与 UI 全部在 app.py 中，不做过早拆分；质量保障单独成模块（verify / agent_trace / evaluate），因为它需要独立自测、且未来要复用到别的领域
2. **无外部 Embedding 依赖** — 因 HuggingFace 国内被墙，选用 sklearn TF-IDF 本地特征提取，零网络依赖
3. **ChromaDB 智能版本管理** — 集合名编码维度+数量，数据变更自动创建新集合，避免手动清理
4. **ReAct 输出解析鲁棒** — greedy regex + boundary 截断，处理 LLM 输出不稳定问题
5. **用户画像闭环** — 从对话中自动提取偏好 → 注入 System Prompt → 影响检索排序 → 持久化跨会话复用
6. **校验层无 LLM 依赖** — 纯规则判定，每条结论都带证据位置，可复现、可回归、可进 CI

---

**GitHub:** https://github.com/aikudewo/shopping-guide

**运行方式:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
