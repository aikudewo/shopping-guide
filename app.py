import streamlit as st
import chromadb
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import json
import re
import base64
import uuid
import time
import os
import random

st.set_page_config(page_title="多模态导购Agent", page_icon="🛒", layout="wide")

st.title("🛒 多模态电商导购 Agent")
st.caption("基于 RAG + ReAct 的智能导购 —— 图片识别 · 自主决策 · 长期记忆 · 主动推荐")

PROFILE_PATH = "data/user_profile.json"

# ═══════════════════════════════════════════════════════
# 长期记忆：用户画像
# ═══════════════════════════════════════════════════════
def load_profile():
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "preferred_brands": [],
        "budget_range": None,
        "rejected_products": [],
        "preferred_categories": [],
        "interaction_count": 0,
        "last_searches": [],
    }


def save_profile(profile):
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def profile_to_prompt(profile):
    """将画像转为 System Prompt 片段"""
    parts = ["\n## 用户画像（长期记忆）"]
    if profile.get("preferred_brands"):
        parts.append(f"- 偏好品牌：{'、'.join(profile['preferred_brands'])}")
    if profile.get("budget_range"):
        b = profile["budget_range"]
        if b.get("min") and b.get("max"):
            parts.append(f"- 预算范围：¥{b['min']}-{b['max']}")
        elif b.get("max"):
            parts.append(f"- 预算上限：¥{b['max']}")
        elif b.get("min"):
            parts.append(f"- 预算下限：¥{b['min']}")
    if profile.get("preferred_categories"):
        parts.append(f"- 偏好品类：{'、'.join(profile['preferred_categories'])}")
    if profile.get("rejected_products"):
        parts.append(f"- 已拒绝商品：{'、'.join(p['name'] for p in profile['rejected_products'])}")
    parts.append("- 注意：不要推荐用户已拒绝的商品")
    return "\n".join(parts)


def update_profile_from_conversation(profile):
    """从最近的对话中提取偏好并更新画像"""
    msgs = st.session_state.get("messages", [])
    if len(msgs) < 2:
        return profile

    # 取最近几轮
    recent = []
    for m in msgs[-8:]:
        if m["role"] == "user":
            recent.append(m["content"])

    full_text = " ".join(recent)

    # 提取品牌偏好
    known_brands = ["苹果", "华为", "小米", "三星", "索尼", "Bose", "联想",
                    "华硕", "惠普", "戴尔", "罗技", "雷蛇", "樱桃", "一加",
                    "vivo", "OPPO", "佳明", "漫步者", "森海塞尔", "Keychron"]
    for brand in known_brands:
        if brand in full_text and brand not in profile["preferred_brands"]:
            liked = any(f"喜欢{brand}" in full_text or f"推荐{brand}" in full_text or
                       f"买{brand}" in full_text for _ in [1])
            # 检查是否在"不要XX"的语境中
            if not re.search(rf"(不要|不喜欢|拒绝|排除).*?{brand}", full_text):
                if brand not in profile["preferred_brands"]:
                    profile["preferred_brands"].append(brand)

    # 提取预算
    price_match = re.search(r"预算\s*(\d+)\s*(?:[-~至到]\s*(\d+))?", full_text)
    if price_match:
        profile["budget_range"] = {
            "min": int(price_match.group(1)),
            "max": int(price_match.group(2)) if price_match.group(2) else None,
        }

    # 提取被拒绝的商品
    rejected_pattern = r"(?:不要|不喜欢|排除|不推荐|拒绝)\s*(.+?)(?:[，。,\.]|$)"
    for m in re.finditer(rejected_pattern, full_text):
        name = m.group(1).strip()
        if len(name) >= 2:
            # 检查是否已经记录
            if not any(r["name"] == name for r in profile["rejected_products"]):
                profile["rejected_products"].append({
                    "name": name,
                    "reason": "用户拒绝",
                    "time": time.time(),
                })

    # 记录搜索历史
    for q in recent:
        if len(q) >= 3 and q not in profile["last_searches"]:
            profile["last_searches"].append(q)
    profile["last_searches"] = profile["last_searches"][-10:]  # 只保留最近10条
    profile["interaction_count"] += 1

    return profile


# ═══════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...")

    st.markdown("---")
    st.caption("📷 多模态输入")
    uploaded_image = st.file_uploader("上传商品图片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_image:
        st.image(uploaded_image, caption="已上传", use_container_width=True)

    st.markdown("---")

    # ── 用户画像 ──
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 查看画像", use_container_width=True):
            st.session_state.show_profile = not st.session_state.get("show_profile", False)
    with col2:
        if st.button("🧹 清空画像", use_container_width=True):
            st.session_state.profile = {
                "preferred_brands": [], "budget_range": None,
                "rejected_products": [], "preferred_categories": [],
                "interaction_count": 0, "last_searches": [],
            }
            save_profile(st.session_state.profile)
            st.toast("用户画像已清空", icon="🧹")
            st.rerun()

    if st.session_state.get("show_profile"):
        with st.expander("📋 当前画像", expanded=True):
            p = st.session_state.profile
            st.json(p)

    # ── 主动推荐 ──
    st.markdown("---")
    st.subheader("🎁 猜你喜欢")
    if st.session_state.get("last_recommendation"):
        rec = st.session_state.last_recommendation
        st.markdown(f"**{rec['name']}**\n¥{rec['price']} | {rec['brand']}")
        # 详细卡片
        with st.expander("查看详情"):
            for feat in rec.get("features", []):
                st.caption(f"✅ {feat}")
            if rec.get("suitable_for"):
                st.caption(f"👤 适合：{'、'.join(rec['suitable_for'])}")
            if st.button(f"🔍 以此检索", key="use_rec", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"我想了解 {rec['name']}，价格 {rec['price']}元",
                })
                st.rerun()
    else:
        st.caption("多聊几轮后自动生成...")

    st.markdown("---")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("检索：TF-IDF + BM25 混合检索")
    st.caption("决策：ReAct Agent 自主规划")
    st.caption("记忆：短期(session) + 长期(JSON画像)")
    st.caption("对话：DeepSeek (deepseek-chat)")
    st.caption("商品库：190件真实数码产品（12品类）")

# ═══════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════
CATEGORY_KEYWORDS = {
    "手机": ["手机", "iphone", "小米", "华为", "三星", "一加", "vivo"],
    "笔记本电脑": ["电脑", "笔记本", "轻薄本", "游戏本", "macbook", "thinkpad", "xps"],
    "耳机": ["耳机", "降噪", "tws", "airpods", "头戴", "入耳"],
    "平板": ["平板", "ipad", "pad", "画画", "笔记", "网课"],
    "智能手表": ["手表", "watch", "跑步", "心率", "健康"],
    "键盘": ["键盘", "机械", "cherry", "filco", "vgn"],
    "鼠标": ["鼠标", "无线鼠标", "游戏鼠标", "gpw", "毒蝰"],
    "显示器": ["显示器", "屏幕", "4k", "电竞屏", "oled"],
    "音箱": ["音箱", "音响", "蓝牙音箱", "soundbar", "homepod"],
    "相机": ["相机", "微单", "单反", "gopro", "pocket"],
    "游戏主机": ["游戏机", "ps5", "switch", "xbox", "steamdeck", "vr"],
    "存储设备": ["硬盘", "ssd", "u盘", "nas", "移动硬盘"],
}

# ═══════════════════════════════════════════════════════
# 缓存层
# ═══════════════════════════════════════════════════════
def load_products():
    with open("data/products.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_search_engine(_product_mtime: float):
    """_product_mtime: products.json 的修改时间，文件变了自动重建缓存"""
    products = load_products()
    texts = []
    for p in products:
        text = (
            f"商品：{p.get('name','')} | 品牌：{p.get('brand','')} | 分类：{p.get('category','')} | "
            f"卖点：{'、'.join(p.get('features',[]))} | 描述：{p.get('description','')} | "
            f"适合：{'、'.join(p.get('suitable_for',[]))}"
        )
        texts.append(text)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    embeddings = [tfidf_matrix[i].toarray()[0].tolist() for i in range(len(texts))]

    client = chromadb.PersistentClient(path="./chroma_db_v4")
    # 用 维度+数量 作为集合名，数据一变自动走新集合
    expected_count = len(products)
    col_name = f"products_{len(embeddings[0])}d_{expected_count}n"
    try:
        collection = client.get_collection(col_name)
        if collection.count() != expected_count:
            client.delete_collection(col_name)
            raise Exception("count mismatch, rebuild")
    except Exception:
        collection = client.create_collection(col_name, metadata={"hnsw:space": "cosine"})

    if collection.count() == 0:
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"name": p["name"], "price": p["price"], "brand": p["brand"], "category": p["category"]}
                for p in products
            ],
            ids=[str(i) for i in range(len(products))],
        )

    tokenized_texts = [list(t) for t in texts]
    bm25 = BM25Okapi(tokenized_texts)
    return vectorizer, collection, bm25


# ═══════════════════════════════════════════════════════
# 图片识别
# ═══════════════════════════════════════════════════════
def recognize_image(image_bytes, api_key):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": '识别商品，输出JSON：{"category":"品类","color":"颜色","style":"款式","description":"描述"}',
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别这张图片中的商品"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            temperature=0.1, max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return None
    except Exception:
        return None


def simulate_recognition(image_bytes):
    return {"category": "数码产品", "color": "未知", "style": "未知", "description": "用户上传的商品图片"}


# ═══════════════════════════════════════════════════════
# Query 改写
# ═══════════════════════════════════════════════════════
REWRITE_SYSTEM_PROMPT = (
    "你是电商搜索优化专家。将用户问题改写为3个检索query。每版本30字以内。\n"
    "输出一行JSON数组：[\"版本1\",\"版本2\",\"版本3\"]"
)


def rewrite_queries(query, conv_history_text, api_key):
    try:
        user_content = f"用户当前问题：{query}"
        if conv_history_text:
            user_content = f"对话历史：\n{conv_history_text}\n\n{user_content}"
        user_content += "\n\n输出3个改写版本的JSON数组："
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": REWRITE_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            temperature=0.3, max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            versions = json.loads(m.group())
            if isinstance(versions, list) and len(versions) >= 1:
                return versions[:3]
        return []
    except Exception:
        return []


# ═══════════════════════════════════════════════════════
# 混合检索
# ═══════════════════════════════════════════════════════
def hybrid_search_one(query, vectorizer, collection, bm25, top_k=40):
    products = load_products()
    n = len(products)
    query_vec = vectorizer.transform([query]).toarray()[0].tolist()
    vec_results = collection.query(
        query_embeddings=[query_vec], n_results=n, include=["metadatas", "distances"]
    )
    vec_rank = {m["name"]: rank for rank, m in enumerate(vec_results["metadatas"][0], 1)}
    bm25_scores = bm25.get_scores(list(query))
    bm25_sorted = sorted(
        [(products[i]["name"], bm25_scores[i]) for i in range(n)],
        key=lambda x: x[1], reverse=True,
    )
    bm25_rank = {name: rank for rank, (name, _) in enumerate(bm25_sorted, 1)}
    K = 60
    rrf = {}
    for name, rank in vec_rank.items():
        rrf[name] = 0.5 / (K + rank)
    for name, rank in bm25_rank.items():
        rrf[name] = rrf.get(name, 0) + 0.5 / (K + rank)
    ranked = sorted(rrf.keys(), key=lambda n: rrf[n], reverse=True)[:top_k]
    return [(name, rrf[name], vec_rank.get(name, 999), bm25_rank.get(name, 999)) for name in ranked]


def multi_query_retrieve(queries, vectorizer, collection, bm25):
    seen = {}
    for q in queries:
        items = hybrid_search_one(q, vectorizer, collection, bm25, top_k=10)
        for name, score, vr, br in items:
            if name not in seen:
                seen[name] = (score, vr, br)
    sorted_items = sorted(seen.items(), key=lambda x: x[1][0], reverse=True)[:5]
    products = load_products()
    return [(next(p for p in products if p["name"] == name), score, vr, br)
            for name, (score, vr, br) in sorted_items]


# ═══════════════════════════════════════════════════════
# Agent 工具
# ═══════════════════════════════════════════════════════
def tool_search_products(query=None, max_price=None, min_price=None, category=None):
    query = query or "综合推荐"
    products = load_products()
    vectorizer, collection, bm25 = get_search_engine(os.path.getmtime("data/products.json"))

    # Query 改写：生成多个检索版本
    search_queries = [query]
    if category:
        search_queries.append(f"{category} {query}")
    # 提取品牌关键词额外搜索
    brand_hints = re.findall(r"(苹果|华为|小米|三星|索尼|Bose|联想|戴尔|惠普|华硕|罗技|雷蛇)", query)
    for b in brand_hints:
        search_queries.append(f"{b} {category or ''} {query}".strip())
    search_queries = list(dict.fromkeys(search_queries))[:4]  # 去重，最多4个

    # 混合检索
    results = multi_query_retrieve(search_queries, vectorizer, collection, bm25)

    # ── 智能打分：关键词重叠 + 品类匹配 ──
    query_terms = set(query)
    def smart_score(p, rrf_score):
        s = rrf_score * 10
        # 品类完全匹配大幅加分
        if category and p["category"] == category:
            s += 5
        elif category and category in p["category"]:
            s += 3
        # 品牌/名称含查询词加分
        name_text = p["name"] + p["brand"] + " ".join(p.get("features", []))
        overlap = sum(1 for t in query_terms if t in name_text)
        s += overlap * 2
        return s

    scored = []
    for p, score, vr, br in results:
        s = smart_score(p, score)
        scored.append((p, score, s))

    scored.sort(key=lambda x: x[2], reverse=True)

    # 价格过滤
    filtered = []
    for p, score, s in scored:
        if max_price and p["price"] > max_price:
            continue
        if min_price and p["price"] < min_price:
            continue
        filtered.append((p, score))
        if len(filtered) >= 5:
            break

    # 如果品类过滤后太少，放宽品类限制
    if len(filtered) < 2 and category:
        for p, score, s in scored:
            if len(filtered) >= 5:
                break
            if (p, score) not in filtered:
                if max_price and p["price"] > max_price:
                    continue
                if min_price and p["price"] < min_price:
                    continue
                if (p, score) not in filtered:
                    filtered.append((p, score))

    lines = [f"检索到 {len(filtered)} 件商品："]
    for i, (p, score) in enumerate(filtered[:5], 1):
        lines.append(
            f"{i}. {p['name']} | ¥{p['price']} | {p['brand']} | "
            f"特点：{'、'.join(p['features'][:3])} | 适合：{'、'.join(p['suitable_for'][:2])}"
        )
    return "\n".join(lines)


def tool_compare_products(product_names=None):
    if not product_names:
        product_names = []
    if isinstance(product_names, str):
        product_names = [product_names]
    products = load_products()
    targets = [p for name in product_names for p in products if name in p["name"]]
    if len(targets) < 2:
        return f"仅找到 {len(targets)} 件匹配商品，需要2-3件才能对比"
    lines = ["## 商品对比\n"]
    lines.append("| 维度 | " + " | ".join(p["name"][:15] for p in targets[:3]) + " |")
    lines.append("|------|" + "|".join(["------"] * len(targets[:3])) + "|")
    lines.append("| 价格 | " + " | ".join(f"¥{p['price']}" for p in targets[:3]) + " |")
    lines.append("| 品牌 | " + " | ".join(p["brand"] for p in targets[:3]) + " |")
    all_features = set()
    for p in targets[:3]:
        all_features.update(p["features"])
    for feat in list(all_features)[:6]:
        marks = ["✅" if feat in p["features"] else "❌" for p in targets[:3]]
        lines.append(f"| {feat} | " + " | ".join(marks) + " |")
    return "\n".join(lines)


def tool_ask_user(question="请问你的预算和具体需求是什么？"):
    return f"[ASK_USER]{question}"


def tool_simulate_order(product_name="", quantity=1):
    products = load_products()
    product = next((p for p in products if product_name in p["name"]), None)
    if not product:
        return f"未找到商品「{product_name}」，请确认名称"
    order_id = uuid.uuid4().hex[:12].upper()
    total = product["price"] * quantity
    return (
        f"## 📦 下单成功！\n"
        f"- 订单号：**{order_id}**\n"
        f"- 商品：{product['name']}\n"
        f"- 数量：{quantity}\n"
        f"- 单价：¥{product['price']}\n"
        f"- 合计：**¥{total}**\n"
        f"- 状态：待支付（模拟）"
    )


TOOL_MAP = {
    "search_products": tool_search_products,
    "compare_products": tool_compare_products,
    "ask_user": tool_ask_user,
    "simulate_order": tool_simulate_order,
}


# ═══════════════════════════════════════════════════════
# Agent System Prompt
# ═══════════════════════════════════════════════════════
AGENT_BASE_PROMPT = """你是智能电商导购 Agent（ReAct 模式）。可用工具：

1. search_products — 搜索商品
   参数：{"query":"搜索词","max_price":上限或null,"min_price":下限或null,"category":"品类或null"}
2. compare_products — 对比商品
   参数：{"product_names":["商品名1","商品名2"]}
3. ask_user — 信息不足时追问
   参数：{"question":"追问的问题"}
4. simulate_order — 模拟下单
   参数：{"product_name":"商品名","quantity":数量}

**工作流程：**理解意图→信息不足则追问→充足则检索→用户要对比则对比→用户下单则下单→最后决策辅助

**输出格式：**
```
Thought: 思考过程
Action: 工具名
Action Input: JSON参数
```

得到结果后继续思考，最终：
```
Thought: 信息充足
Final Answer: 给用户的决策辅助回复
```

**决策辅助规则（Final Answer 必须遵守）：**
- 提供 2-3 个商品选项，不要只推一个"最佳"
- 每个选项必须包含：📌适用场景、✅优点、⚠️缺点
- 最后询问用户更倾向于哪个选项，或是否需要调整需求（如预算、品类、功能）
- 格式示例：
  **方案一：iPhone 16 Pro**
  📌适用：苹果生态用户、摄影爱好者
  ✅优点：A18芯片性能强劲、iOS生态完善
  ⚠️缺点：价格较高(¥9999)、不支持应用侧载
  （方案二、三同理）

**基础规则：**每次一个工具、不编造信息、ask_user只用于信息不足时追问、搜到商品必须提供决策辅助（2-3方案+优缺点）、不要只推一个商品、用中文"""


# ═══════════════════════════════════════════════════════
# Agent ReAct 引擎
# ═══════════════════════════════════════════════════════
def parse_react(text):
    thought = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final)|\Z)", text, re.DOTALL)
    action = re.search(r"Action:\s*(\w+)", text)
    action_input = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    final = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    ai_raw = ""
    if action_input:
        ai_raw = action_input.group(1).strip()
        for boundary in ["\nObservation:", "\nThought:", "\nAction:", "\nFinal Answer:"]:
            idx = ai_raw.find(boundary)
            if idx >= 0:
                ai_raw = ai_raw[:idx].strip()
                break
    return {
        "thought": thought.group(1).strip() if thought else "",
        "action": action.group(1).strip() if action else None,
        "action_input": ai_raw,
        "final_answer": final.group(1).strip() if final else None,
    }


def run_agent(user_input, conv_history_text, api_key):
    profile = st.session_state.get("profile", {})
    system_prompt = AGENT_BASE_PROMPT + profile_to_prompt(profile)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = f"用户输入：{user_input}"
    if conv_history_text:
        user_content = f"对话历史：\n{conv_history_text}\n\n{user_content}"
    messages.append({"role": "user", "content": user_content})

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    steps = []
    final_answer = None

    for i in range(6):
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.5, max_tokens=1024,
            stream=True, frequency_penalty=0.4, presence_penalty=0.3,
        )
        raw = ""
        for chunk in resp:
            if chunk.choices[0].delta.content:
                raw += chunk.choices[0].delta.content
        raw = raw.strip()
        parsed = parse_react(raw)
        step = {"step": i + 1, "thought": parsed["thought"], "raw": raw}

        if parsed["final_answer"]:
            step["type"] = "final"
            step["content"] = parsed["final_answer"]
            steps.append(step)
            final_answer = parsed["final_answer"]
            break

        if not parsed["action"]:
            step["type"] = "final"
            step["content"] = raw.replace("Thought:", "").replace("Action:", "").strip()
            steps.append(step)
            final_answer = step["content"]
            break

        step["type"] = "action"
        step["action"] = parsed["action"]
        step["action_input"] = parsed["action_input"]

        tool_fn = TOOL_MAP.get(parsed["action"])
        if not tool_fn:
            observation = f"未知工具「{parsed['action']}」"
        else:
            try:
                params = json.loads(parsed["action_input"])
            except (json.JSONDecodeError, TypeError):
                raw_input = parsed["action_input"].strip()
                if parsed["action"] == "ask_user":
                    params = {"question": raw_input}
                elif parsed["action"] == "search_products":
                    params = {"query": raw_input}
                elif parsed["action"] == "compare_products":
                    names = [n.strip() for n in re.split(r"[,，、]", raw_input) if n.strip()]
                    params = {"product_names": names if names else [raw_input]}
                elif parsed["action"] == "simulate_order":
                    params = {"product_name": raw_input}
                else:
                    params = {"query": raw_input}
            if parsed["action"] == "search_products" and not params.get("query", "").strip():
                params["query"] = user_input
            try:
                observation = tool_fn(**params)
            except Exception as e:
                observation = f"工具执行错误：{e}"

        step["observation"] = observation
        steps.append(step)

        if observation.startswith("[ASK_USER]"):
            final_answer = f"🤔 {observation.replace('[ASK_USER]', '')}"
            break

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    return final_answer or "抱歉，我暂时无法处理。", steps


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════
def get_conversation_context(max_turns=3):
    msgs = st.session_state.get("messages", [])
    pairs = []
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] == "user":
            pairs.insert(0, (msgs[i - 1]["content"], msgs[i]["content"]))
            if len(pairs) >= max_turns:
                break
    if not pairs:
        return ""
    lines = ["## 对话历史"]
    for u, a in pairs:
        lines.append(f"用户：{u}")
        lines.append(f"导购：{a}")
    return "\n".join(lines)


def generate_recommendation(profile):
    """基于用户画像生成主动推荐"""
    products = load_products()
    candidates = []

    for p in products:
        score = 0
        # 偏好品牌加分
        if p["brand"] in profile.get("preferred_brands", []):
            score += 3
        # 偏好品类加分
        if p["category"] in profile.get("preferred_categories", []):
            score += 2
        # 预算匹配加分
        budget = profile.get("budget_range", {})
        if budget:
            pmin, pmax = budget.get("min", 0), budget.get("max", float("inf"))
            if pmin and pmax and pmin <= p["price"] <= pmax:
                score += 2
        # 被拒绝过的商品跳过
        rejected_names = [r.get("name", "") for r in profile.get("rejected_products", [])]
        if p["name"] in rejected_names or any(r in p["name"] for r in rejected_names if r):
            continue
        if score > 0:
            candidates.append((p, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates:
        return candidates[0][0]
    # 没有画像匹配的 → 随机推荐
    return random.choice(products)


# ═══════════════════════════════════════════════════════
# 聊天历史渲染
# ═══════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

# 主动推荐 toast：距上次交互 >30s 时提示
if "last_interaction" in st.session_state:
    elapsed = time.time() - st.session_state.last_interaction
    if elapsed > 30 and st.session_state.messages:
        rec = generate_recommendation(st.session_state.get("profile", {}))
        st.toast(f"🛒 要不要看看这款？{rec['name']} — ¥{rec['price']}", icon="🎁")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("steps"):
            with st.expander("🧠 Agent 决策过程（点击展开）", expanded=False):
                for s in msg["steps"]:
                    icon = "💭" if s["type"] == "action" else "✅"
                    st.markdown(f"**{icon} 第{s['step']}步：思考**")
                    st.caption(s["thought"])
                    if s["type"] == "action":
                        st.markdown(f"**🔧 行动：`{s['action']}`**")
                        try:
                            params = json.loads(s["action_input"])
                            st.json(params)
                        except Exception:
                            st.code(s["action_input"] or "(空)")
                        st.markdown("**📋 结果：**")
                        st.markdown(s["observation"])
                    if s.get("raw"):
                        with st.expander("🔧 原始 LLM 输出", expanded=False):
                            st.code(s["raw"])
                    st.divider()
        st.markdown(msg["content"])
        if msg.get("match_info"):
            with st.expander("📊 最终推荐商品详情", expanded=False):
                st.markdown(msg["match_info"])

# ═══════════════════════════════════════════════════════
# 快捷入口
# ═══════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("#### 💡 试试这些多轮交互：")
    examples = [
        "我想买一款适合学生的轻薄本，预算5000左右",
        "对比一下华为MateBook X Pro和MacBook Pro",
        "帮我推荐一款5000以上的旗舰手机",
        "推荐一款降噪耳机",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

# ═══════════════════════════════════════════════════════
# 用户输入
# ═══════════════════════════════════════════════════════
if prompt := st.chat_input("输入你的导购需求（可先上传图片）..."):
    if not api_key:
        st.error("请先在侧边栏输入 DeepSeek API Key")
    else:
        st.session_state.last_interaction = time.time()

        # ── 图片识别 ──
        image_desc = ""
        if uploaded_image:
            with st.spinner("📷 识别图片中..."):
                recog = recognize_image(uploaded_image.getvalue(), api_key)
                if recog:
                    image_desc = (
                        f"图片识别：品类={recog.get('category')}，颜色={recog.get('color')}，"
                        f"款式={recog.get('style')}，描述={recog.get('description')}"
                    )
                else:
                    recog = simulate_recognition(uploaded_image.getvalue())
                    image_desc = f"(模拟)品类={recog['category']}"

        full_input = f"{image_desc}\n用户文字：{prompt}" if image_desc else prompt

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            if uploaded_image:
                st.image(uploaded_image, width=200)
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                conv_ctx = get_conversation_context()

                with st.spinner("🤖 Agent 分析中..."):
                    answer, steps = run_agent(full_input, conv_ctx, api_key)

                # ── 展示步骤 ──
                with st.expander("🧠 Agent 决策过程（点击展开）", expanded=False):
                    for s in steps:
                        icon = "💭" if s["type"] == "action" else "✅"
                        st.markdown(f"**{icon} 第{s['step']}步：思考**")
                        st.caption(s["thought"])
                        if s["type"] == "action":
                            st.markdown(f"**🔧 行动：`{s['action']}`**")
                            try:
                                params = json.loads(s["action_input"])
                                st.json(params)
                            except Exception:
                                st.code(s.get("action_input", "") or "(空)")
                            st.markdown("**📋 结果：**")
                            st.markdown(s["observation"])
                        if s.get("raw"):
                            with st.expander("🔧 原始 LLM 输出", expanded=False):
                                st.code(s["raw"])
                        st.divider()

                # ── 流式打字机效果 ──
                def stream_answer(text):
                    chunk_size = 3
                    prev = ""
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        yield chunk
                        time.sleep(0.03)
                st.write_stream(stream_answer(answer))

                # ── 更新用户画像 ──
                st.session_state.profile = update_profile_from_conversation(
                    st.session_state.profile
                )
                save_profile(st.session_state.profile)

                # ── 生成主动推荐 ──
                rec = generate_recommendation(st.session_state.profile)
                st.session_state.last_recommendation = rec

                match_info = ""
                for s in steps:
                    if s.get("action") == "search_products" and s.get("observation"):
                        match_info = s["observation"]
                        break

                if match_info:
                    with st.expander("📊 最终推荐商品详情", expanded=False):
                        st.markdown(match_info)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "steps": steps,
                    "match_info": match_info,
                })

                if uploaded_image:
                    uploaded_image = None
                    st.rerun()

            except Exception as e:
                st.error(f"出错了：{e}")
