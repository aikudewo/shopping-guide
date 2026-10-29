# -*- coding: utf-8 -*-
"""
事实校验层（Verification Layer）
================================
把 Agent 的 Final Answer 逐条回溯到本轮真实的检索结果上，找出三类问题：

1. 幻觉商品 —— 回答里推荐了商品库中不存在的商品，或本轮根本没检索到的商品
2. 价格不符 —— 回答里的价格与该商品在库中的真实价格不一致
3. 用户已拒绝 —— 回答里又推荐了用户画像中标记为「已拒绝」的商品

设计原则：**纯规则、无 LLM 调用**。
理由：如果用一个 LLM 去判断另一个 LLM 有没有说错，校验结果本身也不可信。
规则校验虽然覆盖面窄，但每一条判定都能给出确切的证据位置，可复现、可回归。

用法：
    from verify import verify_answer
    report = verify_answer(answer, steps, products, rejected_names)
    if report["status"] != "PASS":
        ...  # 在 UI 上挂一个告警条

自测：
    python verify.py --selftest
"""

import re
import sys

# 价格写法：¥9999 / 9999元 / 9999块 / ¥ 9,999
PRICE_RE = re.compile(r"(?:[¥￥]\s*([0-9][0-9,]*))|(?:([0-9][0-9,]*)\s*(?:元|块))")

# 方案标题：方案一：商品名 / 推荐2: 商品名
PLAN_RE = re.compile(r"(?:方案|推荐)\s*[一二三四五1-5]\s*[:：]\s*([^\n]+)")

# 容量后缀：512GB / 12+256GB / 1TB
CAPACITY_RE = re.compile(r"\s*\d+(?:\s*\+\s*\d+)?\s*(?:GB|TB)", re.IGNORECASE)

# 价格与商品名的关联半径（字符数）
# LLM 的排版习惯是「方案一：商品名」换行后写「✅优点：¥9999」，
# 所以不能要求同一行；放宽到 150 字符，取距离最近的那次提及。
PRICE_LINK_RADIUS = 150

# 出现这些词时，数字更像是在说预算而不是在报这款的价，跳过不核对
BUDGET_HINTS = ("预算", "上限", "以内", "左右", "以下", "不超过", "封顶", "最多", "别超")

# 短于这个长度的别名不参与匹配，避免 "Pro"、"Max" 这类词到处误命中
MIN_ALIAS_LEN = 4


# ─────────────────────────────────────────────────────────
# 文本工具
# ─────────────────────────────────────────────────────────
def _squeeze(text):
    """去掉所有空白，同时记录每个字符在原文中的位置，便于把匹配结果映射回原文坐标。"""
    chars, idx = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def _aliases(product):
    """
    为一个商品生成别名，覆盖 LLM 的几种省略习惯：
      · 全名              iPhone 16 Pro Max 256GB
      · 去容量            iPhone 16 Pro Max
      · 去品牌前缀        Galaxy S25 Ultra（原名 三星Galaxy S25 Ultra 512GB）
      · 词级前缀          惠普战66 六代（原名 惠普战66 六代 锐龙版 R5-7530U）
    歧义别名（一对多）会在 build_alias_index 里被整条丢弃。
    """
    name = product.get("name", "")
    brand = product.get("brand", "")
    out = {name}

    base = CAPACITY_RE.sub("", name).strip()
    cands = {base}
    if brand and base.startswith(brand):
        cands.add(base[len(brand):].strip())

    # 词级前缀：LLM 常常只写商品名里能识别的那一段
    for c in list(cands):
        segs = c.split()
        if len(segs) > 1:
            for i in range(1, len(segs)):
                cands.add(" ".join(segs[:i]))

    out |= cands
    return {a for a in out if len(a) >= MIN_ALIAS_LEN}


def build_alias_index(products):
    """别名 -> 商品名集合。出现一对多（歧义别名）时整条丢弃，宁可漏检不可误判。"""
    raw = {}
    for p in products:
        for a in _aliases(p):
            raw.setdefault(a, set()).add(p["name"])

    index = {}
    for alias, names in raw.items():
        if len(names) == 1:
            index[alias] = next(iter(names))
    # 长别名优先，保证 "iPhone 16 Pro Max" 不会被 "iPhone 16" 抢先匹配
    return sorted(index.items(), key=lambda kv: len(kv[0]), reverse=True)


def find_mentions(text, alias_index):
    """在 text 中找出所有商品提及，返回 [{name, alias, start, end}]（原文坐标，互不重叠）。"""
    if not text:
        return []

    sq, idx = _squeeze(text)
    if not sq:
        return []

    taken = [False] * len(sq)
    hits = []

    for alias, name in alias_index:
        a_sq, _ = _squeeze(alias)
        if not a_sq:
            continue
        start = sq.find(a_sq)
        while start != -1:
            end = start + len(a_sq)
            if not any(taken[start:end]):
                for k in range(start, end):
                    taken[k] = True
                hits.append({
                    "name": name,
                    "alias": alias,
                    "start": idx[start],
                    "end": idx[end - 1] + 1,
                })
            start = sq.find(a_sq, end)

    hits.sort(key=lambda h: h["start"])
    return hits


def find_prices(text):
    """返回 [(金额, 原文位置)]。"""
    out = []
    for m in PRICE_RE.finditer(text or ""):
        raw = m.group(1) or m.group(2)
        try:
            out.append((int(raw.replace(",", "")), m.start()))
        except (ValueError, AttributeError):
            continue
    return out


def extract_plan_headers(text):
    """
    抽取「方案一：XXX」「推荐二：XXX」这类标题里的商品名。
    返回 [(清洗后的商品名, start, end)]。

    这一步是为了抓「商品库里根本不存在的型号」——只靠商品库做词典匹配永远抓不到它，
    而这类编造型号恰恰是最典型的幻觉。
    """
    out = []
    for m in PLAN_RE.finditer(text or ""):
        raw = m.group(1)
        clean = re.sub(r"[*_`#>]", "", raw).strip(" 　:：,，。、;；")
        if len(clean) >= 2:
            s, e = m.span(1)
            out.append((clean, s, e))
    return out


def _line_bounds(text):
    """每行的 [start, end) 区间（保留给调试与扩展用）。"""
    bounds, start = [], 0
    for i, ch in enumerate(text):
        if ch == "\n":
            bounds.append((start, i))
            start = i + 1
    bounds.append((start, len(text)))
    return bounds


def _is_budget_context(text, pos, span=10):
    """价格前面紧邻的位置出现「预算/上限/左右」这类词时，判定为预算而非报价。"""
    return any(h in text[max(0, pos - span):pos] for h in BUDGET_HINTS)


# ─────────────────────────────────────────────────────────
# 主校验
# ─────────────────────────────────────────────────────────
def verify_answer(answer, steps, products, rejected_names=None):
    """
    answer   : Agent 的 Final Answer 文本
    steps    : run_agent 返回的步骤列表，每步含 observation
    products : 商品库（list[dict]）
    rejected : 用户画像中已拒绝的商品名

    返回 {status, coverage, issues, ...}，status ∈ PASS / WARN / FAIL
    """
    alias_index = build_alias_index(products)
    by_name = {p["name"]: p for p in products}
    rejected_names = set(rejected_names or [])

    issues = []

    # ① 本轮真正「有据可查」的商品：出现在任何一条 Observation 里的商品
    observation_text = "\n".join(
        str(s.get("observation") or "") for s in (steps or [])
    )
    grounded = {h["name"] for h in find_mentions(observation_text, alias_index)}

    # ② 回答里提到的商品（能对上商品库的）
    mentions = find_mentions(answer, alias_index)

    # ③ 方案标题抽取：系统提示词强制要求「方案一：商品名」的格式，
    #    所以可以直接从标题里捞出商品名，连商品库里根本不存在的也能抓出来。
    #    否则「编了一个不存在的型号」这种最典型的幻觉反而会漏检。
    anchors = list(mentions)
    mentioned = []
    for m in mentions:
        if m["name"] not in mentioned:
            mentioned.append(m["name"])

    for header, start, end in extract_plan_headers(answer):
        if find_mentions(header, alias_index):
            continue  # 标题里能找到库中商品，说明不是编的
        mentioned.append(header)
        anchors.append({"name": None, "alias": header, "start": start, "end": end})
        issues.append({
            "type": "幻觉商品",
            "severity": "FAIL",
            "claim": header,
            "detail": "商品库中不存在该商品，回答凭空编造了型号",
        })

    anchors.sort(key=lambda a: a["start"])

    # ④ 幻觉商品：回答提到，但本轮检索结果里从未出现
    for name in mentioned:
        if name not in grounded and name not in {i["claim"] for i in issues}:
            issues.append({
                "type": "幻觉商品",
                "severity": "FAIL",
                "claim": name,
                "detail": "该商品不在本轮检索结果中，回答可能凭空引用",
            })

    # ⑤ 价格核对：把每个价格关联到距离最近的那个商品锚点
    prices = find_prices(answer)
    other_prices = {p.get("price") for p in products}

    for amount, pos in prices:
        if _is_budget_context(answer, pos):
            continue  # 「预算5000左右」这类，不是报价

        nearest, best = None, 10 ** 9
        for a in anchors:
            d = min(abs(pos - a["end"]), abs(a["start"] - pos))
            if d < best:
                nearest, best = a, d

        if nearest is None or best > PRICE_LINK_RADIUS:
            continue
        if nearest["name"] is None:
            continue  # 挂在一个已判为幻觉的商品上，不重复报

        product = by_name.get(nearest["name"])
        if not product:
            continue

        real = product.get("price")
        if amount == real:
            continue

        detail = f"库中实价 ¥{real}"
        if amount in other_prices:
            detail += f"，¥{amount} 是另一件商品的价格（疑似张冠李戴）"
        issues.append({
            "type": "价格不符",
            "severity": "FAIL",
            "claim": f"{nearest['name']} 标注 ¥{amount}",
            "detail": detail,
        })

    # ⑥ 已拒绝商品复现
    for name in mentioned:
        if name in rejected_names:
            issues.append({
                "type": "用户已拒绝",
                "severity": "WARN",
                "claim": name,
                "detail": "该商品在用户画像中被标记为已拒绝，不应再次推荐",
            })

    # ⑦ 可溯源率：被提到的商品里，有多少是真正检索到的
    n_mentions = len(mentioned)
    n_grounded = sum(1 for n in mentioned if n in grounded)
    coverage = (n_grounded / n_mentions) if n_mentions else 1.0

    hard_fail = any(i["severity"] == "FAIL" for i in issues)
    soft_warn = bool(issues) or coverage < 1.0

    return {
        "status": "FAIL" if hard_fail else ("WARN" if soft_warn else "PASS"),
        "coverage": round(coverage, 4),
        "n_mentions": n_mentions,
        "n_grounded": n_grounded,
        "n_grounded_pool": len(grounded),
        "issues": issues,
    }


def format_report(report):
    """把校验结果转成给人看的几行文字。"""
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[report["status"]]
    lines = [
        f"{icon} 事实校验 {report['status']} ｜ 可溯源率 "
        f"{report['coverage'] * 100:.0f}%（{report['n_grounded']}/{report['n_mentions']}）"
    ]
    for i in report["issues"]:
        lines.append(f"· [{i['type']}] {i['claim']} —— {i['detail']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 自测：故意构造一条「编了一个不存在的商品 + 报错价格」的回答
# ─────────────────────────────────────────────────────────
def _selftest():
    products = [
        {"name": "iPhone 16 Pro Max 256GB", "category": "手机", "price": 9999,
         "brand": "Apple", "features": ["A18 Pro芯片"], "suitable_for": []},
        {"name": "Redmi Note 14 Pro+ 5G 12+256GB", "category": "手机", "price": 1999,
         "brand": "Redmi", "features": ["天玑7300"], "suitable_for": []},
        {"name": "惠普战66 六代 锐龙版 R5-7530U", "category": "笔记本电脑", "price": 3999,
         "brand": "惠普", "features": ["R5-7530U"], "suitable_for": []},
    ]
    steps = [{
        "step": 1, "type": "action", "action": "search_products",
        "observation": "检索到 2 件商品：\n1. Redmi Note 14 Pro+ 5G 12+256GB | ¥1999\n"
                       "2. 惠普战66 六代 锐龙版 R5-7530U | ¥3999",
    }]

    # 干净回答：只提检索到的商品，价格也对
    clean = (
        "**方案一：Redmi Note 14 Pro+ 5G**\n📌适用：预算有限\n✅优点：¥1999 性价比高\n\n"
        "**方案二：惠普战66 六代**\n📌适用：办公\n✅优点：¥3999 稳定"
    )
    r1 = verify_answer(clean, steps, products)
    assert r1["status"] == "PASS", f"干净回答被误判：{r1}"
    assert r1["coverage"] == 1.0

    # 幻觉回答：推荐了本轮没检索到的 iPhone，且价格写错
    dirty = (
        "**方案一：iPhone 16 Pro Max**\n📌适用：摄影\n✅优点：¥8888 很值\n\n"
        "**方案二：Redmi Note 14 Pro+ 5G**\n✅优点：¥1999"
    )
    r2 = verify_answer(dirty, steps, products)
    assert r2["status"] == "FAIL", f"幻觉回答没被抓出来：{r2}"
    types = {i["type"] for i in r2["issues"]}
    assert "幻觉商品" in types, f"漏了幻觉商品：{r2}"
    assert "价格不符" in types, f"漏了价格不符：{r2}"

    # 已拒绝商品复现
    r3 = verify_answer(clean, steps, products, rejected_names=["Redmi Note 14 Pro+ 5G 12+256GB"])
    assert any(i["type"] == "用户已拒绝" for i in r3["issues"]), f"漏了已拒绝：{r3}"

    # 空格差异不应影响匹配（"华为 Mate 70 Pro+" vs "华为Mate 70 Pro+"）
    p4 = [{"name": "华为Mate 70 Pro+ 512GB", "category": "手机", "price": 6999,
           "brand": "华为", "features": [], "suitable_for": []}]
    s4 = [{"step": 1, "type": "action", "action": "search_products",
           "observation": "1. 华为Mate 70 Pro+ 512GB | ¥6999"}]
    r4 = verify_answer("推荐华为 Mate 70 Pro+，售价 ¥6999", s4, p4)
    assert r4["coverage"] == 1.0, f"空格差异导致漏匹配：{r4}"

    print("selftest 全部通过（4/4）")
    print("\n示例输出：")
    print(format_report(r2))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
