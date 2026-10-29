# -*- coding: utf-8 -*-
"""
评测脚本（Evaluation Harness）
=============================
回答一个原来回答不了的问题：**改了提示词/检索之后，到底是变好了还是变差了？**

做法：每次跑完 Agent，把「问题 + 回答 + 决策步骤 + 耗时」落成一行 JSONL；
再用这个脚本对整批记录打分，得到一组可比对的数字。

用法：
    # 1. 先跑一批，生成记录（在 app 里每次对话会自动追加到 data/runs.jsonl）
    # 2. 打分
    python evaluate.py --runs data/runs.jsonl
    # 3. 改了提示词后再跑一批，和旧的一批对比
    python evaluate.py --runs data/runs_v2.jsonl --baseline data/runs.jsonl

记录格式（每行一条）：
    {"case_id":"R01","query":"...","answer":"...","steps":[...],"latency_ms":1234}

自测（用内置的假数据跑一遍，不需要 API）：
    python evaluate.py --selftest
"""

import argparse
import json
import os
import statistics
import sys

import agent_trace
import verify

BASE = os.path.dirname(os.path.abspath(__file__))
TESTSET = os.path.join(BASE, "evals", "testset.json")
PRODUCTS = os.path.join(BASE, "data", "products.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def mentioned_products(answer, products, alias_index):
    return [h["name"] for h in verify.find_mentions(answer, alias_index)]


def score_one(case, run, products, alias_index, by_name):
    """对单条记录打分，返回 {hit, reason, verify, trace, ...}"""
    answer = run.get("answer") or ""
    steps = run.get("steps") or []
    names = mentioned_products(answer, products, alias_index)
    picked = [by_name[n] for n in names if n in by_name]

    exp = case.get("expect") or {}
    kind = case.get("type")

    hit, reason = False, ""

    if kind == "recommend":
        cat, cap = exp.get("category"), exp.get("price_max")
        ok = [p for p in picked if p["category"] == cat and p["price"] <= cap]
        hit = bool(ok)
        reason = f"命中 {len(ok)} 件" if hit else (
            f"未命中（提到 {len(picked)} 件，无 {cat} 且 ≤¥{cap}）" if picked else "回答里没识别出商品"
        )

    elif kind == "compare":
        need = exp.get("min_products", 2)
        hit = len(set(names)) >= need
        reason = f"对比了 {len(set(names))} 件" if hit else f"只提到 {len(set(names))} 件，需要 ≥{need}"

    elif kind == "ask_user":
        asked = any(s.get("action") == "ask_user" for s in steps)
        hit = asked or (("？" in answer or "?" in answer) and not names)
        reason = "有追问" if hit else "没追问，直接给了推荐"

    elif kind == "order":
        hit = "订单号" in answer
        reason = "生成了订单" if hit else "没有生成订单"

    rep = verify.verify_answer(
        answer, steps, products,
        rejected_names=run.get("rejected_names") or [],
    )
    tr = agent_trace.analyze_steps(steps, image_simulated=run.get("image_simulated", False))

    return {
        "case_id": case["id"],
        "type": kind,
        "hit": hit,
        "reason": reason,
        "status": rep["status"],
        "coverage": rep["coverage"],
        "hallucinations": sum(1 for i in rep["issues"] if i["type"] == "幻觉商品"),
        "price_errors": sum(1 for i in rep["issues"] if i["type"] == "价格不符"),
        "degraded": tr["n_degraded"],
        "rounds": tr["rounds"],
        "latency_ms": run.get("latency_ms") or 0,
    }


def aggregate(rows):
    n = len(rows)
    if not n:
        return {}
    return {
        "n": n,
        "命中率": sum(r["hit"] for r in rows) / n,
        "可溯源率": statistics.mean(r["coverage"] for r in rows),
        "幻觉率": sum(1 for r in rows if r["hallucinations"]) / n,
        "价格错误率": sum(1 for r in rows if r["price_errors"]) / n,
        "平均降级次数": statistics.mean(r["degraded"] for r in rows),
        "平均轮数": statistics.mean(r["rounds"] for r in rows),
        "平均耗时ms": statistics.mean(r["latency_ms"] for r in rows) if any(r["latency_ms"] for r in rows) else 0,
    }


def print_table(rows):
    head = f"{'case':<6}{'类型':<11}{'命中':<6}{'校验':<7}{'可溯源':<8}{'幻觉':<6}{'降级':<6}{'轮数':<6}"
    print(head)
    print("-" * len(head) * 1)
    for r in rows:
        print(
            f"{r['case_id']:<6}{r['type']:<11}"
            f"{'✅' if r['hit'] else '❌':<6}"
            f"{r['status']:<7}"
            f"{r['coverage'] * 100:>5.0f}%  "
            f"{r['hallucinations']:<6}{r['degraded']:<6}{r['rounds']:<6}"
        )


def print_summary(agg, baseline=None):
    print("\n" + "=" * 56)
    print("汇总")
    print("=" * 56)
    for k, v in agg.items():
        if k == "n":
            print(f"  样本数            {v}")
            continue
        val = f"{v * 100:.1f}%" if k.endswith("率") else f"{v:.2f}"
        line = f"  {k:<16}{val}"
        if baseline and k in baseline:
            b = baseline[k]
            if k.endswith("率"):
                delta = (v - b) * 100
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                good = (delta > 0) if k == "命中率" or k == "可溯源率" else (delta < 0)
                mark = " " if abs(delta) < 0.05 else ("✅" if good else "❌")
                line += f"   基线 {b * 100:.1f}%  {arrow}{abs(delta):.1f}pp {mark}"
            else:
                delta = v - b
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                line += f"   基线 {b:.2f}  {arrow}{abs(delta):.2f}"
        print(line)


def run_eval(runs_path, baseline_path=None):
    cases = {c["id"]: c for c in load_json(TESTSET)["cases"]}
    products = load_json(PRODUCTS)
    alias_index = verify.build_alias_index(products)
    by_name = {p["name"]: p for p in products}

    runs = load_jsonl(runs_path)
    if not runs:
        print(f"没有读到记录：{runs_path}")
        return 1

    rows, missing = [], []
    for run in runs:
        cid = run.get("case_id")
        case = cases.get(cid)
        if not case:
            continue
        rows.append(score_one(case, run, products, alias_index, by_name))

    done = {r["case_id"] for r in rows}
    missing = [cid for cid in cases if cid not in done]

    print_table(rows)
    agg = aggregate(rows)
    base = None
    if baseline_path and os.path.exists(baseline_path):
        brows = []
        for run in load_jsonl(baseline_path):
            case = cases.get(run.get("case_id"))
            if case:
                brows.append(score_one(case, run, products, alias_index, by_name))
        base = aggregate(brows)
    print_summary(agg, base)

    if missing:
        print(f"\n未跑到的用例 {len(missing)} 条：{', '.join(missing)}")
    return 0


def _selftest():
    """用两条假记录验证打分逻辑本身是对的。"""
    products = load_json(PRODUCTS)
    alias_index = verify.build_alias_index(products)
    by_name = {p["name"]: p for p in products}

    hp = next(p for p in products if p["category"] == "笔记本电脑" and p["price"] <= 6000)
    case = {"id": "R01", "type": "recommend",
            "expect": {"category": "笔记本电脑", "price_max": 6000}}

    good = {"case_id": "R01", "answer":
            f"**方案一：{hp['name']}**\n✅优点：¥{hp['price']} 合适",
            "steps": [{"step": 1, "type": "action", "action": "search_products",
                       "action_input": '{"query":"轻薄本"}',
                       "observation": f"检索到 1 件商品：\n1. {hp['name']} | ¥{hp['price']}"},
                      {"step": 2, "type": "final", "content": "..."}],
            "latency_ms": 1000}
    r = score_one(case, good, products, alias_index, by_name)
    assert r["hit"], r
    assert r["hallucinations"] == 0, r
    assert r["degraded"] == 0, r

    bad = {"case_id": "R01", "answer":
           "**方案一：联想小新 Pro 16**\n✅优点：¥5299 很值",
           "steps": [{"step": 1, "type": "action", "action": "search_products",
                      "action_input": "轻薄本",   # 非 JSON → 应被记为降级
                      "observation": f"检索到 1 件商品：\n1. {hp['name']} | ¥{hp['price']}"},
                     {"step": 2, "type": "final", "content": "..."}],
           "latency_ms": 800}
    r2 = score_one(case, bad, products, alias_index, by_name)
    assert not r2["hit"], r2
    assert r2["hallucinations"] >= 1, r2
    assert r2["degraded"] >= 1, r2

    print("selftest 全部通过（2/2）")
    print("\n好的记录：", r)
    print("差的记录：", r2)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="运行记录 jsonl")
    ap.add_argument("--baseline", help="基线 jsonl，用于对比")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(_selftest())
    if not a.runs:
        ap.print_help()
        sys.exit(1)
    sys.exit(run_eval(a.runs, a.baseline))
