# -*- coding: utf-8 -*-
"""
降级留痕（Degradation Trace）
=============================
原来的问题是：所有兜底分支都是静默的。
Action Input 解析失败 → 悄悄换成文本解析；图片识别失败 → 悄悄换成模拟识别；
工具抛异常 → 悄悄返回一句"工具执行错误"。用户看到的是一个正常回答，
但没人知道这一轮到底降级了几次、降在哪一步。

没有留痕，就无法回答「改了提示词之后到底变好没有」——因为你连基线都没有。

这个模块做两件事：
  1. 从 run_agent 返回的 steps 里**反推**降级事件（不需要改 run_agent 内部）
  2. 汇总成一个可以直接显示的摘要

用法：
    from agent_trace import analyze_steps
    d = analyze_steps(steps, image_simulated=image_desc.startswith("(模拟)"))
    if d["n_events"]:
        st.caption(f"⚠️ 本轮降级 {d['n_events']} 次")

自测：
    python agent_trace.py --selftest
"""

import json
import sys

KIND_LABEL = {
    "unknown_tool": "调用了不存在的工具",
    "tool_error": "工具执行异常",
    "json_fallback": "Action Input 非合法 JSON，降级为文本解析",
    "image_fallback": "图片识别失败，降级为模拟识别",
    "empty_search": "检索结果为空",
    "no_final": "达到最大轮次仍未给出最终答案",
    "ask_user": "信息不足，主动追问（正常路径，不计入降级）",
}


def analyze_steps(steps, image_simulated=False, max_rounds=6):
    """
    steps  : run_agent 返回的步骤列表
    返回 {n_events, events, by_kind, rounds, summary}
    """
    events = []
    steps = steps or []

    for s in steps:
        obs = str(s.get("observation") or "")

        if obs.startswith("未知工具"):
            events.append({"kind": "unknown_tool", "step": s.get("step"),
                           "detail": obs[:60]})
        elif obs.startswith("工具执行错误"):
            events.append({"kind": "tool_error", "step": s.get("step"),
                           "detail": obs[:60]})
        elif obs.startswith("检索到 0 件商品"):
            events.append({"kind": "empty_search", "step": s.get("step"),
                           "detail": "该轮检索没有命中任何商品"})
        elif obs.startswith("[ASK_USER]"):
            events.append({"kind": "ask_user", "step": s.get("step"),
                           "detail": obs.replace("[ASK_USER]", "")[:40]})

        # Action Input 是不是合法 JSON —— 不合法就说明走了文本解析的兜底分支
        raw_input = s.get("action_input")
        if s.get("type") == "action" and raw_input:
            try:
                json.loads(raw_input)
            except (json.JSONDecodeError, TypeError):
                events.append({"kind": "json_fallback", "step": s.get("step"),
                               "detail": f"原始输入：{str(raw_input)[:40]}"})

    if image_simulated:
        events.append({"kind": "image_fallback", "step": 0,
                       "detail": "视觉 API 未返回结果"})

    # 最后一轮不是 final，说明 6 轮跑完没给出答案
    if steps and steps[-1].get("type") != "final":
        events.append({"kind": "no_final", "step": len(steps),
                       "detail": f"已用满 {max_rounds} 轮"})

    by_kind = {}
    for e in events:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1

    # ask_user 是设计内的正常路径，不算降级
    n_degraded = sum(v for k, v in by_kind.items() if k != "ask_user")

    parts = [f"{KIND_LABEL.get(k, k)}×{v}" for k, v in by_kind.items()]
    return {
        "n_events": len(events),
        "n_degraded": n_degraded,
        "events": events,
        "by_kind": by_kind,
        "rounds": len(steps),
        "summary": "；".join(parts) if parts else "无降级",
    }


def format_trace(d):
    if not d["n_degraded"]:
        return f"✅ 无降级（{d['rounds']} 轮）"
    return f"⚠️ 降级 {d['n_degraded']} 次（{d['rounds']} 轮）：{d['summary']}"


def _selftest():
    steps = [
        {"step": 1, "type": "action", "action": "search_products",
         "action_input": '{"query": "轻薄本", "max_price": 5000}',
         "observation": "检索到 3 件商品：\n1. 惠普战66 | ¥3999"},
        {"step": 2, "type": "action", "action": "compare_products",
         "action_input": "惠普战66 和 Redmi Book",
         "observation": "## 商品对比\n..."},
        {"step": 3, "type": "action", "action": "fly_to_moon",
         "action_input": '{"x":1}', "observation": "未知工具「fly_to_moon」"},
        {"step": 4, "type": "final", "content": "好的"},
    ]
    d = analyze_steps(steps)
    kinds = d["by_kind"]
    assert kinds.get("json_fallback") == 1, d
    assert kinds.get("unknown_tool") == 1, d
    assert d["n_degraded"] == 2, d
    assert "no_final" not in kinds, d

    # 跑满轮次没有 final
    steps2 = [{"step": i, "type": "action", "action": "search_products",
               "action_input": '{"query":"a"}', "observation": "检索到 1 件商品：x"}
              for i in range(1, 7)]
    d2 = analyze_steps(steps2)
    assert "no_final" in d2["by_kind"], d2

    # 图片降级
    d3 = analyze_steps([{"step": 1, "type": "final", "content": "x"}], image_simulated=True)
    assert d3["by_kind"].get("image_fallback") == 1, d3

    print("selftest 全部通过（3/3）")
    print("示例：", format_trace(d))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
