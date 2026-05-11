"""
调用 DeepSeek 生成大型真实数码产品数据库
运行：python build_real_products.py
"""
import json, os, re, time
from openai import OpenAI

api_key = os.environ.get("DEEPSEEK_KEY") or input("请输入 DeepSeek API Key: ").strip()

CATEGORIES = {
    "手机": "旗舰、中端和性价比手机，覆盖iPhone、华为、小米、vivo、OPPO、三星、一加、荣耀、realme、iQOO",
    "笔记本电脑": "轻薄本、游戏本、商务本，覆盖MacBook、联想、华为、华硕、惠普、戴尔、微软、机械革命",
    "耳机": "TWS真无线、头戴降噪、HiFi监听，覆盖AirPods、索尼、Bose、森海塞尔、华为、小米、漫步者、JBL",
    "平板": "旗舰平板、学习平板、办公平板，覆盖iPad、华为、小米、三星、联想、OPPO、微软",
    "智能手表": "运动手表、健康手表、商务手表，覆盖Apple Watch、华为、佳明、三星、小米、OPPO、Amazfit",
    "键盘": "机械键盘、静电容键盘、薄膜键盘，覆盖CHERRY、罗技、雷蛇、Keychron、达尔优、VGN、Filco",
    "鼠标": "游戏鼠标、办公鼠标、轨迹球，覆盖罗技、雷蛇、ROG、赛睿、Zowie、微软",
    "显示器": "4K显示器、电竞显示器、设计师显示器，覆盖戴尔、LG、华硕、三星、AOC、明基",
    "音箱": "蓝牙音箱、智能音箱、桌面音箱，覆盖JBL、Bose、哈曼卡顿、苹果、华为、小米、漫步者",
    "相机": "微单相机、运动相机、卡片机，覆盖索尼、佳能、尼康、富士、GoPro、大疆",
    "游戏主机": "家用主机、掌机、VR设备，覆盖索尼PS、任天堂、微软Xbox、Steam Deck、Meta Quest",
    "存储设备": "移动硬盘、U盘、NAS，覆盖西部数据、希捷、三星、闪迪、群晖、联想",
}

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
all_products = []

for cat, desc in CATEGORIES.items():
    print(f"\n{'='*50}")
    print(f"正在生成 {cat} 品类...")

    # 分两批：高端 + 中端/性价比
    for tier, tier_desc in [("高端旗舰", "高端/旗舰级"), ("中端性价比", "中端/性价比级")]:
        prompt = f"""你是数码产品数据库专家。列出8款市面上真实存在的"{cat}"产品，聚焦{tier_desc}。

要求：
1. 产品必须真实存在，型号准确（2024-2025年市售款）
2. price 是2025年中国市场参考价（人民币，数字）
3. features 列出4-5个核心卖点（具体参数，不要泛泛而谈）
4. suitable_for 列出2-3个适合人群
5. description 一句话概述（20字以内）

输出纯JSON数组（不要```标记），格式：
[{{"name":"完整产品名+关键规格","category":"{cat}","price":价格数字,"brand":"品牌","features":["卖点1","卖点2","卖点3","卖点4"],"description":"一句话描述","suitable_for":["人群1","人群2"]}}]

{tier}产品："""

        for retry in range(2):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=4096,
                )
                raw = resp.choices[0].message.content.strip()
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    items = json.loads(m.group())
                    # 去重
                    existing = {p["name"] for p in all_products}
                    new_items = [it for it in items if it["name"] not in existing]
                    print(f"  {tier}: {len(new_items)} 件")
                    all_products.extend(new_items)
                    break
                else:
                    print(f"  {tier}: 未找到JSON (重试{retry+1})")
                    print(f"  Raw: {raw[:200]}")
            except Exception as e:
                print(f"  {tier}: 出错 - {e} (重试{retry+1})")
                time.sleep(2)

    time.sleep(0.5)  # 避免请求过快

print(f"\n{'='*50}")
print(f"总计生成 {len(all_products)} 件真实产品")

# 保存
out_path = "data/products.json"
if os.path.exists(out_path):
    import shutil
    shutil.copy(out_path, "data/products_bak.json")
    print("已备份旧数据")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_products, f, ensure_ascii=False, indent=2)
print(f"已保存到 {out_path}")

# 统计
cats = {}
for p in all_products:
    cats[p["category"]] = cats.get(p["category"], 0) + 1
print("\n品类分布：")
for c, n in sorted(cats.items()):
    print(f"  {c}: {n}件")
