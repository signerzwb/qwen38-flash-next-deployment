import json, os, random, argparse

MATH1 = ("一个水池有甲乙两个进水管和一个排水管。甲管单独注满水池需要 6 小时，乙管单独注满需要 4 小时，"
         "排水管单独排空满池水需要 12 小时。现在三个管子同时打开，水池初始为空；2 小时后关掉排水管，"
         "其余两个进水管继续进水。问：关掉排水管后，还需要多少小时水池才刚好注满？请给出精确分数。")
MATH2 = ("从 1 到 100 的所有整数中随机取出两个不同的数（不考虑顺序）。求这两个数之和为偶数的概率，"
         "答案用最简分数表示。")
CODE = ("写一个 Python 函数 merge_intervals(intervals)，输入是一个闭区间列表 [[start, end], ...]，"
        "这些区间可能乱序、可能互相重叠、可能相邻、也可能重复。要求：\n"
        "1) 返回合并后的区间列表，按起点升序排列；\n"
        "2) 不能修改传入的列表；\n"
        "3) 必须正确处理空列表；\n"
        "4) 复杂度必须是 O(n log n)；\n"
        "5) 只能用标准库。\n"
        "只输出代码，不要任何解释、不要 markdown 代码围栏。")
HALL1 = ("请介绍一下 Qwen3.8-Flash-Next-27B-Mini 这个模型：它的参数量是多少、上下文窗口多大、"
         "什么时候发布的，并给出官方发布链接。")
FMT = ("请输出恰好 3 行，每行格式为「编号. 一个汉字」，编号从 1 开始。"
       "不要输出任何其他内容，不要有前后缀，不要用 markdown。")
NEG = ("用不超过 50 个汉字介绍杭州这座城市。硬性要求：整段话里不能出现「西湖」两个字，"
       "也不能出现任何标点符号（逗号、句号、顿号都不行），不能有换行。")
WEATHER_TOOL = [{"type": "function", "function": {"name": "get_weather",
    "description": "查询某个城市指定日期的天气",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名"},
        "unit": {"type": "string", "enum": ["c", "f"], "description": "温度单位"}},
        "required": ["city", "unit"]}}}]
TOOL_Q = "帮我查一下上海和北京明天的天气，用摄氏度。"
NOVEL = ("写一段约 600 字的中文小说片段。场景：深夜的便利店，一个刚下夜班的护士和一个值夜班的店员。"
         "硬性要求：\n"
         "1) 全文不能出现「疲惫」和「孤独」这两个词，但要让读者感受到这两种状态；\n"
         "2) 必须包含一个只有在这个现场才会存在的具体感官细节；\n"
         "3) 结尾必须停在一个动作上，不要总结、不要抒情、不要升华；\n"
         "4) 第三人称视角。\n"
         "只输出小说正文，不要标题、不要解释。")
SCRIPT = ("用标准的中文影视剧本格式写一场戏，约 400 字。场景：医院走廊，凌晨三点。人物：护士林岚、保安老周。\n"
          "必须包含：场景标题行（含地点、时间、内景/外景）、人物动作描写、至少 5 句对白、一个音效提示。\n"
          "只输出剧本，不要任何解释文字。")
PROMPTQ = ("我要用 AI 绘画工具生成一张图，帮我写提示词。\n"
           "画面想法：雨天傍晚的东京小巷，一个撑着透明雨伞的女孩背影，路边有自动贩卖机，"
           "霓虹灯倒影在积水里。\n"
           "要求：输出必须是英文、单段、逗号分隔的关键词式，不要写成整句；必须包含镜头（例如 35mm）、"
           "光线、色调、画质这几类词；最后另起一行输出一行为负面提示词，以 Negative prompt: 开头。")
VIDEO = ("把下面这段内容拆成一个 6 个镜头的短视频分镜脚本，用于 15 秒短视频。\n"
         "内容：一杯手冲咖啡从磨豆到入口的全过程。\n"
         "要求：用 Markdown 表格输出，表头必须正好是 | 镜头 | 时长 | 画面 | 运镜 | 音效 |；"
         "共 6 行数据；6 个时长加起来必须正好 15 秒；不要任何表格以外的解释文字。")

def build_doc(seed=20260930):
    rnd = random.Random(seed)
    words = ("the quarterly review covered staffing budgets throughput incident reports vendor contracts migration "
             "windows and the residual risk register each section was signed off by the responsible lead before "
             "the next cycle began and the minutes were archived in the shared drive for later audit").split()
    def block(n):
        return ["[%05d] %s." % (i, " ".join(rnd.choice(words) for _ in range(rnd.randint(16, 34)))) for i in range(n)]
    parts = []
    parts += block(420)
    parts.append("\n补充说明：经财务复核，会议编号 KR-4471 的最终结算金额为 837462.19 元。该笔款项已于上一季度归档。\n")
    parts += block(120)
    parts.append("干扰记录：会议编号 KR-4147 的最终结算金额为 837624.19 元；会议编号 KR-4741 的最终结算金额为 873462.19 元。\n")
    parts += block(420)
    parts.append("\n组织架构说明：项目组 A 负责的区域代码是 QX-9。该代码用于所有与该组相关的派工单和巡检记录。\n")
    parts += block(160)
    parts.append("干扰记录：项目组 B 负责的区域代码是 QX-6；项目组 C 负责的区域代码是 XQ-9。\n")
    parts += block(420)
    parts.append("\n质量数据：QX-9 区域在 2026 年第二季度的缺陷率是 0.37%。\n")
    parts += block(120)
    parts.append("干扰记录：QX-6 区域在 2026 年第二季度的缺陷率是 3.7%；QX-9 区域在 2026 年第一季度的缺陷率是 0.73%。\n")
    parts += block(260)
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    doc = build_doc()
    open(os.path.join(a.out, "longdoc.txt"), "w", encoding="utf-8").write(doc)
    tests = []
    def add(i, cat, content, tools=None, max_tokens=3000):
        t = {"id": i, "category": cat, "max_tokens": max_tokens,
             "messages": [{"role": "user", "content": content}]}
        if tools: t["tools"] = tools
        tests.append(t)
    add("m1_pool", "数学", MATH1)
    add("m2_prob", "数学", MATH2)
    add("c1_code", "代码", CODE, max_tokens=2500)
    add("h1_fake_model", "幻觉", HALL1)
    add("f1_fmt3", "指令遵循", FMT, max_tokens=1200)
    add("f2_neg_hz", "指令遵循", NEG, max_tokens=1200)
    add("t1_tool", "工具调用", TOOL_Q, tools=WEATHER_TOOL, max_tokens=1500)
    add("L1_needle", "长文", doc + "\n\n以上是全部文档。问题：文档中编号 KR-4471 的最终结算金额是多少元？只回答数字，不要其他任何内容。")
    add("L2_multihop", "长文", doc + "\n\n以上是全部文档。问题：文档里项目组 A 负责的那个区域，在 2026 年第二季度的缺陷率是多少？只回答百分比，不要其他任何内容。")
    add("w1_novel", "小说", NOVEL)
    add("w2_script", "剧本", SCRIPT)
    add("w3_prompt", "提示词", PROMPTQ)
    add("w4_video", "视频分镜", VIDEO, max_tokens=2500)
    p = os.path.join(a.out, "tests.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for t in tests: f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("wrote %s  tests=%d  doc_chars=%d" % (p, len(tests), len(doc)))

if __name__ == "__main__":
    main()
