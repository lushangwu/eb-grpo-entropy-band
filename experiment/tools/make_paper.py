"""解析 论文源.md 里的 [[key]] 引注：按首次出现顺序编号，并从 arXiv 元数据生成 GB/T 7714 文献表。

为什么不让模型手写文献条目：作者名单、题名、年份一律从 export.arxiv.org 的 API
返回值生成（快照存 ../调研/arxiv_meta_20261025.json），避免凭记忆补全。
非 arXiv 来源（Kimi K3.1 Pro 官方公告）在 EXTRA_REFS 里手工著录，并标注为公告级。

用法（在课程论文作业目录下）：
    python experiment/tools/make_paper.py
产出：
    大语言模型后训练强化学习中的熵稳定机制分析与算法设计.md   （引注已换成 [n]）
    调研/参考文献查验链接.md                                   （逐条出处与实读字段）
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE, "论文源.md")
DST = os.path.join(BASE, "大语言模型后训练强化学习中的熵稳定机制分析与算法设计.md")
META = os.path.join(BASE, "调研", "arxiv_meta_20261025.json")
AUDIT = os.path.join(BASE, "调研", "参考文献查验链接.md")

# 机构作者（按 arXiv 返回值的小写形式匹配）
ORG_AUTHORS = {"deepseek-ai", "kimi team", "moonshot ai", "mistral ai", "meta ai"}

# 引注键 -> arXiv id
KEY2ID = {
    "ppo": "1707.06347",
    "rlhf": "1706.03741",
    "instructgpt": "2203.02155",
    "cai": "2212.08073",
    "dpo": "2305.18290",
    "prm": "2305.20050",
    "grpo": "2402.03300",
    "tulu3": "2411.15124",
    "r1": "2501.12948",
    "k15": "2501.12599",
    "dapo": "2503.14476",
    "drgrpo": "2503.20783",
    "vapo": "2504.05118",
    "entropy": "2505.22617",
    "hightoken": "2506.01939",
    "leash": "2507.14843",
    "k2": "2507.20534",
    "agentssurvey": "2509.02547",
    "lrmsurvey": "2509.08827",
    "qae": "2509.22611",
    "v32": "2512.02556",
    "v4": "2606.19348",
    "k3": "2607.24653",
}

# 非 arXiv 来源：手工著录（公告级，无独立技术报告，引用时如实说明）
EXTRA_REFS = {
    "k31pro": "MOONSHOT AI. Kimi K3.1 Pro 发布公告[EB/OL]. (2026-10-24)[2026-10-25]. "
              "https://platform.kimi.ai/docs/release-notes.",
}
# 官方公告里实际读到的字段，写进查验文档
EXTRA_NOTE = {
    "k31pro": "公告明确其为 K3 的后训练版本、同底座 3T 级、大幅增加 RL 算力；"
              "KCB 3.0 长程 55.95%→61.2%，事实错误率 23.35%→6.85%，Terminal-Bench 4.0 65.7%。",
}
# 已发表版本补记（顺序编码制下写在条目末尾，不进上标）
PUBLISHED_NOTE = {
    "r1": "另见 Nature, 2025, 645: 633-638.",
    "dapo": "NeurIPS 2025.",
    "hightoken": "NeurIPS 2025.",
    "agentssurvey": "TMLR.",
    "instructgpt": "NeurIPS 2022.",
    "rlhf": "NeurIPS 2017.",
    "dpo": "NeurIPS 2023.",
}


def surname_initial(name: str) -> str:
    """'Zhihong Shao' -> 'SHAO Z'；机构作者原样保留。"""
    parts = name.strip().split()
    if len(parts) == 1 or not parts[0][0].isalpha():
        return name.strip()
    if len(parts) > 2 and parts[-1].upper() in {"JR", "SR", "II", "III"}:
        parts = parts[:-1]
    first, last = parts[0], parts[-1]
    return f"{last.upper()} {first[0].upper()}"


def authors_gbtd(names: list[str]) -> str:
    names = [n.strip() for n in names if n and n[0].isalpha()]
    if not names:
        return "ANONYMOUS"
    # 机构作者必须显式判定：arXiv 的个人作者名本身就是"名 姓"两段，
    # 用"含空格"当机构特征会把每个人都判成机构。
    if names[0].lower() in ORG_AUTHORS:
        return names[0]
    initials = [surname_initial(n) for n in names]
    if len(initials) > 3:
        return ", ".join(initials[:3]) + ", et al"
    return ", ".join(initials)


def build_entry(key: str, meta: dict) -> tuple[str, str]:
    """返回 (GB/T 7714 条目, 查验说明)。"""
    if key in EXTRA_REFS:
        return EXTRA_REFS[key], EXTRA_NOTE[key]
    aid = KEY2ID[key]
    m = meta[aid]
    year = m["published"][:4]
    title = m["title"].rstrip(".")
    entry = (f"{authors_gbtd(m['authors'])}. {title}[J/OL]. arXiv:{aid}, {year}"
             f"[2026-10-25]. https://arxiv.org/abs/{aid}.")
    if key in PUBLISHED_NOTE:
        entry = entry[:-1] + f". {PUBLISHED_NOTE[key]}"
    note = f"第一作者 {m['authors'][0]}，共 {len(m['authors'])} 位作者，提交日期 {m['published']}，题名按 API 返回值逐字录入。"
    return entry, note


def main() -> int:
    if not os.path.exists(META):
        raise SystemExit(f"缺少 arXiv 元数据快照：{META}")
    meta = json.load(io.open(META, encoding="utf-8"))
    text = io.open(SRC, encoding="utf-8").read()

    keys: list[str] = []
    for m in re.finditer(r"\[\[([a-z0-9]+)\]\]", text):
        if m.group(1) not in keys:
            keys.append(m.group(1))
    unknown = [k for k in keys if k not in KEY2ID and k not in EXTRA_REFS]
    if unknown:
        raise SystemExit("未登记的引注键：" + ", ".join(unknown))
    order = {k: i + 1 for i, k in enumerate(keys)}

    text = re.sub(r"\[\[([a-z0-9]+)\]\]", lambda m: f"[{order[m.group(1)]}]", text)

    entries, audit = [], ["# 参考文献查验记录", "",
                          "逐条按 arXiv API（export.arxiv.org）返回值著录，快照见同目录 "
                          "`arxiv_meta_20261025.json`；非 arXiv 来源单独标注证据级别。", ""]
    for k in keys:
        entry, note = build_entry(k, meta)
        entries.append((order[k], k, entry))
        src = "官方公告（无独立技术报告）" if k in EXTRA_REFS else f"https://arxiv.org/abs/{KEY2ID[k]}"
        audit.append(f"[{order[k]}] {k} — 状态：已坐实")
        audit.append(f"    - 来源：{src}")
        audit.append(f"    - 实读：{note}")
        audit.append("")

    body = text.rstrip() + "\n\n## 参考文献\n\n"
    for num, _, entry in entries:
        body += f"[{num}] {entry}\n\n"
    io.open(DST, "w", encoding="utf-8").write(body)
    io.open(AUDIT, "w", encoding="utf-8").write("\n".join(audit) + "\n")

    print("引注顺序：", ", ".join(f"{order[k]}={k}" for k in keys))
    print("条目数：", len(entries))
    print("成品 md：", DST)
    print("查验记录：", AUDIT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
