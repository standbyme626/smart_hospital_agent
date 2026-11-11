#!/usr/bin/env python3
"""把 console 页迁移到三栏 ConsoleShell（对齐演示站）。

只做结构搬运：抽出页头标题/副标题 → 移除页头与旧导航 → main 换成普通容器 → 包一层 ConsoleShell。
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path("app")
ENDS = {"clinician": "clinician", "ops": "ops", "platform": "platform"}
NAV_NAMES = ("ClinicianNav", "OpsNav", "PlatformNav")


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_head(header: str) -> tuple[str, str, bool]:
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", header, re.S)
    paras = re.findall(r"<p[^>]*>(.*?)</p>", header, re.S)
    dynamic = False
    if h1:
        title = clean(h1.group(1))
        subtitle = clean(paras[0]) if paras else ""
    else:
        title = clean(paras[0]) if paras else ""
        subtitle = clean(paras[1]) if len(paras) > 1 else ""
    if "{" in title or "{" in subtitle:
        dynamic = True
    return title, subtitle, dynamic


def migrate(path: pathlib.Path, end: str, dry: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if "ConsoleShell" in text:
        return "skip: already migrated"

    header_match = re.search(r"[ \t]*<header[^>]*>.*?</header>\n", text, re.S)
    header = header_match.group(0) if header_match else ""
    if header:
        title, subtitle, dynamic = extract_head(header)
    else:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.S)
        title, subtitle, dynamic = (clean(h1.group(1)) if h1 else "", "", False)

    if dynamic or not title:
        return "skip: dynamic/missing title -> " + repr(title)

    props = f'end="{end}" title="{title}"' + (f' subtitle="{subtitle}"' if subtitle else "")
    changed = text

    # 1) 去掉页头与旧导航
    if header:
        changed = changed.replace(header, "")
    for name in NAV_NAMES:
        changed = re.sub(r"[ \t]*<div className=\"[^\"]*\">\s*<" + name + r" />\s*</div>\n", "", changed)
        changed = re.sub(r"[ \t]*<" + name + r" />\n", "", changed)
        changed = re.sub(r'import \{ ' + name + r' \} from "@/components/' + name + r'";\n', "", changed)

    # 2) 外层与 main
    if re.search(r'<div className="min-h-dvh bg-canvas">', changed):
        changed = changed.replace('<div className="min-h-dvh bg-canvas">', "<ConsoleShell " + props + ">", 1)
        changed = re.sub(r"\n    </div>\n  \);\n}", "\n    </ConsoleShell>\n  );\n}", changed, count=1)
        changed = re.sub(r'<main[^>]*>', '<div className="space-y-4">', changed, count=1)
        changed = changed.replace("</main>", "</div>", 1)
    else:
        changed = re.sub(r"<main[^>]*>", "<ConsoleShell " + props + ">", changed, count=1)
        changed = changed.replace("</main>", "</ConsoleShell>", 1)

    # 3) import
    changed = changed.replace(
        'import { PageHead } from "@/components/AppFrame";',
        'import { ConsoleShell } from "@/components/ConsoleShell";',
    )
    if 'from "@/components/ConsoleShell"' not in changed:
        changed = re.sub(
            r'(import \{[^}]*\} from "@/components/ui";\n)',
            r'\1import { ConsoleShell } from "@/components/ConsoleShell";\n',
            changed,
            count=1,
        )
    if 'from "@/components/ConsoleShell"' not in changed:
        changed = re.sub(
            r'(import .*? from "lucide-react";\n)',
            r'\1import { ConsoleShell } from "@/components/ConsoleShell";\n',
            changed,
            count=1,
        )

    if changed == text:
        return "no-op"
    if dry:
        return "would migrate: " + title + " / " + subtitle
    path.write_text(changed, encoding="utf-8")
    return "migrated: " + title + " / " + subtitle


def main() -> None:
    dry = "--apply" not in sys.argv
    for end in ENDS:
        for path in sorted((ROOT / end).rglob("page.tsx")):
            print(f"{path}: {migrate(path, end, dry)}")
    print("dry-run" if dry else "applied")


if __name__ == "__main__":
    main()
