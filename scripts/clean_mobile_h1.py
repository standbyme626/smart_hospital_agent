import pathlib, re

root = pathlib.Path("frontend/app")
pattern = re.compile(r'[ \t]*<h1 className="text-xl font-semibold text-ink lg:hidden">[^<]*</h1>\n')
removed = 0
for path in root.rglob("page.tsx"):
    text = path.read_text(encoding="utf-8")
    new = pattern.sub("", text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        removed += 1
        print("removed mobile h1:", path)
print("files:", removed)
