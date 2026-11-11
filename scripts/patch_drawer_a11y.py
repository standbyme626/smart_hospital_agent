import pathlib
p = pathlib.Path("components/AppFrame.tsx")
t = p.read_text(encoding="utf-8")

anchor = "  const groups = NAV[end];"
assert anchor in t
t = t.replace(
    anchor,
    '''  useEffect(() => {
    const open = siderOpen || agentOpen;
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [siderOpen, agentOpen]);

  const groups = NAV[end];''',
    1,
)
t = t.replace('          aria-expanded={siderOpen}', '          aria-expanded={siderOpen}\n          aria-controls="app-sider"', 1)
t = t.replace('<aside className={"app-sider" + (siderOpen ? " open" : "")} aria-label="主导航">',
              '<aside id="app-sider" className={"app-sider" + (siderOpen ? " open" : "")} aria-label="主导航">', 1)
p.write_text(t, encoding="utf-8")
print("patched")
