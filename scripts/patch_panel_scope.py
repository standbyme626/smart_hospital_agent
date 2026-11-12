import pathlib

p = pathlib.Path("app/routes_panel.py")
t = p.read_text(encoding="utf-8")

old_recent = '''    recent_rows = await db.fetch(
        """
        SELECT model, task, latency_ms, created_at
        FROM model_calls
        WHERE created_at >= now() - interval '7 days'
        ORDER BY created_at DESC
        LIMIT 6
        """
    )'''
new_recent = '''    # 平台端看全租户；其余端按“本账号”过滤，避免把别人的调用显示给患者/医生
    scope = "tenant" if public_end == "platform" else "self"
    actor = None if scope == "tenant" else str(user.get("sub"))

    if actor:
        recent_rows = await db.fetch(
            """
            SELECT model, task, latency_ms, created_at
            FROM model_calls
            WHERE created_at >= now() - interval '7 days' AND user_id = $1
            ORDER BY created_at DESC
            LIMIT 6
            """,
            actor,
        )
    else:
        recent_rows = await db.fetch(
            """
            SELECT model, task, latency_ms, created_at
            FROM model_calls
            WHERE created_at >= now() - interval '7 days'
            ORDER BY created_at DESC
            LIMIT 6
            """
        )'''
assert old_recent in t
t = t.replace(old_recent, new_recent, 1)

old_counts = '''    counts = await db.fetchrow(
        """
        SELECT count(*) AS calls, coalesce(avg(latency_ms), 0) AS avg_ms
        FROM model_calls
        WHERE created_at >= now() - interval '7 days'
        """
    )'''
new_counts = '''    if actor:
        counts = await db.fetchrow(
            """
            SELECT count(*) AS calls, coalesce(avg(latency_ms), 0) AS avg_ms
            FROM model_calls
            WHERE created_at >= now() - interval '7 days' AND user_id = $1
            """,
            actor,
        )
    else:
        counts = await db.fetchrow(
            """
            SELECT count(*) AS calls, coalesce(avg(latency_ms), 0) AS avg_ms
            FROM model_calls
            WHERE created_at >= now() - interval '7 days'
            """
        )'''
assert old_counts in t
t = t.replace(old_counts, new_counts, 1)

old_ret = '''        "recent": recent,'''
new_ret = '''        "recent": recent,
        "scope": scope,'''
assert old_ret in t
t = t.replace(old_ret, new_ret, 1)

p.write_text(t, encoding="utf-8")
print("panel scoped")
