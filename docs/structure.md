# Repository Structure Convention

更新时间：2026-02-28

## 1. 目标

本约定用于区分以下六类资产，降低维护成本并减少 Git 噪音：

1. `source`
2. `docs`
3. `scripts`
4. `data`
5. `runtime artifacts`
6. `third_party`

## 2. 分类与目录边界

| 分类 | 目录 | 说明 | 入库策略 |
| --- | --- | --- | --- |
| source | `backend/`, `frontend_new/`, `src/` | 可运行源码与测试 | 允许 |
| docs | `docs/`, `README.md` | 架构、治理、交接文档 | 允许 |
| scripts | `scripts/`, `backend/scripts/` | 运维、评估、发布脚本 | 允许 |
| data | `data/`, `backend/data/` | 可复现数据样本、配置型数据 | 允许（限制体积） |
| runtime artifacts | `logs/`, `cache/`, `reports/current/`, `data_persist/`, `backups_runtime/`, `volumes/` | 运行时生成产物 | 禁止入库 |
| third_party | `third_party/` | 明确纳管第三方代码 | 允许（需来源说明） |

## 3. 命名与放置规则

- 根目录只保留入口级文件：`README.md`、`docker-compose.yml`、`Dockerfile`、关键配置。
- 新增文档统一进入 `docs/`，避免散落在根目录。
- 临时验证脚本与一次性实验脚本不得直接放根目录。
- 第三方仓库若长期保留，迁入 `third_party/<name>/` 并附来源说明。

## 4. 体积与噪音控制

- 运行产物必须被 `.gitignore` 覆盖。
- 模型权重、数据库文件、压缩包默认不入库。
- 需要版本化的大文件应采用外部制品库或对象存储，不直接进 Git。

## 5. 渐进式治理策略

1. 先修冲突与入口文档，保证仓库可读。
2. 先“忽略”再“迁移”：先通过 `.gitignore` 控制新增噪音，再安排分批迁移。
3. 删除高风险目录前必须完成：
   - 来源确认
   - 运行依赖确认
   - 备份与回滚方案
