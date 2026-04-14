## 免费部署 PlanA（有持久化磁盘/卷）

你的项目是 **FastAPI + SQLite +（可选）Chroma 向量库**。要做到“像官网一样稳定”，核心要求是：

- **进程能长期运行**（不是一次性脚本）
- **有持久化存储**（否则 `data/app.db`、`data/tenants/*`、向量库会在重启后丢失）

本方案只讲 **PlanA：选择有持久卷/磁盘的部署方式**。

---

## 1. 必须设置的环境变量

- **JWT_SECRET**：商家登录 JWT 的签名密钥（必须）
- **PORT**：监听端口（平台会指定时按平台为准）
- （可选）**ZHIPUAI_API_KEY**：智谱 Key（不配也能跑，走离线模式）
- （可选）**ENABLE_DOCS=1**：开启 `/docs`（不建议公网开启）

示例：

```ini
JWT_SECRET=change_me_to_a_long_random_string
PORT=8011
ZHIPUAI_API_KEY=your_key_here
```

---

## 2. 持久化目录（非常关键）

你的数据默认落在项目根目录的 `./data/`：

- `data/app.db`：用户/商家/审核状态（SQLite）
- `data/tenants/<merchant_id>/qa_pairs.json`：商家导入后保存的问答对
- `data/tenants/<merchant_id>/chroma_db/`：在线模式下的向量库持久化目录（如果启用在线 embedding）

因此部署时需要：

- 让 **`./data`** 挂载到持久卷（Persistent Volume / Disk / Mount）

---

## 3. 低成本“真公网”建议（几乎零成本）

### 方案 A（推荐，最稳）：云厂商“学生/免费套餐”小主机 + systemd

优点：
- 真正常驻、持久化可控、跟“官网”最像

部署参考你已有文档：
- `deploy/README_LINUX.md`

额外建议：
- 把 `JWT_SECRET`、`ZHIPUAI_API_KEY` 写到 `/etc/aix-demo.env`
- 用 systemd 方式自启（你已有 `deploy/systemd/aix-demo.service` 示例）

### 方案 B（平台托管 + 持久卷）

不同平台名字不一样，但你要找的能力关键词通常是：
- “Web service / Always on / Long running”
- “Persistent disk / Volume / Mount”
- “Environment variables”

只要平台能满足上面三点，就能按以下通用步骤部署：

1) 构建镜像/或直接运行命令（示例）：

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port ${PORT}
```

2) 持久化挂载：
- 把平台的持久卷挂到应用目录的 `data/`

3) 设置环境变量（至少 `JWT_SECRET`）

---

## 4. 反向代理（可选，但建议）

你可以让应用内部跑在 `8011`，对外提供 `80/443`：

- 对外：`https://your-domain.com/`
- 内部：`http://127.0.0.1:8011/`

常见做法：
- Nginx / Caddy 反向代理 + 自动 HTTPS

---

## 5. 上线后的最小验收

1) 打开 `/register` 注册第一个账号 → 应成为 admin
2) `/admin` 能看到商家列表，能启用商家
3) 商家登录后访问：`/m/<slug>/merchant` 导入聊天记录
4) 客户访问：`/m/<slug>/client` 提问，回答正常
5) 重启服务后：商家仍存在、审核状态不丢、知识库仍可用（PlanA 的关键）

