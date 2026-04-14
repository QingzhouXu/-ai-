# Windows 持续运营（比赛后）

本文档面向：你的 Windows 电脑（无域名，公网 IP + HTTP）。

## 0. 前置提醒（公网访问）

- 若你要“所有人都能访问”，必须确保你的网络环境允许公网访问（通常需要：公网 IP 或端口映射）。\n+- 你当前是 HTTP 明文：不要传敏感信息（可后续再上域名/HTTPS）。\n+- 强烈建议设置 `MERCHANT_TOKEN`，避免任何人都能改商户配置/重建知识库。

## 1. 创建虚拟环境并安装依赖

在项目目录（例如 `C:\\Users\\xxx\\Desktop\\AIX`）打开 PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## 2. 配置环境变量（推荐用 .env）

在项目根目录创建 `.env`（不要提交到仓库），示例：

```ini
PORT=8011
MERCHANT_TOKEN=change_me_to_a_long_random_string
# ENABLE_DOCS=1
# ZHIPUAI_API_KEY=your_key_here
```

说明：
- 不设置 `ENABLE_DOCS` 时：`/docs`、`/openapi.json` 默认不可访问\n+- 设置 `MERCHANT_TOKEN` 后：`/configure` 与 `/ingest` 需要请求头 `X-Merchant-Token`

## 3. 手动启动（最简单）

```powershell
$env:PORT="8011"
.\.venv\Scripts\uvicorn.exe src.api_server:app --host 0.0.0.0 --port 8011
```

访问：
- 客户端：`http://127.0.0.1:8011/client`\n+- 商户端：`http://127.0.0.1:8011/merchant`

若端口冲突，换一个端口即可（例如 8012）。

## 4. 开机自启方案 A：NSSM（推荐，最省事）

NSSM 可以把任意命令注册为 Windows 服务。\n+\n+大致步骤（示例名 `aix-demo`）：\n+\n+1) 下载 NSSM（自行下载并解压到例如 `C:\\nssm`）。\n+2) 以管理员身份打开 PowerShell：\n+\n+```powershell
C:\nssm\nssm.exe install aix-demo
```\n+\n+在弹窗里填写：\n+- **Path**: `C:\\Users\\你的用户名\\Desktop\\AIX\\.venv\\Scripts\\uvicorn.exe`\n+- **Arguments**: `src.api_server:app --host 0.0.0.0 --port 8011`\n+- **Startup directory**: `C:\\Users\\你的用户名\\Desktop\\AIX`\n+\n+然后启动：\n+\n+```powershell
C:\nssm\nssm.exe start aix-demo
```\n+\n+停止/重启：\n+\n+```powershell
C:\nssm\nssm.exe stop aix-demo
C:\nssm\nssm.exe restart aix-demo
```\n+\n+提示：环境变量可放到系统环境变量里，或用 NSSM 的“Environment”配置项添加 `MERCHANT_TOKEN`、`ENABLE_DOCS` 等。\n+
## 5. 开机自启方案 B：任务计划程序（无需额外工具）

思路：创建一个“开机/登录时运行”的任务，执行 `uvicorn`。\n+\n+要点：\n+- 任务使用“最高权限”\n+- “起始于”设置为项目目录\n+- 程序填写 `...\nvenv\Scripts\uvicorn.exe`，参数同上\n+
