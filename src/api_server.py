"""
API 服务 - FastAPI 版

提供：
- GET /: 简单网页演示
- GET /healthz: 健康检查
- GET /status: 查看知识库与运行模式
- POST /configure: 设置商家人设
- POST /ingest: 导入问答对并构建知识库
- POST /query: 提问，返回答案与参考来源

运行：
  uvicorn src.api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.auth import CurrentUser, hash_password, jwt_decode, jwt_encode, verify_password
from src.db import count_users, fetch_all, fetch_one, get_conn, init_db
from src.env_loader import load_dotenv
from src.data_processor import ChatDataProcessor
from src.rag_engine import CustomerServiceRAG


DATA_DIR = Path("./data")
DEFAULT_QA_PATH = DATA_DIR / "qa_pairs.json"
DEFAULT_CHROMA_DIR = str(DATA_DIR / "chroma_db")
TENANTS_DIR = DATA_DIR / "tenants"


class ConfigureRequest(BaseModel):
    merchant_name: str = Field(default="客服", min_length=1, max_length=50)
    personality: str = Field(default="热情专业", min_length=1, max_length=100)


class IngestRequest(BaseModel):
    qa_pairs: Optional[List[Dict[str, Any]]] = None
    qa_json_path: str = str(DEFAULT_QA_PATH)
    persist_directory: str = DEFAULT_CHROMA_DIR
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    mode: str


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=6, max_length=128)
    merchant_slug: Optional[str] = Field(default=None, max_length=40)
    merchant_name: str = Field(default="我的店铺", min_length=1, max_length=50)
    personality: str = Field(default="热情专业", min_length=1, max_length=100)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=6, max_length=128)


class IngestChatRequest(BaseModel):
    """
    从“聊天记录文本”提取问答对并构建知识库。

    chat_text 支持简单格式（每行一条）：
    - 客户: ...
    - 商家: ...
    - 客服: ...
    也支持在上一条消息后续行继续补充内容（会自动拼接）。
    """

    chat_text: str = Field(min_length=1)
    persist_directory: str = DEFAULT_CHROMA_DIR
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\-_.]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:40] or "merchant"


def _load_qa_pairs_from_path(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"未找到问答对文件：{path.as_posix()}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("问答对 JSON 必须是列表")
    return data


def _create_rag(persist_directory: str, overrides: IngestRequest) -> CustomerServiceRAG:
    return CustomerServiceRAG(
        api_key=os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY"),
        persist_directory=persist_directory,
        collection_name="customer_service_kb",
        llm_model=overrides.llm_model or "glm-4",
        embedding_model=overrides.embedding_model or "embedding-3",
        temperature=overrides.temperature if overrides.temperature is not None else 0.7,
        top_k=overrides.top_k if overrides.top_k is not None else 3,
    )


def _create_rag_from_chat_overrides(
    persist_directory: str, overrides: IngestChatRequest
) -> CustomerServiceRAG:
    return CustomerServiceRAG(
        api_key=os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY"),
        persist_directory=persist_directory,
        collection_name="customer_service_kb",
        llm_model=overrides.llm_model or "glm-4",
        embedding_model=overrides.embedding_model or "embedding-3",
        temperature=overrides.temperature if overrides.temperature is not None else 0.7,
        top_k=overrides.top_k if overrides.top_k is not None else 3,
    )


def _parse_chat_text(chat_text: str) -> List[Dict[str, str]]:
    """
    将聊天记录文本解析成 chat_records（role/content）。

    约定：
    - 以“客户/买家/user/u”开头 -> customer
    - 以“商家/店家/客服/seller/assistant”开头 -> merchant
    - 其它行：尝试自动识别角色；否则当作上一条消息的续写
    """

    def normalize_prefix(prefix: str) -> str:
        return prefix.strip().lower()

    customer_prefixes = {"客户", "买家", "user", "u", "customer"}
    merchant_prefixes = {"商家", "店家", "客服", "seller", "assistant", "merchant"}
    customer_name_markers = ("客户", "买家")
    merchant_name_markers = ("客服", "店", "商家", "售后", "官方")

    def _strip_leading_timestamp(text: str) -> str:
        # 兼容：2026-01-01 12:30:00 客户：...
        # 兼容：[12:30] 客服：...
        t = text.strip()
        for pattern in (
            r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*",
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*",
        ):
            t = re.sub(pattern, "", t)
        return t.strip()

    def _guess_role_by_text(text: str) -> Optional[str]:
        """
        在没有明确角色前缀时，用启发式猜测 role。
        目标：宁可返回 None（交给续写/后续逻辑），也不要乱猜。
        """
        t = text.strip()
        if not t:
            return None

        # 问句更像客户
        customer_markers = ("？", "?", "吗", "么", "咋", "怎么", "如何", "能不能", "可以吗", "多少钱", "有货吗")
        if any(m in t for m in customer_markers):
            return "customer"

        # 典型客服话术更像商家/客服
        merchant_markers = (
            "亲",
            "您好",
            "这边",
            "可以的",
            "支持",
            "发货",
            "物流",
            "运费",
            "退换",
            "退款",
            "售后",
            "麻烦您",
            "请您",
            "我们店",
            "咱们",
        )
        if any(m in t for m in merchant_markers):
            return "merchant"

        # 句子很短且像回复语气，也偏 merchant，但不强判
        if len(t) <= 6 and t in {"好的", "可以", "可以的", "收到", "明白", "行", "没问题"}:
            return "merchant"

        return None

    records: List[Dict[str, str]] = []
    for raw_line in chat_text.splitlines():
        line = _strip_leading_timestamp(raw_line)
        if not line:
            continue

        # 支持 "角色: 内容" 或 "角色：内容"（角色既可以是固定前缀，也可以是昵称）
        role = None
        content = None
        for sep in (":", "："):
            if sep in line:
                left, right = line.split(sep, 1)
                maybe = normalize_prefix(left)
                if maybe in customer_prefixes:
                    role = "customer"
                    content = right.strip()
                elif maybe in merchant_prefixes:
                    role = "merchant"
                    content = right.strip()
                else:
                    # 昵称判别：昵称里含“客服/店/商家”等关键词 -> merchant；含“客户/买家” -> customer
                    left_raw = left.strip()
                    if any(m in left_raw for m in merchant_name_markers):
                        role = "merchant"
                        content = right.strip()
                    elif any(m in left_raw for m in customer_name_markers):
                        role = "customer"
                        content = right.strip()
                break

        if role and content:
            records.append({"role": role, "content": content})
            continue

        # 无明确前缀：尝试自动识别
        guessed = _guess_role_by_text(line)
        if guessed:
            records.append({"role": guessed, "content": line})
            continue

        # 仍无法识别：当作续写拼接到上一条；如果没有上一条，默认 customer
        if records:
            records[-1]["content"] = f"{records[-1]['content']} {line}".strip()
        else:
            records.append({"role": "customer", "content": line})

    return records


class _TenantRuntime:
    def __init__(self) -> None:
        self.rag: Optional[CustomerServiceRAG] = None
        self.persist_directory: str = ""
        self.collection_name: str = ""
        self.merchant_name: str = ""
        self.personality: str = ""
        self.mode: str = "offline"
        self.qa_pairs_count: int = 0


tenant_cache: Dict[int, _TenantRuntime] = {}


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


_ENABLE_DOCS = _env_truthy(os.getenv("ENABLE_DOCS"))

app = FastAPI(
    title="AI客服智能体 API",
    version="0.2.0",
    docs_url="/docs" if _ENABLE_DOCS else None,
    redoc_url="/redoc" if _ENABLE_DOCS else None,
    openapi_url="/openapi.json" if _ENABLE_DOCS else None,
)


def _current_user(request: Request) -> CurrentUser:
    authz = request.headers.get("Authorization", "")
    if not authz.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    token = authz.removeprefix("Bearer ").strip()
    try:
        payload = jwt_decode(token)
        return CurrentUser(
            id=int(payload["sub"]),
            email=str(payload.get("email", "")),
            role=str(payload.get("role", "")),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"登录无效：{exc}") from exc


def require_user(user: CurrentUser = Depends(_current_user)) -> CurrentUser:
    return user


def require_admin(user: CurrentUser = Depends(_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


def _tenant_paths(merchant_id: int) -> Dict[str, str]:
    tenant_dir = TENANTS_DIR / str(merchant_id)
    persist_directory = str(tenant_dir / "chroma_db")
    qa_pairs_path = str(tenant_dir / "qa_pairs.json")
    return {
        "tenant_dir": str(tenant_dir),
        "persist_directory": persist_directory,
        "qa_pairs_path": qa_pairs_path,
        "collection_name": f"customer_service_kb_{merchant_id}",
    }


def _load_or_create_runtime(merchant_row: Any) -> _TenantRuntime:
    merchant_id = int(merchant_row["id"])
    runtime = tenant_cache.get(merchant_id)
    if runtime is None:
        runtime = _TenantRuntime()
        tenant_cache[merchant_id] = runtime

    paths = _tenant_paths(merchant_id)
    runtime.persist_directory = paths["persist_directory"]
    runtime.collection_name = paths["collection_name"]
    runtime.merchant_name = str(merchant_row["name"])
    runtime.personality = str(merchant_row["personality"])

    return runtime


_BASE_CSS = """
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.82);
      --text: #1d1a16;
      --muted: #685f57;
      --accent: #b85c38;
      --accent-2: #2f6f60;
      --border: rgba(29,26,22,0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(184,92,56,0.18), transparent 28%),
        radial-gradient(circle at right, rgba(47,111,96,0.16), transparent 24%),
        linear-gradient(135deg, #f7f1e8 0%, #efe4d4 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 18px 48px;
    }
    .hero { margin-bottom: 20px; }
    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 48px);
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 18px;
      backdrop-filter: blur(8px);
      box-shadow: 0 14px 40px rgba(29, 26, 22, 0.08);
    }
    .card h2 { margin: 0 0 12px; font-size: 18px; }
    .status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .pill {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--border);
      font-size: 14px;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 14px;
      color: var(--muted);
    }
    input, textarea, button {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      font: inherit;
    }
    input, textarea {
      padding: 12px 14px;
      background: rgba(255,255,255,0.9);
    }
    textarea { min-height: 120px; resize: vertical; }
    button {
      margin-top: 14px;
      padding: 12px 14px;
      color: white;
      background: linear-gradient(135deg, var(--accent), #d98b52);
      cursor: pointer;
      border: none;
      font-weight: 600;
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #4d8b7d);
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.7;
      background: rgba(255,255,255,0.86);
      border-radius: 16px;
      padding: 14px;
      min-height: 140px;
      border: 1px solid var(--border);
    }
    .sources { margin-top: 12px; padding-left: 18px; color: var(--muted); }
    .tip { margin-top: 12px; font-size: 13px; color: var(--muted); line-height: 1.6; }
    .nav {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .nav a { color: inherit; text-decoration: none; border-bottom: 1px dotted rgba(104,95,87,0.6); }
    .nav a:hover { color: var(--text); border-bottom-color: rgba(29,26,22,0.45); }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
"""


CLIENT_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI客服智能体 - 客户端</title>
  <style>
    __BASE_CSS__
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>AI客服智能体</h1>
      <p>客户端入口：用于提问与查看回答。未配置智谱 API 时会自动切到离线模式。</p>
      <div class="nav">
        <span>入口：</span>
        <a href="./client">客户端</a>
        <span>·</span>
        <a href="./merchant">商户端</a>
      </div>
    </section>
    <div class="grid">
      <section class="card">
        <h2>提问演示</h2>
        <div class="status" id="status"></div>
        <label for="question">客户问题</label>
        <textarea id="question">这件T恤多少钱？</textarea>
        <button id="askBtn">发送问题</button>
        <div class="tip">可直接测试：这件T恤多少钱？ / 不喜欢可以退货吗？ / 这是什么面料？</div>
      </section>
      <section class="card">
        <h2>使用说明</h2>
        <div class="tip">
          你可以直接提问，比如：这件T恤多少钱？/ 不喜欢可以退货吗？/ 这是什么面料？<br/>
          如需修改店铺名称、客服风格或导入知识库，请前往 <a href="/merchant">商户端</a>。
        </div>
      </section>
    </div>
    <section class="card" style="margin-top: 18px;">
      <h2>回答结果</h2>
      <div id="answer" class="answer">等待提问...</div>
      <ul id="sources" class="sources"></ul>
    </section>
  </div>
  <script>
    function basePath() {
      const p = window.location.pathname;
      return p.endsWith('/client') ? p.slice(0, -('/client'.length)) : '';
    }
    async function loadStatus() {
      const response = await fetch(basePath() + '/status');
      const data = await response.json();
      document.getElementById('status').innerHTML = `
        <div class="pill">模式：${data.mode}</div>
        <div class="pill">知识库：${data.has_rag ? '已加载' : '未加载'}</div>
        <div class="pill">问答数量：${data.qa_pairs_count}</div>
        <div class="pill">店铺：${data.merchant_name}</div>
      `;
    }

    async function askQuestion() {
      const question = document.getElementById('question').value.trim();
      if (!question) return;

      const response = await fetch(basePath() + '/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await response.json();
      document.getElementById('answer').textContent = data.answer || data.detail || '请求失败';

      const sources = document.getElementById('sources');
      sources.innerHTML = '';
      (data.sources || []).forEach((item) => {
        const li = document.createElement('li');
        li.textContent = `[${item.category}] ${item.question}`;
        sources.appendChild(li);
      });
      await loadStatus();
    }

    async function updateConfig() {
      const merchant_name = document.getElementById('merchantName').value.trim();
      const personality = document.getElementById('personality').value.trim();

      await fetch(basePath() + '/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ merchant_name, personality })
      });
      await loadStatus();
      document.getElementById('answer').textContent = '配置已更新，可以继续提问。';
    }

    document.getElementById('askBtn').addEventListener('click', askQuestion);
    loadStatus();
  </script>
</body>
</html>
"""


MERCHANT_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI客服智能体 - 商户端</title>
  <style>
    __BASE_CSS__
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>商户端</h1>
      <p>用于配置店铺名称/客服风格，以及（可选）导入问答对构建知识库。</p>
      <div class="nav">
        <span>入口：</span>
        <a href="./client">客户端</a>
        <span>·</span>
        <a href="./merchant">商户端</a>
      </div>
    </section>
    <div class="grid">
      <section class="card">
        <h2>当前状态</h2>
        <div class="status" id="status"></div>
        <div class="tip">提示：本版本使用登录 Token（JWT）。请先在 <a href="/login">登录</a> 后再操作。</div>
      </section>
      <section class="card">
        <h2>商家配置</h2>
        <label for="merchantName">店铺名称</label>
        <input id="merchantName" value="时尚T恤店" />
        <label for="personality">客服风格</label>
        <input id="personality" value="热情活泼" />
        <button id="configBtn" class="secondary">更新配置</button>
      </section>
    </div>

    <section class="card" style="margin-top: 18px;">
      <h2>导入/构建知识库（可选）</h2>
      <label for="qaPath">qa_pairs.json 路径</label>
      <input id="qaPath" value="data/qa_pairs.json" />
      <button id="ingestBtn">从路径导入并构建</button>
      <div class="tip">默认会在服务启动时尝试加载 <code>data/qa_pairs.json</code>。只有你需要换数据时才用这个按钮。</div>
      <div id="ingestResult" class="tip"></div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <h2>粘贴聊天记录 → 一键入库（推荐）</h2>
      <label for="chatText">聊天记录（每行一条，示例：客户：这件T恤多少钱？ / 客服：亲，这款99元…）</label>
      <textarea id="chatText" placeholder="客户：这件T恤多少钱？
客服：亲，这款到手价99元，活动期间还有满减～
客户：不喜欢可以退吗？
客服：支持7天无理由，保持吊牌完整就可以～"></textarea>
      <button id="ingestChatBtn" class="secondary">从聊天记录提取问答并构建</button>
      <div id="ingestChatResult" class="tip"></div>
      <div class="tip">提示：会自动按“客户→商家/客服”的相邻消息配对生成问答；其它内容会被忽略或并入上一条。</div>
    </section>
  </div>
  <script>
    function basePath() {
      const p = window.location.pathname;
      return p.endsWith('/merchant') ? p.slice(0, -('/merchant'.length)) : '';
    }
    function authHeaders() {
      const token = localStorage.getItem('access_token') || '';
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return headers;
    }

    async function loadStatus() {
      const response = await fetch(basePath() + '/status');
      const data = await response.json();
      document.getElementById('status').innerHTML = `
        <div class="pill">模式：${data.mode}</div>
        <div class="pill">知识库：${data.has_rag ? '已加载' : '未加载'}</div>
        <div class="pill">问答数量：${data.qa_pairs_count}</div>
        <div class="pill">店铺：${data.merchant_name}</div>
      `;
      document.getElementById('merchantName').value = data.merchant_name;
      document.getElementById('personality').value = data.personality;
    }

    async function updateConfig() {
      const merchant_name = document.getElementById('merchantName').value.trim();
      const personality = document.getElementById('personality').value.trim();

      const response = await fetch(basePath() + '/configure', {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ merchant_name, personality })
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.detail || '更新失败');
        return;
      }
      await loadStatus();
      alert('配置已更新');
    }

    async function ingestFromPath() {
      const qa_json_path = document.getElementById('qaPath').value.trim();
      const response = await fetch(basePath() + '/ingest', {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ qa_json_path })
      });
      const data = await response.json();
      document.getElementById('ingestResult').textContent = response.ok ? `已导入：${data.qa_pairs} 条，模式：${data.mode}` : (data.detail || '导入失败');
      await loadStatus();
    }

    async function ingestFromChat() {
      const chat_text = document.getElementById('chatText').value.trim();
      if (!chat_text) {
        document.getElementById('ingestChatResult').textContent = '请先粘贴聊天记录';
        return;
      }
      const response = await fetch(basePath() + '/ingest_chat', {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ chat_text })
      });
      const data = await response.json();
      document.getElementById('ingestChatResult').textContent = response.ok
        ? `已从聊天提取并导入：${data.qa_pairs} 条，模式：${data.mode}`
        : (data.detail || '导入失败');
      await loadStatus();
    }

    document.getElementById('configBtn').addEventListener('click', updateConfig);
    document.getElementById('ingestBtn').addEventListener('click', ingestFromPath);
    document.getElementById('ingestChatBtn').addEventListener('click', ingestFromChat);
    loadStatus();
  </script>
</body>
</html>
"""

CLIENT_HTML = CLIENT_HTML.replace("__BASE_CSS__", _BASE_CSS)
MERCHANT_HTML = MERCHANT_HTML.replace("__BASE_CSS__", _BASE_CSS)


def _build_rag_for_merchant(
    merchant_id: int,
    merchant_name: str,
    personality: str,
    qa_pairs: List[Dict[str, Any]],
    overrides: Optional[IngestRequest] = None,
    overrides_chat: Optional[IngestChatRequest] = None,
) -> Dict[str, Any]:
    if not qa_pairs:
        raise ValueError("问答对为空，无法构建知识库")

    paths = _tenant_paths(merchant_id)
    tenant_dir = Path(paths["tenant_dir"])
    tenant_dir.mkdir(parents=True, exist_ok=True)

    # 保存一份可恢复的 QA（离线/重启可用）
    Path(paths["qa_pairs_path"]).write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    if overrides is not None:
        rag = _create_rag(persist_directory=paths["persist_directory"], overrides=overrides)
    elif overrides_chat is not None:
        rag = _create_rag_from_chat_overrides(persist_directory=paths["persist_directory"], overrides=overrides_chat)
    else:
        rag = CustomerServiceRAG(
            api_key=os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY"),
            persist_directory=paths["persist_directory"],
            collection_name=paths["collection_name"],
        )

    rag.collection_name = paths["collection_name"]
    rag.build_knowledge_base(qa_pairs)
    rag.setup_qa_chain(merchant_name=merchant_name, personality=personality)

    runtime = tenant_cache.get(merchant_id) or _TenantRuntime()
    runtime.rag = rag
    runtime.persist_directory = paths["persist_directory"]
    runtime.collection_name = paths["collection_name"]
    runtime.merchant_name = merchant_name
    runtime.personality = personality
    runtime.mode = getattr(rag, "_mode", "offline")
    runtime.qa_pairs_count = len(qa_pairs)
    tenant_cache[merchant_id] = runtime

    return {
        "ok": True,
        "qa_pairs": len(qa_pairs),
        "persist_directory": runtime.persist_directory,
        "mode": runtime.mode,
    }


@app.on_event("startup")
def startup_event() -> None:
    load_dotenv()
    init_db()


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/login")


LOGIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>登录 - AI客服</title>
  <style>__BASE_CSS__</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>登录</h1>
      <p>登录后可进入商家端；客户访问请直接打开商家链接（/m/&lt;slug&gt;/client）。</p>
      <div class="nav"><a href="/register">去注册</a></div>
    </section>
    <section class="card">
      <label>邮箱</label>
      <input id="email" placeholder="you@example.com" />
      <label>密码</label>
      <input id="password" type="password" />
      <button id="loginBtn">登录</button>
      <div id="msg" class="tip"></div>
    </section>
  </div>
  <script>
    function detailText(detail) {
      if (!detail) return '';
      if (typeof detail === 'string') return detail;
      try { return JSON.stringify(detail); } catch { return String(detail); }
    }
    async function login() {
      const email = document.getElementById('email').value.trim();
      const password = document.getElementById('password').value;
      const resp = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById('msg').textContent = detailText(data.detail) || '登录失败';
        return;
      }
      localStorage.setItem('access_token', data.access_token);
      document.getElementById('msg').textContent = '登录成功';
      if (data.role === 'admin') {
        window.location.href = '/admin';
      } else if (data.merchant_slug) {
        window.location.href = '/m/' + data.merchant_slug + '/merchant';
      }
    }
    document.getElementById('loginBtn').addEventListener('click', login);
  </script>
</body>
</html>
"""


REGISTER_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>注册 - AI客服</title>
  <style>__BASE_CSS__</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>注册</h1>
      <p>第一个注册者将成为运营管理员；之后注册默认为商家，需运营审核通过后启用。</p>
      <div class="nav"><a href="/login">去登录</a></div>
    </section>
    <section class="card">
      <label>邮箱</label>
      <input id="email" placeholder="you@example.com" />
      <label>密码</label>
      <input id="password" type="password" />
      <label>商家标识（slug）</label>
      <input id="merchant_slug" placeholder="例如: myshop（仅商家需要）" />
      <label>店铺名称</label>
      <input id="merchant_name" value="我的店铺" />
      <label>客服风格</label>
      <input id="personality" value="热情专业" />
      <button id="regBtn">注册</button>
      <div id="msg" class="tip"></div>
    </section>
  </div>
  <script>
    function detailText(detail) {
      if (!detail) return '';
      if (typeof detail === 'string') return detail;
      try { return JSON.stringify(detail); } catch { return String(detail); }
    }
    async function reg() {
      const payload = {
        email: document.getElementById('email').value.trim(),
        password: document.getElementById('password').value,
        merchant_slug: document.getElementById('merchant_slug').value.trim() || null,
        merchant_name: document.getElementById('merchant_name').value.trim(),
        personality: document.getElementById('personality').value.trim()
      };
      const resp = await fetch('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      document.getElementById('msg').textContent = resp.ok ? '注册成功，请登录' : (detailText(data.detail) || '注册失败');
      if (resp.ok) setTimeout(()=> window.location.href='/login', 700);
    }
    document.getElementById('regBtn').addEventListener('click', reg);
  </script>
</body>
</html>
"""


ADMIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>运营后台 - AI客服</title>
  <style>__BASE_CSS__</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>运营后台</h1>
      <p>审核商家：启用后商家可使用商家端导入与配置。</p>
      <div class="nav"><a href="/login">退出（清 token）</a></div>
    </section>
    <section class="card">
      <button id="refreshBtn" class="secondary">刷新列表</button>
      <div id="list" class="tip"></div>
    </section>
  </div>
  <script>
    function authHeaders() {
      const token = localStorage.getItem('access_token') || '';
      return token ? { 'Authorization': 'Bearer ' + token } : {};
    }
    async function loadMerchants() {
      const resp = await fetch('/admin/api/merchants', { headers: authHeaders() });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById('list').textContent = data.detail || '加载失败';
        return;
      }
      const lines = data.items.map(m => {
        const btn = m.enabled ? '禁用' : '启用';
        return `<div style="margin:10px 0;padding:10px;border:1px solid rgba(29,26,22,0.08);border-radius:14px;">
          <b>${m.slug}</b>（${m.name}） - 状态：${m.enabled ? '已启用' : '待审核'}<br/>
          owner_user_id=${m.owner_user_id} | created_at=${m.created_at}<br/>
          <button onclick="toggle(${m.id}, ${m.enabled ? 0 : 1})" class="secondary" style="margin-top:8px;">${btn}</button>
          <a href="/m/${m.slug}/client" style="margin-left:10px;">打开客户端</a>
          <a href="/m/${m.slug}/merchant" style="margin-left:10px;">打开商家端</a>
        </div>`;
      });
      document.getElementById('list').innerHTML = lines.join('') || '暂无商家';
    }
    async function toggle(id, enabled) {
      const resp = await fetch('/admin/api/merchants/' + id + '/enable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ enabled })
      });
      const data = await resp.json();
      if (!resp.ok) { alert(data.detail || '操作失败'); return; }
      await loadMerchants();
    }
    window.toggle = toggle;
    document.getElementById('refreshBtn').addEventListener('click', loadMerchants);
    loadMerchants();
  </script>
</body>
</html>
"""


LOGIN_HTML = LOGIN_HTML.replace("__BASE_CSS__", _BASE_CSS)
REGISTER_HTML = REGISTER_HTML.replace("__BASE_CSS__", _BASE_CSS)
ADMIN_HTML = ADMIN_HTML.replace("__BASE_CSS__", _BASE_CSS)

@app.get("/login", response_class=HTMLResponse)
def login_page() -> str:
    return LOGIN_HTML


@app.get("/register", response_class=HTMLResponse)
def register_page() -> str:
    return REGISTER_HTML


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> str:
    return ADMIN_HTML


@app.get("/m/{slug}/client", response_class=HTMLResponse)
def tenant_client_page(slug: str) -> str:
    return CLIENT_HTML


@app.get("/m/{slug}/merchant", response_class=HTMLResponse)
def tenant_merchant_page(slug: str) -> str:
    return MERCHANT_HTML


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register")
def auth_register(req: RegisterRequest) -> Dict[str, Any]:
    init_db()
    conn = get_conn()
    try:
        first = count_users(conn) == 0
        role = "admin" if first else "merchant"

        email = req.email.strip().lower()
        if "@" not in email:
            raise HTTPException(status_code=400, detail="邮箱格式不正确")

        pw_hash, pw_salt = hash_password(req.password)
        cur = conn.execute(
            "INSERT INTO users(email, password_hash, password_salt, role) VALUES (?,?,?,?)",
            (email, pw_hash, pw_salt, role),
        )
        user_id = int(cur.lastrowid)

        if role == "merchant":
            slug = _slugify(req.merchant_slug or email.split("@", 1)[0])
            conn.execute(
                "INSERT INTO merchants(slug, name, personality, owner_user_id, enabled) VALUES (?,?,?,?,0)",
                (slug, req.merchant_name, req.personality, user_id),
            )

        conn.commit()
        return {"ok": True, "role": role}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="邮箱或商家标识已存在")
    finally:
        conn.close()


@app.post("/auth/login")
def auth_login(req: LoginRequest) -> Dict[str, Any]:
    conn = get_conn()
    try:
        row = fetch_one(conn, "SELECT * FROM users WHERE email=?", (req.email.strip().lower(),))
        if row is None:
            raise HTTPException(status_code=400, detail="账号或密码错误")
        if not verify_password(req.password, row["password_hash"], row["password_salt"]):
            raise HTTPException(status_code=400, detail="账号或密码错误")

        payload = {"sub": int(row["id"]), "email": row["email"], "role": row["role"]}
        token = jwt_encode(payload)

        merchant_slug = None
        if row["role"] == "merchant":
            m = fetch_one(conn, "SELECT slug FROM merchants WHERE owner_user_id=?", (int(row["id"]),))
            if m:
                merchant_slug = m["slug"]

        return {"access_token": token, "token_type": "bearer", "role": row["role"], "merchant_slug": merchant_slug}
    finally:
        conn.close()


@app.get("/auth/me")
def auth_me(user: CurrentUser = Depends(require_user)) -> Dict[str, Any]:
    return {"id": user.id, "email": user.email, "role": user.role}


@app.get("/admin/api/merchants")
def admin_merchants(_: CurrentUser = Depends(require_admin)) -> Dict[str, Any]:
    conn = get_conn()
    try:
        rows = fetch_all(conn, "SELECT id,slug,name,personality,owner_user_id,enabled,created_at FROM merchants ORDER BY id DESC")
        items = [
            {
                "id": int(r["id"]),
                "slug": r["slug"],
                "name": r["name"],
                "personality": r["personality"],
                "owner_user_id": int(r["owner_user_id"]),
                "enabled": bool(r["enabled"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return {"ok": True, "items": items}
    finally:
        conn.close()


class _EnableMerchantRequest(BaseModel):
    enabled: int = Field(ge=0, le=1)


@app.post("/admin/api/merchants/{merchant_id}/enable")
def admin_enable_merchant(
    merchant_id: int,
    req: _EnableMerchantRequest,
    _: CurrentUser = Depends(require_admin),
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        conn.execute("UPDATE merchants SET enabled=? WHERE id=?", (int(req.enabled), int(merchant_id)))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def _get_merchant_by_slug(conn: Any, slug: str) -> Any:
    row = fetch_one(conn, "SELECT * FROM merchants WHERE slug=?", (slug,))
    if row is None:
        raise HTTPException(status_code=404, detail="商家不存在")
    return row


def _require_merchant_owner_enabled(
    slug: str,
    user: CurrentUser,
    conn: Any,
) -> Any:
    merchant = _get_merchant_by_slug(conn, slug)
    if user.role != "merchant":
        raise HTTPException(status_code=403, detail="需要商家账号")
    if int(merchant["owner_user_id"]) != int(user.id):
        raise HTTPException(status_code=403, detail="无权限操作该商家")
    if not bool(merchant["enabled"]):
        raise HTTPException(status_code=403, detail="商家未通过运营审核")
    return merchant


@app.get("/m/{slug}/status")
def tenant_status(slug: str) -> Dict[str, Any]:
    conn = get_conn()
    try:
        merchant = _get_merchant_by_slug(conn, slug)
        runtime = _load_or_create_runtime(merchant)
        has_rag = runtime.rag is not None
        return {
            "has_rag": has_rag,
            "persist_directory": runtime.persist_directory,
            "merchant_name": runtime.merchant_name,
            "personality": runtime.personality,
            "mode": runtime.mode,
            "qa_pairs_count": runtime.qa_pairs_count,
            "enabled": bool(merchant["enabled"]),
        }
    finally:
        conn.close()


@app.post("/m/{slug}/configure")
def tenant_configure(
    slug: str,
    req: ConfigureRequest,
    user: CurrentUser = Depends(require_user),
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        merchant = _require_merchant_owner_enabled(slug, user, conn)
        conn.execute(
            "UPDATE merchants SET name=?, personality=? WHERE id=?",
            (req.merchant_name, req.personality, int(merchant["id"])),
        )
        conn.commit()

        runtime = _load_or_create_runtime({**dict(merchant), "name": req.merchant_name, "personality": req.personality})
        runtime.merchant_name = req.merchant_name
        runtime.personality = req.personality
        if runtime.rag is not None:
            runtime.rag.setup_qa_chain(merchant_name=runtime.merchant_name, personality=runtime.personality)
            runtime.mode = getattr(runtime.rag, "_mode", runtime.mode)

        return {"ok": True, "merchant_name": runtime.merchant_name, "personality": runtime.personality, "mode": runtime.mode}
    finally:
        conn.close()


@app.post("/m/{slug}/ingest")
def tenant_ingest_from_path(
    slug: str,
    req: IngestRequest,
    user: CurrentUser = Depends(require_user),
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        merchant = _require_merchant_owner_enabled(slug, user, conn)
        qa_pairs = _load_qa_pairs_from_path(req.qa_json_path)
        result = _build_rag_for_merchant(
            merchant_id=int(merchant["id"]),
            merchant_name=str(merchant["name"]),
            personality=str(merchant["personality"]),
            qa_pairs=qa_pairs,
            overrides=req,
        )
        conn.execute(
            "INSERT INTO kb_versions(merchant_id, qa_pairs_count, persist_directory, mode) VALUES (?,?,?,?)",
            (int(merchant["id"]), int(result["qa_pairs"]), result["persist_directory"], result["mode"]),
        )
        conn.commit()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"构建知识库失败：{exc}") from exc
    finally:
        conn.close()


@app.post("/m/{slug}/ingest_chat")
def tenant_ingest_chat(
    slug: str,
    req: IngestChatRequest,
    user: CurrentUser = Depends(require_user),
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        merchant = _require_merchant_owner_enabled(slug, user, conn)
        chat_records = _parse_chat_text(req.chat_text)
        processor = ChatDataProcessor()
        qa_pairs = processor.extract_qa_pairs(chat_records)
        if not qa_pairs:
            raise ValueError("未能从聊天记录中提取到有效问答对（需要至少出现“客户→商家/客服”的回复链）")

        result = _build_rag_for_merchant(
            merchant_id=int(merchant["id"]),
            merchant_name=str(merchant["name"]),
            personality=str(merchant["personality"]),
            qa_pairs=qa_pairs,
            overrides_chat=req,
        )
        conn.execute(
            "INSERT INTO kb_versions(merchant_id, qa_pairs_count, persist_directory, mode) VALUES (?,?,?,?)",
            (int(merchant["id"]), int(result["qa_pairs"]), result["persist_directory"], result["mode"]),
        )
        conn.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"从聊天构建知识库失败：{exc}") from exc
    finally:
        conn.close()


@app.post("/m/{slug}/query", response_model=QueryResponse)
def tenant_query(slug: str, req: QueryRequest) -> QueryResponse:
    conn = get_conn()
    try:
        merchant = _get_merchant_by_slug(conn, slug)
        runtime = _load_or_create_runtime(merchant)

        if runtime.rag is None:
            paths = _tenant_paths(int(merchant["id"]))
            qa_path = Path(paths["qa_pairs_path"])
            if qa_path.exists():
                qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
                _build_rag_for_merchant(
                    merchant_id=int(merchant["id"]),
                    merchant_name=runtime.merchant_name,
                    personality=runtime.personality,
                    qa_pairs=qa_pairs,
                )
                runtime = tenant_cache[int(merchant["id"])]

        if runtime.rag is None:
            raise HTTPException(status_code=400, detail="该商家知识库未初始化，请先在商家端导入聊天记录")

        try:
            result = runtime.rag.query(req.question)
            runtime.mode = result.get("mode", runtime.mode)
            return QueryResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                mode=result.get("mode", runtime.mode),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"查询失败：{exc}") from exc
    finally:
        conn.close()


# Legacy routes: keep for backwards compatibility (redirect)
@app.get("/client")
def legacy_client() -> RedirectResponse:
    return RedirectResponse(url="/login")


@app.get("/merchant")
def legacy_merchant() -> RedirectResponse:
    return RedirectResponse(url="/login")
