# Linux 开发板部署（比赛期）

本文档面向：Linux 开发板（无域名，公网 IP + HTTP）。

## 1. 准备目录

建议把项目放到 `/opt/aix`（与 `systemd` 示例保持一致）：

```bash
sudo mkdir -p /opt/aix
sudo rsync -a --delete ./ /opt/aix/
cd /opt/aix
```

## 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3. 配置环境变量

复制环境变量文件并编辑：

```bash
sudo cp deploy/systemd/aix-demo.env.example /etc/aix-demo.env
sudo nano /etc/aix-demo.env
```

至少设置：
- `PORT=8011`（或你想用的端口）

强烈建议（公网环境）：
- 设置 `MERCHANT_TOKEN=...`（保护 `/configure` 与 `/ingest`）

## 4. 安装为 systemd 服务

```bash
sudo cp deploy/systemd/aix-demo.service /etc/systemd/system/aix-demo.service
sudo systemctl daemon-reload
sudo systemctl enable --now aix-demo
```

查看状态与日志：

```bash
systemctl status aix-demo --no-pager
journalctl -u aix-demo -f
```

## 5. 开放端口（按你的系统选择）

如果有 `ufw`：

```bash
sudo ufw allow 8011/tcp
```

如果用 `iptables`（示例）：

```bash
sudo iptables -I INPUT -p tcp --dport 8011 -j ACCEPT
```

## 6. 验证

在开发板本机：

```bash
curl http://127.0.0.1:8011/healthz
```

外网访问（用公网 IP 替换）：
- 客户端：`http://<公网IP>:8011/client`
- 商户端：`http://<公网IP>:8011/merchant`

## 7. 重要说明（生产建议）

- `ENABLE_DOCS` 默认关闭：不开 `ENABLE_DOCS=1` 时，`/docs` 与 `/openapi.json` 不可访问。\n+- 若设置了 `MERCHANT_TOKEN`：商户端页面里填写 token 后，页面会自动在请求头带 `X-Merchant-Token`。\n+- 你当前是 HTTP 明文：公网部署时不要传敏感信息（后续可再加 HTTPS/域名/反代）。

