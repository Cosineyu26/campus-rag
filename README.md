# Campus RAG — 校园官网知识库问答系统

把学校官网（教务处、本科生院、通知公告…）变成**能自然语言问答、带引用来源**的本地知识库。

```
问："本科生最长可以读几年？"
答：本科4年制专业最长学习年限为6年 [1]；5年制专业最长7年 [1]
来源：[1] 宁夏大学本科生学籍管理规定（宁大校发〔2025〕20号）→ 官网原文链接
```

- 全部**本地部署**（数据不出校，不依赖任何外部 API）
- 回答强制**引用来源**，检索不到就**诚实兜底**（不编造）
- 每周自动增量更新（官网改版/新政策自动入库，下架页面自动标记过期）

> 本项目以宁夏大学为例落地，代码与流程对任意高校官网通用（配置栏目 URL 即可）。

## 特性

- **数据管线**：定向爬取官网 → 正文清洗（自动剔除导航/栏目页噪声）→ 条款级分块 → BGE-M3 双路嵌入（稠密+稀疏）→ Qdrant 入库
- **增量同步**：内容哈希比对，只处理新增/变化/消失的页面（不会重复入库、不会误删）
- **检索链**：查询改写（多轮追问）→ 混合检索（向量+关键词 RRF 融合）→ 重排精筛 → 过期政策过滤
- **生成**：Ollama + Qwen3 本地推理（关闭思考模式、8192 上下文）
- **API**：SSE 流式聊天接口 + 会话历史，前端可接任意聊天 UI

## 架构

```
官网 HTML/PDF ──爬取/清洗/分块──▶ Qdrant（469 篇 / 4.4k 块，持续增长）
                                    │ 检索（混合召回+重排）
学生问题 ──▶ FastAPI ──▶ Ollama(Qwen3) ──▶ 带引用回答（SSE 流式）
              │
              └─ MySQL：会话历史 / 文档注册表
```

| 组件 | 选型 | 说明 |
|---|---|---|
| 生成 | Ollama + Qwen3:8b | 本地 GPU，`think:false` + `num_ctx=8192` |
| 嵌入 | BGE-M3（FlagEmbedding） | dense(1024)+sparse 双输出 |
| 重排 | bge-reranker-v2-m3 | 检索精筛 |
| 向量库 | Qdrant | RRF 混合检索 + 状态过滤 |
| 编排 | FastAPI | 自研管线，无 LangChain 依赖 |
| 存储 | MySQL 8 | 会话 + 文档注册表 |

## 三步跑起来

需要：一台 Linux + NVIDIA GPU（16GB 显存即可；纯 CPU 可行但较慢）、Docker、Python 3.11+。

### ① 克隆 + 装数据管线

```bash
git clone https://github.com/Cosineyu26/campus-rag.git && cd campus-rag
docker compose up -d mysql qdrant          # 起 MySQL + Qdrant（先配好 .env 密码，见下）
cd pipeline
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

**配 .env（仓库根目录，必填）**：
```bash
MYSQL_ROOT_PASSWORD=改成强密码
MYSQL_PASSWORD=改成强密码
DB_URL=mysql+pymysql://campus:密码URL编码后@localhost:3306/campus_rag?charset=utf8mb4
```

### ② 配置你的学校官网 + 首次入库

编辑 `pipeline/config.yaml`（参照 `config.nxu.example.yaml`，宁夏大学真实配置可直接用）：

```yaml
sites:
  - name: 教务处规章制度
    category: 教务政策
    entry_urls:            # 栏目入口（列表页），爬虫自动跟进详情页
      - https://你的学校.edu.cn/gltl.htm
    allowed_domain: 你的学校.edu.cn
    max_depth: 2
    delay_seconds: 2.5     # 礼貌爬取：每页间隔

options:
  chunk_size: 500
  chunk_overlap: 60
  # 只收录"文章"URL（排除栏目/列表页骨架）；学校 CMS 形态不同时按需调整
  article_url_pattern: '(/info/|content\.jsp|\.pdf)'
```

先跑嵌入服务（GPU），再执行管线：

```bash
cd embedding_service && docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t campus-rag-embedding . && cd ..
# BGE-M3 模型：hf-mirror.com 下载 BAAI/bge-m3 到 ./models/bge-m3
docker compose up -d embedding
cd pipeline && .venv/bin/python -m pipeline run --config config.yaml
# 期望输出：同步完成: new=N updated=0 stale=0 failed=[]
```

跑两次（第二次应为 `new=0`）即验证增量幂等。配置 cron 可每周自动同步（见 `deploy/crontab.example`）。

### ③ 起问答服务

```bash
# 生成模型（已装 Ollama 则跳过拉取）
ollama pull qwen3:8b

# 重排模型：hf-mirror.com 下载 BAAI/bge-reranker-v2-m3 到 ./models/reranker
docker compose up -d --build
```

**试问**：

```bash
curl -sN -X POST http://localhost:8080/api/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"t1","message":"本科生最长可以读几年？"}'
```

回答带 `[n]` 引用与来源列表即成功。API 一览：

| 接口 | 说明 |
|---|---|
| `POST /api/chat` | SSE 流式问答（`session_id` 相同即多轮） |
| `GET /api/sessions/{id}` | 会话历史 |
| `DELETE /api/sessions/{id}` | 清空会话 |
| `GET /healthz` | 健康检查 |

## 目录结构

```
├── pipeline/           # 数据管线：爬取/清洗/分块/嵌入/入库/增量同步（31 单测）
├── backend/            # 后端：检索链 + 生成编排 + 聊天 API（20 单测）
├── embedding_service/  # BGE-M3 嵌入服务（Docker）
├── reranker_service/   # bge-reranker 重排服务（Docker）
├── docker-compose.yml  # 全栈编排（MySQL/Qdrant/嵌入/重排/Ollama/后端）
├── docs/               # 设计文档与实施计划（含全部技术决策记录）
└── deploy/             # cron 模板等
```

## 设计要点（踩坑记录）

- **BGE-M3 必须用 FlagEmbedding**，sentence-transformers 的模型卡无 sparse 输出（换 4 个版本实测均不行）
- **学校 CMS 是静态站**——纯 HTTP 抓取即可，不需要 Playwright 浏览器（更快更稳）
- **栏目页/列表页会污染知识库**——三层过滤：URL 形态白名单 + 正文精度提取 + 最小长度
- **一次失败的周爬不能误标整个知识库过期**——失败感知的增量标记（stale 可自动复活）
- **qwen3 默认思考模式会把回答写进 reasoning、content 为空**——必须 `think:false`
- Qdrant 点 ID 只收 UUID——确定性映射保证重复入库幂等

## 合规与致谢

- 数据仅来自学校**公开网页**；请在爬取前阅读目标网站 robots.txt 并控制频率
- 本项目的价值来自学校公开的政策文件，请合理使用、勿打包二次分发
- 测试基座：宁夏大学（www.nxu.edu.cn / jxyxbzb.nxu.edu.cn）公开页面

## License

MIT
