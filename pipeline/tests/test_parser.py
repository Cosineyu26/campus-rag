from datetime import date
from pathlib import Path

import fitz  # PyMuPDF，仅用于构造 PDF fixture

from pipeline.models import RawPage
from pipeline.parser import clean_text, extract_effective_date, parse, sha256


def make_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    # PyMuPDF 默认 helv 字体不含中文字形（中文会被替换成占位符），须显式指定内置中文字体
    page.insert_text((72, 72), text, fontname="china-s")
    doc.save(path)
    doc.close()


# 接近真实新闻页的结构（导航/页脚/侧栏噪声 + 正文容器），trafilatura 才能正确剥离噪声
HTML = """<html><head><title>学生公寓管理规定</title></head>
<body>
<div id="header">
  <div class="logo">学校官网</div>
  <nav class="top-nav"><ul>
    <li><a href="/">首页</a></li><li><a href="/news/">新闻动态</a></li>
    <li><a href="/notice/">通知公告</a></li><li>导航噪声</li>
    <li><a href="/rules/">规章制度</a></li><li><a href="/service/">办事大厅</a></li>
    <li><a href="/contact/">联系我们</a></li>
  </ul></nav>
</div>
<div class="container">
  <div class="breadcrumb">当前位置：首页 &gt; 通知公告 &gt; 正文</div>
  <div class="main">
    <article class="article">
      <h1 class="article-title">学生公寓管理规定</h1>
      <div class="meta">发布时间：2025-08-20　作者：学生工作处</div>
      <div class="article-content">
        <p>为进一步加强学生公寓管理，维护学生公寓的正常秩序，营造安全、整洁、文明、和谐的学习生活环境，根据学校有关规定，结合我校实际情况，特制定本规定。</p>
        <p>学生公寓实行统一管理，入住学生应当自觉遵守国家法律法规和学校各项规章制度，服从公寓管理人员的管理与安排，爱护公共设施设备，节约用水用电，保持室内外环境卫生整洁。</p>
        <p>公寓实行晚23:00锁门制度。锁门后学生须凭校园卡登记后方可进入，未经批准不得擅自晚归、夜不归宿。严禁在公寓内存放易燃易爆等危险物品，严禁使用违规电器，注意防火防盗，保障人身与财产安全。</p>
        <p>本规定自2025年9月1日起施行。</p>
      </div>
    </article>
    <aside class="sidebar">
      <h3>相关链接</h3>
      <ul><li><a href="/a1">公寓报修</a></li><li>导航噪声</li><li><a href="/a2">作息时间</a></li></ul>
    </aside>
  </div>
</div>
<footer class="site-footer">版权所有 © 2025 学校信息中心</footer>
</body></html>"""


def test_parse_html_extracts_body_and_metadata(tmp_path):
    html = tmp_path / "p.html"
    html.write_text(HTML, encoding="utf-8")
    raw = RawPage(url="https://school.edu.cn/p", category="教务政策", html_path=html, title="")
    docs = parse(raw)
    assert len(docs) == 1
    d = docs[0]
    assert "导航噪声" not in d.text and "版权所有" not in d.text  # 正文提取剥离噪声
    assert "23:00锁门" in d.text
    assert d.effective_date == date(2025, 9, 1)
    assert d.content_hash == sha256(d.text)


def test_parse_pdf(tmp_path):
    html = tmp_path / "p.html"
    html.write_text(HTML, encoding="utf-8")
    pdf = tmp_path / "rule.pdf"
    make_pdf(pdf, "为规范学生选课行为，维护正常教学秩序，特制定学生选课管理办法，"
                  "全体学生应严格按照本办法执行选课程序。本规定自2024年3月1日起执行。")
    raw = RawPage(url="https://school.edu.cn/p", category="教务政策", html_path=html,
                  pdf_files=[("https://school.edu.cn/r.pdf", pdf)])
    docs = parse(raw)
    assert len(docs) == 2
    pdf_doc = docs[1]
    assert pdf_doc.url == "https://school.edu.cn/r.pdf"
    assert pdf_doc.effective_date == date(2024, 3, 1)
    assert "选课管理办法" in pdf_doc.text


def test_parse_skips_empty(tmp_path):
    html = tmp_path / "empty.html"
    html.write_text("<html><body></body></html>", encoding="utf-8")
    raw = RawPage(url="https://school.edu.cn/e", category="教务政策", html_path=html)
    assert parse(raw) == []


def test_parse_skips_short_text(tmp_path):
    """导航残留/近空页面（<MIN_TEXT_LENGTH）不入库——Task 10 实测列表页噪声。"""
    html = tmp_path / "nav.html"
    html.write_text("<html><body><div id=\"header\"><nav>首页 机构设置 教务平台 "
                    "管理系统 教室管理平台 档案管理系统</nav></div></body></html>",
                    encoding="utf-8")
    raw = RawPage(url="https://school.edu.cn/nav", category="教务政策", html_path=html)
    assert parse(raw) == []


def test_clean_text_collapses_blank_lines():
    assert clean_text("第一行\n\n  \n第二行\n") == "第一行\n第二行"
