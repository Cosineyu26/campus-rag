from app.prompts import FALLBACK_TEXT, SYSTEM_PROMPT, rewrite_prompt


def test_system_prompt_has_discipline():
    assert "只根据" in SYSTEM_PROMPT and "编造" in SYSTEM_PROMPT
    assert "[" in SYSTEM_PROMPT  # 引用规范标注
    assert "参考来源" in SYSTEM_PROMPT or "参考资料" in SYSTEM_PROMPT


def test_fallback_refuses_generation():
    assert "没有找到" in FALLBACK_TEXT
    assert "建议" in FALLBACK_TEXT


def test_rewrite_prompt_includes_history_and_question():
    p = rewrite_prompt(["上轮问题", "上轮回答"], "现在的问题")
    assert "上轮问题" in p and "上轮回答" in p and "现在的问题" in p
