# app.py — Vercel & 本地通用稳定版

import os
import json
import re
import google.generativeai as genai
from flask import Flask, request, render_template_string

# ========== 1. 配置 Gemini ==========

API_KEY = os.environ.get("GEMINI_API_KEY")
HAS_API_KEY = bool(API_KEY)

if HAS_API_KEY:
    genai.configure(api_key=API_KEY)
    MODEL_NAME = "gemini-2.5-flash"
else:
    MODEL_NAME = None  # 占位，避免导入时报错

# 句子成分颜色
ROLE_COLORS = {
    "主题": "#ffc4c4",
    "主语": "#ffb3ba",
    "宾语": "#fff2a8",
    "谓语": "#b3ffd9",
    "状语": "#d9b3ff",
    "定语": "#c7b3ff",
    "补语": "#b3e6ff",
    "其他": "#eeeeee",
}

SYSTEM_PROMPT = """
あなたは日本語の文構造と翻訳の専門教師です。
与えられた日本語の文を、意味・文法のまとまりに基づいて構造的に分析し、指定の JSON 形式で出力してください。
以下の４つのステップに必ず従ってください。

────────────────────────────
【ステップ１：文の構成要素の分解】
文全体をよく読み、意味や文法の単位（語・句）ごとに区切ります。
それぞれの単位に文中での成分ラベルを付けてください：
主語・述語・目的語・連体修飾（定语）・連用修飾（状语）・補語・独立成分（主题）など。
助詞は必ず他の語から分離し、"助詞" として扱います。

────────────────────────────
【ステップ２：核心フレームと従属節（从句）の識別】
文の主節（核心構造）を特定し、従属節を括弧で示します。
ネストがある場合は次の括弧を使い分けてください：
- 最内層：小括弧 ( )
- その外側：中括弧 [ ]
- さらに外側：大括弧 { }

例：
（日本語を話すこと）は難しいです。
これは[私が(昨日、駅前の店で友達に)買ってもらった]プレゼントです。

従属節は補助的情報として括弧で示すだけで、独立した成分ラベルは付けません。

────────────────────────────
【ステップ３：助詞の機能分析】
文中に出現するすべての助詞について、その機能を簡潔に説明してください。
例：「が」→ 主語を示す、「で」→ 動作場所を示す、「と」→ 条件や引用など。

────────────────────────────
【ステップ４：中文翻訳】
文全体の自然で流暢な中国語訳を生成してください。

────────────────────────────
【出力形式（絶対遵守）】
出力は JSON オブジェクトのみで、説明文やコードブロック（```）は一切付けないこと。

{
  "sentence": "原文",
  "translation_zh": "中文翻译",
  "chunks": [
    { "text": "私は", "role": "主题" },
    { "text": "果物", "role": "主语" },
    { "text": "が", "role": "助詞", "note": "主語を示す" },
    { "text": "好き", "role": "谓语" },
    { "text": "で", "role": "助詞", "note": "理由・接続を示す" },
    { "text": "食べていないと", "role": "其他", "note": "括弧内の従属節部分" },
    { "text": "気がすまないほう", "role": "谓语" },
    { "text": "だ", "role": "助詞", "note": "断定を表す" }
  ]
}

【補足ルール】
1. JSON の前後に説明文やコメントを付けない。
2. 助詞は常に他の語と分離し、"note" に機能を記述。
3. 括弧（( ) / [ ] / { }）は従属節の範囲を示すだけで、"从句" ラベルを出力しない。
4. 中国語訳は自然で簡潔に。
5. もし出力に迷う場合、最低限空の配列を返す。
"""


def call_gemini(sentence: str) -> str:
    """调用 Gemini，返回字符串（应为 JSON 文本）"""
    if not HAS_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set on server.")

    model = genai.GenerativeModel(MODEL_NAME)
    prompt = SYSTEM_PROMPT + "\n\n対象の文：\n" + sentence
    response = model.generate_content(
        prompt,
        request_options={"timeout": 40},
    )

    text = (response.text or "").strip()
    return text


# 日文标签 → 中文标签 映射表
JP_ROLE_MAP = {
    "主語": "主语",
    "述語": "谓语",
    "目的語": "宾语",
    "連体修飾": "定语",
    "連用修飾": "状语",
    "補語": "补语",
    "トピック": "主题",
    "話題": "主题",
    "その他": "其他",
}

def parse_chunks(raw_text: str):
    """
    解析 Gemini 输出：
    - 支持顶层是 { sentence, translation_zh, chunks } 的形式
    - 返回 (chunks, translation_zh)
    """
    cleaned = (raw_text or "").strip()

    if not cleaned:
        raise ValueError("模型返回为空。")

    # 有时会包一层 ```json ... ```，这里剥掉
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()

    # 如果前面还有说明文字，尝试从中间抽出 { ... } 或 [ ... ]
    stripped = cleaned.lstrip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        # 把前面一小段原始输出写进错误信息，方便排查
        preview = cleaned[:80].replace("\n", " ")
        raise ValueError(f"模型输出不是合法 JSON（前80字节）：{preview}") from e

    translation_zh = ""
    chunks_data = []

    # 顶层是对象：{ sentence, translation_zh, chunks: [...] }
    if isinstance(data, dict):
        translation_zh = data.get("translation_zh", "") or ""
        chunks_data = data.get("chunks", []) or []
    # 顶层直接就是列表（兼容老格式）
    elif isinstance(data, list):
        chunks_data = data
    else:
        chunks_data = []

    chunks = []
    for item in chunks_data:
        text = item.get("text", "")
        role = item.get("role", "其他")
        note = item.get("note", "")

        # 把日文标签映射成中文标签
        role = JP_ROLE_MAP.get(role, role)

        if not text:
            continue
        if role not in ROLE_COLORS and role not in ("助詞", "从句"):
            role = "其他"

        chunks.append({
            "text": text,
            "role": role,
            "note": note,
        })

    return chunks, translation_zh



def build_chunks_html(chunks):
    """
    生成句子彩色块 HTML：
    - 助詞：不高亮，显示 tooltip
    - role == "从句"：只加括号，不显示成分标签，用中性颜色
    """
    pieces = []
    for item in chunks:
        text = item.get("text", "")
        role = item.get("role", "其他")
        note = item.get("note", "")

        if not text:
            continue

        display_text = text

        # 从句：在文本外面加日文括号
        if role == "从句":
            display_text = f"（{text}）"

        # 助詞：不高亮，用特殊样式 + 悬浮说明
        if role == "助詞":
            safe_note = note or "助詞"
            pieces.append(f"""
            <div class="chunk">
              <div class="label">&nbsp;</div>
              <div class="word particle">
                {display_text}
                <span class="particle-note">{safe_note}</span>
              </div>
            </div>
            """)
            continue

        # 其他成分：正常彩色圆角块
        # 从句不参与成分标注：标签留空、颜色用“其他”
        if role == "从句":
            label = "&nbsp;"
            color = ROLE_COLORS["其他"]
        else:
            label = role
            color = ROLE_COLORS.get(role, ROLE_COLORS["其他"])

        pieces.append(f"""
        <div class="chunk">
          <div class="label">{label}</div>
          <div class="word" style="background-color: {color};">
            {display_text}
          </div>
        </div>
        """)

    return "".join(pieces)


# ========== Flask 网页部分 ==========

app = Flask(__name__)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Gemini 日语句子成分分析</title>
<style>
{% raw %}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 32px;
    line-height: 1.7;
    background: #f5f5f7;
}
.container {
    max-width: 960px;
    margin: 0 auto;
    background: #f9fafb;
    border-radius: 24px;
    padding: 24px 28px 32px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}
h1 {
    font-size: 1.8rem;
    margin-bottom: 1rem;
}
.info {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.error {
    font-size: 0.9rem;
    color: #b91c1c;
    margin: 0.5rem 0 0.75rem;
}
textarea {
    width: 100%;
    height: 80px;
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid #d4d4d8;
    font-size: 1rem;
    resize: vertical;
}
button {
    margin-top: 8px;
    padding: 6px 18px;
    border-radius: 999px;
    border: none;
    cursor: pointer;
    font-size: 0.95rem;
    background: #111827;
    color: white;
}
button[disabled] {
    opacity: 0.4;
    cursor: not-allowed;
}
button:hover { opacity: 0.9; }
.sentence-original { margin: 1rem 0; font-size: 1.05rem; }
.sentence {
    display: flex; flex-wrap: wrap; gap: 4px 10px;
    align-items: flex-end; margin-bottom: 8px;
}
.chunk { text-align: center; margin-bottom: 12px; }
.label { font-size: 0.8rem; margin-bottom: 3px; color: #555; }
.word {
    padding: 4px 10px; border-radius: 999px;
    display: inline-block; background: #fff;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.06);
}
/* 助詞用のスタイル：背景なし、点線枠 + ホバー説明 */
.word.particle {
    background: transparent;
    border: 1px dashed #d4d4d8;
    position: relative;
}

.word.particle .particle-note {
    display: none;
    position: absolute;
    left: 50%;
    top: 120%;
    transform: translateX(-50%);
    white-space: nowrap;
    background: #111827;
    color: #f9fafb;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    z-index: 10;
}

.word.particle:hover .particle-note {
    display: inline-block;
}

/* 中文翻译折叠区域 */
.translation-box {
    margin-top: 0.75rem;
    font-size: 0.9rem;
}

.translation-box summary {
    cursor: pointer;
    list-style: none;
    outline: none;
}

.translation-box summary::-webkit-details-marker {
    display: none;
}

.translation-box summary::before {
    content: "▶ ";
    font-size: 0.8rem;
}

.translation-box[open] summary::before {
    content: "▼ ";
}

.translation-text {
    margin-top: 0.4rem;
    padding: 0.6rem 0.8rem;
    border-radius: 0.5rem;
    background: #f3f4f6;
    color: #111827;
}

.no-result { color: #9ca3af; font-size: 0.9rem; }
.debug-box {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    background: #f3f4f6;
    border-radius: 0.75rem;
    font-size: 0.8rem;
    color: #4b5563;
    white-space: pre-wrap;
}
.debug-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}
{% endraw %}
</style>
</head>
<body>
<div class="container">
  <h1>Gemini 日语句子成分分析</h1>

  {% if not has_api_key %}
    <div class="error">
      サーバー側の設定エラー：GEMINI_API_KEY が設定されていません。<br>
      （Vercel の Environment Variables に GEMINI_API_KEY を追加してください。）
    </div>
  {% else %}
    <div class="info">
      文を入力して「分析」を押すと、文の成分（主語・述語など）が色分けされて表示されます。
    </div>
  {% endif %}

  <form method="post">
    <textarea name="sentence" placeholder="ここに日本語の文を入力してください">{{ sentence }}</textarea><br>
    <button type="submit" {% if not has_api_key %}disabled{% endif %}>分析</button>
  </form>

  {% if error_msg %}
    <div class="error">{{ error_msg }}</div>
  {% endif %}

  {% if sentence and not error_msg and chunks_html %}
    <div class="sentence-original">原句：{{ sentence }}</div>

    {% if translation_zh %}
      <details class="translation-box">
        <summary>中文翻译を表示</summary>
        <div class="translation-text">{{ translation_zh }}</div>
      </details>
    {% endif %}

    <div class="sentence">{{ chunks_html|safe }}</div>
  {% elif not sentence %}
    <p class="no-result">上に文を入力して「分析」を押してください。</p>
  {% endif %}

</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    sentence = ""
    chunks_html = ""
    error_msg = ""
    debug_text = ""
    translation_zh = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        if sentence and HAS_API_KEY:
            try:
                raw = call_gemini(sentence)
                debug_text = raw or ""
                try:
                    chunks, translation_zh = parse_chunks(raw)
                    chunks_html = build_chunks_html(chunks)
                except Exception as e:
                    error_msg = f"JSON解析エラー: {e}"
            except Exception as e:
                error_msg = f"Gemini 呼び出しエラー: {e}"

    return render_template_string(
        PAGE_TEMPLATE,
        sentence=sentence,
        chunks_html=chunks_html,
        error_msg=error_msg,
        has_api_key=HAS_API_KEY,
        debug_text=debug_text,
        translation_zh=translation_zh,
    )


# Vercel / 本地入口
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
