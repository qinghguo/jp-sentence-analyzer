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
学習者が日本語の文の「骨格」と「修飾」をはっきり把握できるように、
与えられた日本語の文を、次の４つのステップで丁寧に分析し、指定の JSON 形式で出力してください。

────────────────────────────
【ステップ１：語・句ごとの成分ラベル付け】
1. まず文全体を読み、意味や文法の単位（語・句）ごとに区切ってください。
2. 区切られたそれぞれの単位に、文中での成分ラベルを１つだけ付けてください。
   成分ラベルは必ず次の中国語ラベルのいずれか１つにします：

   - 主题（文全体の話題・トピック）
   - 主语（述語の主語）
   - 宾语（目的語）
   - 谓语（文の中心となる述語）
   - 状语（時間・場所・原因・程度・方法など、述語を修飾する連用修飾）
   - 定语（名詞を修飾する連体修飾）
   - 补语（主語や述語などを補って意味を完成させる成分）
   - 其他（上のどれにも当てはまらない場合のみ、最後の手段として使用）

3. 可能な限り「其他」は使わないでください。
   他のラベルで説明できる場合は、そちらを優先してください。

4. 助詞は必ず他の語から分離し、１つの単位として扱います。
   助詞には role として「助詞」を使い、成分ラベルとは別枠で扱います。

────────────────────────────
【ステップ２：核心フレームと従属節（从句）の括弧表示】
1. 文の中の「骨格」（主節：主語＋述語、主語＋述語＋目的語など）を意識してください。
2. 従属節・名詞節・関係節など、主節にぶら下がる部分を括弧で示します。

   重要：
   - 括弧は text ではなく、専用フィールド "sentence_with_brackets" の中だけに入れてください。
   - "chunks" の各要素の text には、括弧を含めないでください（純粋に語・句だけ）。

   括弧の使い分け（"sentence_with_brackets" 内で使用）：
   - 最内層の従属節：小括弧 ( )
   - その外側にある従属節：中括弧 [ ]
   - さらに外側のレベル：大括弧 { }

   例１：
   原文: 日本語を話すことは難しいです。
   sentence_with_brackets: (日本語を話すこと)は難しいです。

   この場合、chunks は次のように細かく分解します：
   "chunks": [
     { "text": "日本語", "role": "主语" },
     { "text": "を", "role": "助詞", "note": "賓語を示す" },
     { "text": "話す", "role": "谓语" },
     { "text": "こと", "role": "补语" },
     { "text": "は", "role": "助詞", "note": "主題を示す" },
     { "text": "難しい", "role": "谓语" },
     { "text": "です。", "role": "助詞", "note": "断定を表す" }
   ]

3. 従属節であっても、role は「从句」ではなく、あくまで「主语」「定语」などの成分ラベルを付けてください。
   従属節であることは "sentence_with_brackets" 内の括弧で表現し、
   "chunks" の各要素は通常どおりに細かくラベリングしてください。

────────────────────────────
【ステップ３：助詞の機能分析】
1. 文中に登場する助詞（が・を・に・へ・で・と・は・も・から・まで・より など）は、
   必ず他の語と分けて１つの text として出力してください。
2. 助詞の role は必ず "助詞" にしてください。
3. 各助詞には "note" フィールドを付け、この文における具体的な機能を短く説明します。
   例：
   - 「が」: "note": "主語を示す"
   - 「を」: "note": "賓語を示す"
   - 「で」: "note": "動作が行われる場所を示す"
   - 「と」: "note": "条件を示す"
   - 「は」: "note": "主題・対比を示す"

────────────────────────────
【ステップ４：中文翻译】
1. 文全体の自然で流暢な中国語訳を "translation_zh" に出力してください。
2. あまり直訳的になりすぎず、意味が分かりやすい自然な中国語にしてください。

────────────────────────────
【出力形式（絶対遵守）】
出力は次の JSON オブジェクトのみとし、前後に説明文やコードブロック（```）などを一切付けないでください。

{
  "sentence": "原文",
  "sentence_with_brackets": "括号で従属節を示した文",
  "translation_zh": "中文翻译",
  "chunks": [
    { "text": "私は", "role": "主题" },
    { "text": "果物", "role": "主语" },
    { "text": "が", "role": "助詞", "note": "主語を示す" },
    { "text": "好き", "role": "谓语" },
    { "text": "で", "role": "助詞", "note": "理由・接続を示す" },
    { "text": "食べていないと", "role": "状语", "note": "条件を示す従属節" },
    { "text": "年中", "role": "状语" },
    { "text": "気がすまないほう", "role": "谓语" },
    { "text": "だ。", "role": "助詞", "note": "断定を表す" }
  ]
}

【追加ルール】
- JSON の前後に余分な文字やコメント、説明文を加えないでください。
- "role" には必ず上記の中国語ラベルのいずれかを使い、意味的に最も近いものを選んでください。
- できるだけ「其他」は使わず、主语・宾语・状语・定语などで丁寧に分類してください。
- 助詞は必ず text が助詞だけになり、"role": "助詞", "note": "…" の形にしてください。
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
    "主題": "主题",
    "トピック": "主题",
    "話題": "主题",
    "状語": "状语",
    "修飾語": "定语",
    "その他": "其他",
}


def parse_chunks(raw_text: str):
    """
    解析 Gemini 输出：
    - 顶层是 { sentence, sentence_with_brackets, translation_zh, chunks } 的形式
    - 返回 (chunks, translation_zh, sentence_with_brackets)
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
        preview = cleaned[:80].replace("\n", " ")
        raise ValueError(f"模型输出不是合法 JSON（前80字节）：{preview}") from e

    translation_zh = ""
    sentence_with_brackets = ""
    chunks_data = []

    if isinstance(data, dict):
        translation_zh = data.get("translation_zh", "") or ""
        sentence_with_brackets = data.get("sentence_with_brackets", "") or ""
        chunks_data = data.get("chunks", []) or []
    elif isinstance(data, list):
        chunks_data = data
    else:
        chunks_data = []

    chunks = []
    for item in chunks_data:
        text = item.get("text", "")
        role = item.get("role", "其他")
        note = item.get("note", "")

        role = (role or "").strip()  # 先去掉空格

        # 日文标签 → 中文标签
        role = JP_ROLE_MAP.get(role, role)

        # 助詞 / 助词 统一成 “助词”
        if role in ("助詞", "助词"):
            role = "助词"

        if not text:
            continue

        # 助词不走 ROLE_COLORS
        if role not in ROLE_COLORS and role != "助词":
            role = "其他"

        chunks.append({
            "text": text,
            "role": role,
            "note": note,
        })

    return chunks, translation_zh, sentence_with_brackets






def build_chunks_html(chunks):
    """
    生成句子彩色块 HTML：
    - 助词：不高亮，虚线框 + 悬浮说明（中文）
    - 其他成分：彩色圆角块
    """
    pieces = []
    for item in chunks:
        text = item.get("text", "")
        role = item.get("role", "其他")
        note = item.get("note", "")

        if not text:
            continue

        display_text = text

        # 助词：不高亮，用特殊样式 + 悬浮说明
        if role == "助词":
            safe_note = note or "助词"
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
        color = ROLE_COLORS.get(role, ROLE_COLORS["其他"])
        label = role

        pieces.append(f"""
        <div class="chunk">
          <div class="label">{label}</div>
          <div class="word" style="background-color: {color};">
            {display_text}
          </div>
        </div>
        """)

    return "".join(pieces)

def colorize_bracket_sentence(sentence_with_brackets: str) -> str:
    """
    给括号结构句子添加颜色：
    () 淡黄色，[] 粉色，{} 紫色，主干文字为红色。
    括号本身 + 括号里的文字都一起高亮。
    """
    if not sentence_with_brackets:
        return ""

    html = ""
    color_map = {
        "(": "#fff7cc",  # 小括号：淡黄色
        "[": "#ffd6e7",  # 中括号：粉色
        "{": "#e6ccff",  # 大括号：淡紫色
    }

    stack = []
    current_color = None  # 当前 span 的背景色（None 表示主句红色或普通）

    def open_span(new_color):
        nonlocal html, current_color
        # 关闭旧 span
        if current_color is not None:
            html += "</span>"
        # 打开新 span（如果有颜色）
        if new_color:
            html += f'<span style="background-color:{new_color};">'
        current_color = new_color

    def add_main_clause_char(ch):
        """不在任何括号里的文字：主句，用红色字体"""
        nonlocal html, current_color
        # 确保没有背景色 span 开着
        if current_color is not None:
            html += "</span>"
            current_color = None
        html += f'<span style="color:#b91c1c;">{ch}</span>'

    for ch in sentence_with_brackets:
        if ch in "([{":
            # 进入更深一层括号
            stack.append(color_map[ch])
            new_color = stack[-1]
            if new_color != current_color:
                open_span(new_color)
            html += ch
        elif ch in ")]}":
            if stack:
                # 括号内文字的颜色 = 当前栈顶颜色
                new_color = stack[-1]
                if new_color != current_color:
                    open_span(new_color)
                html += ch
                # 弹出这一层括号
                stack.pop()
                # 恢复上一层颜色或主句
                new_color = stack[-1] if stack else None
                if new_color != current_color:
                    open_span(new_color)
            else:
                # 理论上不会，但以防万一：按主句处理
                add_main_clause_char(ch)
        else:
            if stack:
                # 在某种括号内部：背景色 = 栈顶颜色
                new_color = stack[-1]
                if new_color != current_color:
                    open_span(new_color)
                html += ch
            else:
                # 不在任何括号里：主句文字
                add_main_clause_char(ch)

    # 最后收尾
    if current_color is not None:
        html += "</span>"

    return html



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
.sentence-brackets {
    margin: 0.25rem 0 0.75rem;
    font-size: 0.95rem;
    color: #374151;
}
.sentence-brackets span {
    line-height: 1.8;
    border-radius: 4px;
    padding: 0;
}
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
    {% if sentence_with_brackets %}
      <div class="sentence-brackets">
        括号结构：<span style="font-family: 'Noto Sans JP', sans-serif;">{{ colored_brackets_html|safe }}</span>
      </div>
    {% endif %}


    
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
    sentence_with_brackets = ""
    sentence = ""
    chunks_html = ""
    error_msg = ""
    debug_text = ""
    translation_zh = ""
    colored_brackets_html = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        if sentence and HAS_API_KEY:
            try:
                raw = call_gemini(sentence)
                debug_text = raw or ""
                try:
                    chunks, translation_zh, sentence_with_brackets = parse_chunks(raw)
                    chunks_html = build_chunks_html(chunks)
                    colored_brackets_html = colorize_bracket_sentence(sentence_with_brackets)
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
        sentence_with_brackets=sentence_with_brackets,
        colored_brackets_html=colored_brackets_html,
    )


# Vercel / 本地入口
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
