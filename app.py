# app.py — Vercel 版本
# 本地运行：export GEMINI_API_KEY="你的真实key" && python app.py
# 部署到 Vercel 后，会自动从环境变量读取 API key

import os
import json
import google.generativeai as genai
from flask import Flask, request, render_template_string, jsonify

# ========== 1. 配置 Gemini ==========
API_KEY = os.environ.get("GEMINI_API_KEY")  # 从环境变量读取
if not API_KEY:
    raise ValueError("❌ 未找到 GEMINI_API_KEY 环境变量，请在 Vercel 上配置。")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

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
あなたは日本語の構文解析アシスタントです。
与えられた1つの日本語の文を、意味的・構文的にまとまりのよい「句」に分け、
それぞれの句に「文の成分ラベル」を付けてください。

出力は必ず JSON のみ、説明文やコードブロックを一切付けずに返してください。
JSON の形式は次の通りです：
[
  { "text": "私は", "role": "主题" },
  { "text": "果物が", "role": "主语" },
  { "text": "好きで、", "role": "谓语" }
]

使えるラベルは以下の8種類に限定してください：
- 主题
- 主语
- 宾语
- 谓语
- 状语
- 定语
- 补语
- 其他
"""

def call_gemini(sentence: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = SYSTEM_PROMPT + "\n\n対象の文：\n" + sentence
    response = model.generate_content(prompt, request_options={"timeout": 30})
    try:
        text = response.text
    except Exception:
        text = str(response)
    return text.strip()

def parse_chunks(raw_text: str):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()
    data = json.loads(cleaned)
    chunks = []
    for item in data:
        text = item.get("text", "")
        role = item.get("role", "其他")
        if not text:
            continue
        if role not in ROLE_COLORS:
            role = "其他"
        chunks.append((text, role))
    return chunks

def build_chunks_html(chunks):
    pieces = []
    for text, role in chunks:
        color = ROLE_COLORS.get(role, ROLE_COLORS["其他"])
        pieces.append(f"""
        <div class="chunk">
          <div class="label">{role}</div>
          <div class="word" style="background-color: {color};">
            {text}
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
.no-result { color: #9ca3af; font-size: 0.9rem; }
{% endraw %}
</style>
</head>
<body>
<div class="container">
  <h1>Gemini 日语句子成分分析</h1>

  <form method="post">
    <textarea name="sentence" placeholder="ここに日本語の文を入力してください">{{ sentence }}</textarea><br>
    <button type="submit">分析</button>
  </form>

  {% if sentence %}
    <div class="sentence-original">原句：{{ sentence }}</div>
    <div class="sentence">{{ chunks_html|safe }}</div>
  {% else %}
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
    error = ""
    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        if sentence:
            try:
                raw = call_gemini(sentence)
                chunks = parse_chunks(raw)
                chunks_html = build_chunks_html(chunks)
            except Exception as e:
                error = f"Error: {e}"
    return render_template_string(
        PAGE_TEMPLATE,
        sentence=sentence,
        chunks_html=chunks_html,
        error=error,
    )

# === 在 Vercel / 本地都能正常启动 ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
