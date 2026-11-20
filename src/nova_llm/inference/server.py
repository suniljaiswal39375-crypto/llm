"""OpenAI-compatible FastAPI serving endpoint + Agentic Web Search API & Live Search-Chat UI."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from nova_llm.inference.generator import generate, stream_generate
from nova_llm.inference.sampler import SamplingParams
from nova_llm.models.transformer import NovaLM
from nova_llm.tokenizer.bpe import ByteBPETokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "nova-llm"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=112, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)
    stream: bool = False
    enable_thinking: bool = False


class SearchChatRequest(BaseModel):
    query: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    force_search: bool = False
    stream: bool = False


_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Nova-Search — Agentic Web-Search LLM</title>
  <style>
    :root {
      --bg: #0b0f19;
      --panel: #131a2a;
      --border: #24304a;
      --text: #e6edf7;
      --muted: #8ea0c0;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --green: #22c55e;
      --user-bg: #1e293b;
      --bot-bg: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column;
    }
    header {
      padding: 14px 22px; background: var(--panel); border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); }
    .brand h1 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: 0.2px; }
    .badge {
      font-size: 12px; color: var(--muted); background: #0f172a; border: 1px solid var(--border);
      padding: 4px 10px; border-radius: 999px;
    }
    .controls { display: flex; align-items: center; gap: 14px; font-size: 12px; color: var(--muted); flex-wrap: wrap; }
    .controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
    .controls input[type=range] { width: 76px; accent-color: var(--accent); }
    .suggestions {
      padding: 10px 22px; display: flex; gap: 8px; flex-wrap: wrap; border-bottom: 1px solid var(--border);
      background: #0e1422;
    }
    .chip {
      background: var(--panel); color: var(--text); border: 1px solid var(--border);
      border-radius: 999px; padding: 6px 12px; font-size: 12px; cursor: pointer; transition: 0.15s;
    }
    .chip:hover { border-color: var(--accent); background: #1a233a; }
    main {
      flex: 1; overflow-y: auto; padding: 20px 22px; display: flex; flex-direction: column; gap: 14px;
      max-width: 940px; width: 100%; margin: 0 auto;
    }
    .msg {
      padding: 14px 16px; border-radius: 12px; border: 1px solid var(--border);
      line-height: 1.55; font-size: 14.5px; white-space: pre-wrap; max-width: 90%;
    }
    .msg.user { align-self: flex-end; background: var(--user-bg); border-color: #334155; }
    .msg.assistant { align-self: flex-start; background: var(--bot-bg); width: 100%; max-width: 90%; }
    .msg .role {
      font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
      color: var(--muted); margin-bottom: 6px;
    }
    .search-pill {
      display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #c7d2fe;
      background: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px; padding: 5px 10px;
      margin-bottom: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .sources-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px;
      margin-bottom: 12px;
    }
    .source-card {
      background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px;
      text-decoration: none; color: var(--text); display: block; transition: 0.15s;
    }
    .source-card:hover { border-color: var(--accent); }
    .source-title { font-size: 12.5px; font-weight: 600; margin-bottom: 3px; color: #e2e8f0; }
    .source-meta { font-size: 11px; color: var(--muted); }
    .meta { font-size: 11px; color: var(--muted); margin-top: 8px; }
    footer {
      padding: 14px 22px; background: var(--panel); border-top: 1px solid var(--border);
    }
    form { max-width: 940px; margin: 0 auto; display: flex; gap: 10px; }
    input[type=text] {
      flex: 1; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--border);
      background: var(--bg); color: var(--text); font-size: 14.5px; outline: none;
    }
    input[type=text]:focus { border-color: var(--accent); }
    button {
      padding: 0 20px; border-radius: 10px; border: none; background: var(--accent);
      color: white; font-weight: 600; font-size: 14px; cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .clear-btn { background: transparent; border: 1px solid var(--border); color: var(--muted); padding: 6px 12px; font-size: 12px; }
    .clear-btn:hover { background: #1e293b; color: var(--text); }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="dot"></span>
      <h1>Nova-Search LLM</h1>
      <span id="model-info" class="badge">Loading model…</span>
    </div>
    <div class="controls">
      <label><input id="search-mode" type="checkbox" checked /> Web Search Agent</label>
      <label>Temp <input id="temp" type="range" min="0" max="1" step="0.05" value="0.0" /><span id="temp-val">0.00</span></label>
      <label>Max tokens <input id="max-tok" type="range" min="32" max="192" step="16" value="128" /><span id="max-tok-val">128</span></label>
      <button type="button" class="clear-btn" onclick="clearChat()">Clear</button>
    </div>
  </header>
  <div class="suggestions">
    <button class="chip" onclick="ask('What is DeepSeek-V3?')">DeepSeek-V3?</button>
    <button class="chip" onclick="ask('What is MuonClip?')">MuonClip?</button>
    <button class="chip" onclick="ask('When did Chandrayaan-3 land on the Moon?')">Chandrayaan-3?</button>
    <button class="chip" onclick="ask('What has the James Webb Space Telescope discovered?')">JWST discoveries?</button>
    <button class="chip" onclick="ask('What is new in Python 3.13?')">Python 3.13?</button>
    <button class="chip" onclick="ask('How tall is Mount Everest?')">Mount Everest?</button>
    <button class="chip" onclick="ask('What is HTTP/3 and QUIC?')">HTTP/3 & QUIC?</button>
    <button class="chip" onclick="ask('What is 12 * 12?')">12 * 12 (direct)</button>
  </div>
  <main id="chat">
    <div class="msg assistant">
      <div class="role">Nova-Search</div>
      Hello! I am <strong>Nova-Search</strong>, trained specifically for agentic web searching: I decide when to call <code>web_search</code>, formulate targeted queries, read live search snippets, and synthesize grounded answers with inline citations <code>[1]</code>. Click any prompt above or ask a question!
    </div>
  </main>
  <footer>
    <form id="chat-form">
      <input id="prompt" type="text" placeholder="Ask Nova-Search anything..." autocomplete="off" required />
      <button id="send-btn" type="submit">Search &amp; Ask</button>
    </form>
  </footer>
  <script>
    const chatEl = document.getElementById("chat");
    const form = document.getElementById("chat-form");
    const promptInput = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const searchModeEl = document.getElementById("search-mode");
    const tempEl = document.getElementById("temp");
    const tempVal = document.getElementById("temp-val");
    const maxTokEl = document.getElementById("max-tok");
    const maxTokVal = document.getElementById("max-tok-val");

    tempEl.oninput = () => tempVal.textContent = Number(tempEl.value).toFixed(2);
    maxTokEl.oninput = () => maxTokVal.textContent = maxTokEl.value;

    fetch("/health").then(r => r.json()).then(d => {
      const p = d.params && d.params.total ? (d.params.total / 1e6).toFixed(2) + "M params" : "";
      document.getElementById("model-info").textContent = `${d.model} · ${p} · vocab ${d.vocab_size}`;
    }).catch(() => {});

    function clearChat() { chatEl.innerHTML = ""; }

    function addMessage(role, text) {
      const div = document.createElement("div");
      div.className = "msg " + role;
      const roleEl = document.createElement("div");
      roleEl.className = "role";
      roleEl.textContent = role === "user" ? "You" : "Nova-Search";
      const extras = document.createElement("div");
      const body = document.createElement("div");
      body.className = "body";
      body.textContent = text;
      div.appendChild(roleEl);
      div.appendChild(extras);
      div.appendChild(body);
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
      return { div, extras, body };
    }

    function ask(text) {
      promptInput.value = text;
      form.dispatchEvent(new Event("submit"));
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = promptInput.value.trim();
      if (!text) return;
      promptInput.value = "";
      addMessage("user", text);
      const bot = addMessage("assistant", "");
      sendBtn.disabled = true;

      const t0 = performance.now();
      let chunks = 0;
      try {
        const useSearch = searchModeEl.checked;
        const endpoint = useSearch ? "/v1/search/chat" : "/v1/chat/completions";
        const payload = useSearch
          ? { query: text, temperature: Number(tempEl.value), max_tokens: Number(maxTokEl.value), stream: true }
          : { model: "nova-llm", messages: [{ role: "user", content: text }], temperature: Number(tempEl.value), max_tokens: Number(maxTokEl.value), stream: true };

        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\\n");
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") continue;
            try {
              const obj = JSON.parse(raw);
              if (obj.type === "tool_call") {
                const pill = document.createElement("div");
                pill.className = "search-pill";
                pill.textContent = `🔍 web_search("${obj.query}")`;
                bot.extras.appendChild(pill);
              } else if (obj.type === "sources" && obj.sources?.length) {
                const grid = document.createElement("div");
                grid.className = "sources-grid";
                for (const s of obj.sources) {
                  const a = document.createElement("a");
                  a.className = "source-card";
                  a.href = s.url;
                  a.target = "_blank";
                  a.rel = "noopener noreferrer";
                  a.innerHTML = `<div class="source-title">[${s.id}] ${s.title}</div><div class="source-meta">${s.domain} · ${s.date}</div>`;
                  grid.appendChild(a);
                }
                bot.extras.appendChild(grid);
              } else if (obj.type === "delta" && obj.content) {
                bot.body.textContent += obj.content;
                chunks++;
                chatEl.scrollTop = chatEl.scrollHeight;
              } else if (obj.choices?.[0]?.delta?.content) {
                bot.body.textContent += obj.choices[0].delta.content;
                chunks++;
                chatEl.scrollTop = chatEl.scrollHeight;
              }
            } catch {}
          }
        }
        const dt = (performance.now() - t0) / 1000;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${dt.toFixed(2)}s`;
        bot.div.appendChild(meta);
      } catch (err) {
        bot.body.textContent = "Error: " + err.message;
      } finally {
        sendBtn.disabled = false;
        promptInput.focus();
      }
    });
  </script>
</body>
</html>
"""


def create_app(
    model: NovaLM,
    tokenizer: ByteBPETokenizer,
    model_name: str = "nova-search",
    search_backend: Any | None = None,
) -> FastAPI:
    """Create FastAPI app with live Web-Search Chat UI, `/v1/search/chat`, and OpenAI `/v1/chat/completions`."""
    from nova_llm.agent.search_agent import SearchAgent
    from nova_llm.agent.search_engine import HybridSearchBackend

    app = FastAPI(title="Nova-Search LLM Inference Server", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    model.eval()
    agent = SearchAgent(model, tokenizer, search_backend=search_backend or HybridSearchBackend(), top_k_results=2)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _CHAT_HTML

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": model_name,
            "search_agent": True,
            "params": model.count_parameters(),
            "vocab_size": tokenizer.vocab_size,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "nova-llm"}],
        }

    @app.post("/v1/search/chat")
    def search_chat(req: SearchChatRequest) -> Any:
        """Agentic Web Search endpoint: tool-call query -> search execution -> grounded cited synthesis."""
        if req.stream:
            def event_stream():
                for evt in agent.stream_run(
                    req.query,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    force_search=req.force_search,
                ):
                    yield f"data: {json.dumps(evt)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        res = agent.run(
            req.query,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            force_search=req.force_search,
        )
        return {
            "answer": res.answer,
            "used_search": res.used_search,
            "tool_calls": res.tool_calls,
            "sources": [s.to_dict() for s in res.sources],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Any:
        prompt_text = tokenizer.format_chat(
            [m.model_dump() for m in req.messages],
            add_generation_prompt=True,
            enable_thinking=req.enable_thinking,
        )
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt_text, add_bos=True, allow_special=True)], dtype=torch.long
        )
        stop_ids = [tokenizer.eos_id, tokenizer.special_to_id["<|im_end|>"]]
        params = SamplingParams(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            stop_token_ids=stop_ids,
            stop_strings=["<|im_end|>", "<|eos|>"],
        )
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:
            def event_stream():
                for chunk in stream_generate(model, prompt_ids, tokenizer, params=params):
                    clean = chunk.replace("<|im_end|>", "").replace("<|eos|>", "")
                    if not clean:
                        continue
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": clean}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                done_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        res = generate(model, prompt_ids, params=params, tokenizer=tokenizer, use_cache=True)
        content = (res.text or "").replace("<|im_end|>", "").replace("<|eos|>", "").strip()
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_ids.shape[1]),
                "completion_tokens": int(res.new_token_ids.shape[1]),
                "total_tokens": int(res.token_ids.shape[1]),
            },
        }

    return app
