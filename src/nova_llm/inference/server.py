"""OpenAI-compatible FastAPI serving endpoint + Agentic Web Search API & Live Search-Chat UI.

Runs seamlessly on EITHER native `NovaLM` checkpoints OR Hugging Face / Unsloth models
(e.g. Llama 3.2 3B Instruct 4-bit + Google Drive LoRA adapters) via `ChatGeneratorBackend`.
"""

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
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)
    stream: bool = False
    enable_thinking: bool = False


class SearchChatRequest(BaseModel):
    query: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    force_search: bool = False
    stream: bool = False


_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Nova-Search — Unified LLM + Live Web Search</title>
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
    .think-box {
      font-size: 12.5px; color: #cbd5e1; background: #0f172a; border: 1px dashed #334155;
      border-radius: 8px; padding: 8px 10px; margin-bottom: 10px;
    }
    .think-box summary { cursor: pointer; color: #94a3b8; font-weight: 600; }
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
      <h1>Nova-Search</h1>
      <span id="model-info" class="badge">Loading model…</span>
    </div>
    <div class="controls">
      <label><input id="search-mode" type="checkbox" checked /> Web Search Agent</label>
      <label><input id="force-search" type="checkbox" /> Always Search</label>
      <label>Temp <input id="temp" type="range" min="0" max="1" step="0.05" value="0.2" /><span id="temp-val">0.20</span></label>
      <label>Max tokens <input id="max-tok" type="range" min="64" max="512" step="32" value="256" /><span id="max-tok-val">256</span></label>
      <button type="button" class="clear-btn" onclick="clearChat()">Clear</button>
    </div>
  </header>
  <main id="chat">
    <div class="msg assistant">
      <div class="role">Nova-Search</div>
      Ask any question! In <strong>Web Search Agent</strong> mode, the model decides when to search the web, inspects live results, and answers with inline citations <code>[1]</code>, <code>[2]</code>.
    </div>
  </main>
  <footer>
    <form id="chat-form">
      <input id="prompt" type="text" placeholder="Ask anything — code, science, current events, math..." autocomplete="off" required />
      <button id="send-btn" type="submit">Ask</button>
    </form>
  </footer>
  <script>
    const chatEl = document.getElementById("chat");
    const form = document.getElementById("chat-form");
    const promptInput = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const searchModeEl = document.getElementById("search-mode");
    const forceSearchEl = document.getElementById("force-search");
    const tempEl = document.getElementById("temp");
    const tempVal = document.getElementById("temp-val");
    const maxTokEl = document.getElementById("max-tok");
    const maxTokVal = document.getElementById("max-tok-val");

    tempEl.oninput = () => tempVal.textContent = Number(tempEl.value).toFixed(2);
    maxTokEl.oninput = () => maxTokVal.textContent = maxTokEl.value;

    fetch("/health").then(r => r.json()).then(d => {
      const p = d.params && d.params.total ? (d.params.total / 1e6).toFixed(1) + "M params" : "";
      document.getElementById("model-info").textContent = `${d.model} · ${p}`;
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

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = promptInput.value.trim();
      if (!text) return;
      promptInput.value = "";
      addMessage("user", text);
      const bot = addMessage("assistant", "");
      sendBtn.disabled = true;

      const t0 = performance.now();
      try {
        const useSearch = searchModeEl.checked;
        const endpoint = useSearch ? "/v1/search/chat" : "/v1/chat/completions";
        const payload = useSearch
          ? { query: text, temperature: Number(tempEl.value), max_tokens: Number(maxTokEl.value), force_search: forceSearchEl.checked, stream: true }
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
              if (obj.type === "thinking" && obj.content) {
                const det = document.createElement("details");
                det.className = "think-box";
                det.innerHTML = `<summary>💭 Reasoning trace</summary><div></div>`;
                det.querySelector("div").textContent = obj.content;
                bot.extras.appendChild(det);
              } else if (obj.type === "tool_call") {
                const pill = document.createElement("div");
                pill.className = "search-pill";
                pill.textContent = `🔍 ${obj.name || "web_search"}("${obj.query}")`;
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
                chatEl.scrollTop = chatEl.scrollHeight;
              } else if (obj.choices?.[0]?.delta?.content) {
                bot.body.textContent += obj.choices[0].delta.content;
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
    model: NovaLM | None = None,
    tokenizer: ByteBPETokenizer | None = None,
    model_name: str = "nova-search",
    search_backend: Any | None = None,
    backend: Any | None = None,
) -> FastAPI:
    """Create FastAPI app backed by EITHER `(model, tokenizer)` (NovaLM) OR a `ChatGeneratorBackend` (e.g. Llama-3.2-3B)."""
    from nova_llm.agent.backends import NovaLMBackend
    from nova_llm.agent.search_agent import SearchAgent
    from nova_llm.agent.search_engine import HybridSearchBackend

    if backend is None:
        if model is None or tokenizer is None:
            raise ValueError("Provide either `backend` or both `(model, tokenizer)`")
        model.eval()
        active_backend: Any = NovaLMBackend(model, tokenizer, model_name=model_name)
    else:
        active_backend = backend

    app = FastAPI(title="Nova-Search LLM Inference Server", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    agent = SearchAgent(
        backend=active_backend,
        search_backend=search_backend or HybridSearchBackend(),
        top_k_results=3,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _CHAT_HTML

    @app.get("/health")
    def health() -> dict[str, Any]:
        info = active_backend.info()
        return {
            "status": "ok",
            "model": info.get("model", model_name),
            "backend": info.get("backend", "nova_lm"),
            "search_agent": True,
            "params": info.get("params", {}),
            "vocab_size": info.get("vocab_size", 0),
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        info = active_backend.info()
        return {
            "object": "list",
            "data": [{"id": info.get("model", model_name), "object": "model", "owned_by": "nova-llm"}],
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
            "thinking": res.thinking,
            "used_search": res.used_search,
            "tool_calls": res.tool_calls,
            "sources": [s.to_dict() for s in res.sources],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Any:
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        msgs = [m.model_dump() for m in req.messages]

        # Native NovaLM fast path (preserves exact prompt/completion token usage counts)
        if model is not None and tokenizer is not None and backend is None:
            prompt_text = tokenizer.format_chat(
                msgs, add_generation_prompt=True, enable_thinking=req.enable_thinking
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

        # Pluggable backend path (Llama 3.2 3B / Hugging Face / Unsloth)
        if req.stream:
            def hf_stream():
                for chunk in active_backend.stream_chat(
                    msgs,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                ):
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
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

            return StreamingResponse(hf_stream(), media_type="text/event-stream")

        content = active_backend.generate_chat(
            msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app
