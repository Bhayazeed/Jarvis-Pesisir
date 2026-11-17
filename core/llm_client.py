"""
Local LLM client for MARK XL.

Supports two backends — selected via  "llm_provider"  in config/api_keys.json:

  "llm_provider": "ollama"   (default)
        Uses Ollama's native /api/chat endpoint.
        Download: https://ollama.com
        Default port: 11434

  "llm_provider": "openai"
        Uses any OpenAI-compatible server: LM Studio, Jan, LocalAI,
        llama.cpp server, vLLM, etc.
        LM Studio download: https://lmstudio.ai   (default port: 1234)
        Set  "llm_url": "http://localhost:1234"  in config.
        Note: tool-calling support depends on the model; use a model that
        supports function/tool calls (e.g. Qwen2.5, Llama-3.1, Mistral).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Generator

import requests

# Matches a sentence boundary: [.!?] followed by whitespace, or a blank line.
# Avoids splitting on decimals (3.5) because those have no space after the dot.
_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

_DEFAULTS = {
    "llm_url":      "http://localhost:11434",
    "llm_model":    "llama3.2",
    "llm_provider": "ollama",   # "ollama" | "openai" | "silab"
    "llm_api_key":  "",
}

_openai_client = None


def get_llm_provider() -> str:
    """Returns 'ollama' or 'openai' (covers LM Studio, LocalAI, Jan, etc.)."""
    raw = _load_config().get("llm_provider", "ollama").strip().lower()
    return "openai" if raw in ("openai", "lmstudio", "localai", "jan", "llamacpp", "silab") else "ollama"


def get_llm_api_key() -> str:
    cfg = _load_config()
    key = cfg.get("llm_api_key", "").strip()
    if not key:
        key = os.getenv("SILAB_KEY") or os.getenv("SILAB_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    return key


def _get_openai_client(url: str):
    global _openai_client
    import openai
    
    # Base URL parsing
    base_url = url.rstrip('/')
    if not (base_url.endswith("/v1") or base_url.endswith("/api")):
        base_url = f"{base_url}/v1"
        
    api_key = get_llm_api_key() or "sk-no-key"
    
    if _openai_client is None or getattr(_openai_client, "base_url", None) != base_url or getattr(_openai_client, "api_key", None) != api_key:
        _openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        
    return _openai_client


def _prepare_openai_messages(messages: list) -> list:
    """
    OpenAI API expects the 'arguments' field in function/tool_calls to be a JSON string,
    but JARVIS's internal framework stores it as a parsed dictionary. We convert it
    back to a JSON string here to avoid '400 Bad Request' errors.
    """
    clean_messages = []
    for msg in messages:
        new_msg = dict(msg)
        if "tool_calls" in new_msg and new_msg["tool_calls"]:
            new_tool_calls = []
            for tc in new_msg["tool_calls"]:
                new_tc = dict(tc)
                if "type" not in new_tc:
                    new_tc["type"] = "function"
                if "function" in new_tc:
                    fn = dict(new_tc["function"])
                    if "arguments" in fn:
                        args = fn["arguments"]
                        if isinstance(args, (dict, list)):
                            fn["arguments"] = json.dumps(args)
                        elif isinstance(args, str):
                            try:
                                json.loads(args)
                            except Exception:
                                fn["arguments"] = "{}"
                    new_tc["function"] = fn
                new_tool_calls.append(new_tc)
            new_msg["tool_calls"] = new_tool_calls
        clean_messages.append(new_msg)
    return clean_messages


def _get_openai_health_endpoint_and_headers(url: str) -> tuple[str, dict]:
    if url.endswith("/v1") or url.endswith("/v1/"):
        health = f"{url.rstrip('/')}/models"
    elif url.endswith("/api") or url.endswith("/api/"):
        health = f"{url.rstrip('/')}/models"
    else:
        health = f"{url.rstrip('/')}/v1/models"
    
    headers = {}
    api_key = get_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return health, headers


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ensure_ollama_running(timeout: int = 15) -> bool:
    """
    For Ollama: ping /api/tags; auto-launch 'ollama serve' if not running.
    For OpenAI-compatible providers: just ping /v1/models (server must be started manually).
    Returns True if the LLM server is reachable.
    """
    url, _   = get_llm_settings()
    provider = get_llm_provider()

    if provider == "openai":
        # OpenAI-compatible servers (LM Studio, LocalAI, etc.) must be started
        # by the user — we just check if they're reachable.
        health, headers = _get_openai_health_endpoint_and_headers(url)
        try:
            ok = requests.get(health, headers=headers, timeout=5).status_code == 200
            if ok:
                print(f"[LLM] OpenAI-compatible server reachable at {url}")
            else:
                print(f"[LLM] Server at {url} returned non-200.  Is it running?")
            return ok
        except Exception as e:
            print(
                f"[LLM] Cannot reach OpenAI-compatible server at {url}.\n"
                "      Make sure LM Studio / LocalAI / Jan is running and the server is started."
            )
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    health = f"{url}/api/tags"

    def _is_up() -> bool:
        try:
            return requests.get(health, timeout=3).status_code == 200
        except Exception:
            return False

    if _is_up():
        return True

    print("[LLM] Ollama not running — launching 'ollama serve'…")
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        print("[LLM] 'ollama' command not found. Install Ollama from https://ollama.com")
        return False
    except Exception as e:
        print(f"[LLM] Could not launch Ollama: {e}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        if _is_up():
            print("[LLM] Ollama started successfully.")
            return True

    print("[LLM] Ollama did not respond within the timeout.")
    return False


def warmup_model(system_prompt: str | None = None) -> bool:
    """
    Pre-load the model AND prime Ollama's KV prefix cache.

    Why the system_prompt matters
    ─────────────────────────────
    Ollama caches the KV attention state of the prompt prefix across requests.
    If warmup includes the same system prompt that real requests will use, Ollama
    evaluates those tokens ONCE at startup.  Every subsequent request only needs
    to evaluate the small delta (user message ± time context) instead of the full
    300-500 token system prompt → drops first-token latency from ~17 s to <1 s.

    Pass the *static* part of the system prompt (the JARVIS protocol text, without
    timestamps or per-minute context) so the prefix stays valid across calls.
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    print(f"[LLM] Warming up '{model}' ({provider})…")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})

    if provider == "openai":
        # OpenAI-compatible: just fire a minimal request to ensure the model is loaded.
        # No keep_alive or KV-cache priming available — server manages this internally.
        client = _get_openai_client(url)
        try:
            client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=1,
                temperature=0.0,
            )
            print(f"[LLM] '{model}' ready (OpenAI-compatible server).")
            return True
        except Exception as e:
            print(f"[LLM] Warmup failed (non-fatal): {e}")
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        # num_gpu:99 → push ALL transformer layers to GPU (Ollama caps at available)
        # This is safe even without a GPU — Ollama silently ignores if n_gpu_layers=0
        "options":    {"num_predict": 1, "num_gpu": 99},
    }
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        print(f"[LLM] '{model}' loaded and KV cache primed.")
        return True
    except Exception as e:
        print(f"[LLM] Warmup failed (non-fatal): {e}")
        return False


def check_model_available(log: Callable | None = None) -> bool:
    """
    Returns True if the configured model is already pulled in Ollama.
    Logs an actionable warning (to console + optional UI callback) if not.
    Always returns True for non-Ollama providers (cannot inspect their model list).
    """
    if get_llm_provider() != "ollama":
        return True

    url, model = get_llm_settings()
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        pulled = [m.get("name", "") for m in resp.json().get("models", [])]
        model_base = model.split(":")[0]
        found = any(
            m == model or m == model_base or m.startswith(model_base + ":")
            for m in pulled
        )
        if not found:
            available = ", ".join(pulled) if pulled else "none"
            warn = (
                f"WRN: Model '{model}' is not pulled in Ollama.\n"
                f"     Available: {available}\n"
                f"     Fix: ollama pull {model}"
            )
            print(warn)
            if log:
                log(f"WRN: '{model}' not found — run: ollama pull {model}")
        return found
    except Exception:
        return True   # Ollama might still be starting up; non-blocking


def get_llm_settings() -> tuple[str, str]:
    """Returns (base_url, model_name)."""
    cfg   = _load_config()
    url   = cfg.get("llm_url",   _DEFAULTS["llm_url"]).rstrip("/")
    model = cfg.get("llm_model", _DEFAULTS["llm_model"])
    return url, model


def call_llm(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> dict:
    """
    Non-streaming chat request.  Routes to Ollama or OpenAI-compatible backend.

    Returns:
        {"content": str, "tool_calls": list}
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()

    if provider == "openai":
        client = _get_openai_client(url)
        kwargs = {
            "model":      model,
            "messages":   _prepare_openai_messages(messages),
            "stream":     False,
            "max_tokens": 150,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg    = choice.message
            
            raw_tc  = getattr(msg, "tool_calls", None) or []
            tc_list = []
            for t in raw_tc:
                try:
                    args = json.loads(t.function.arguments)
                except Exception:
                    args = t.function.arguments
                tc_list.append({
                    "id":       getattr(t, "id", None) or f"call_{len(tc_list)}",
                    "function": {
                        "name":      t.function.name,
                        "arguments": args,
                    },
                })
            return {
                "content":    (getattr(msg, "content", None) or "").strip(),
                "tool_calls": tc_list,
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible LLM call failed: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        "options":    {"num_predict": 150, "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg  = data.get("message", {})
        return {
            "content":    (msg.get("content") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
        }
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] ConnectionError — trying to restart Ollama… ({e})")
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                msg  = data.get("message", {})
                return {
                    "content":    (msg.get("content") or "").strip(),
                    "tool_calls": msg.get("tool_calls") or [],
                }
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 s.")
    except requests.exceptions.HTTPError as e:
        print(f"[LLM] HTTPError: {e.response.status_code} — {e.response.text[:200]}")
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        print(f"[LLM] Unexpected error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def call_llm_text(
    prompt:  str | list,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
) -> str:
    """
    Simple text-only generation (no tools).
    Used by planner, executor, error_handler, code_helper, dev_agent.
    """
    url, default_model = get_llm_settings()
    provider = get_llm_provider()
    m        = model or default_model

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    
    if isinstance(prompt, list):
        import base64
        import io
        text_part = ""
        b64_img = ""
        for item in prompt:
            if isinstance(item, str):
                text_part += item + "\n"
            else:
                buf = io.BytesIO()
                item.convert("RGB").save(buf, format="JPEG")
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        if provider == "openai":
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_part.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": text_part.strip(),
                "images": [b64_img]
            })
    else:
        messages.append({"role": "user", "content": prompt})

    if provider == "openai":
        client = _get_openai_client(url)
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                stream=False,
                max_tokens=600,
                temperature=0.0,
            )
            return (getattr(resp.choices[0].message, "content", None) or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible LLM text call failed: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = {"model": m, "messages": messages, "stream": False, "keep_alive": -1, "options": {"num_predict": 600}}

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()
    except requests.exceptions.ConnectionError:
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                return (resp.json().get("message", {}).get("content") or "").strip()
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"LLM text call failed: {e}")


def _stream_openai(
    messages: list,
    tools:    list | None,
    timeout:  int,
) -> Generator[dict, None, None]:
    """
    Streaming backend for OpenAI-compatible servers (LM Studio, LocalAI, Jan…).

    Parses Server-Sent Events (SSE) and accumulates streaming tool-call fragments
    so the output format is identical to the Ollama backend.
    """
    url, model = get_llm_settings()
    client = _get_openai_client(url)

    kwargs = {
        "model":      model,
        "messages":   _prepare_openai_messages(messages),
        "stream":     True,
        "max_tokens": 150,
        "temperature": 0.0,
    }
    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = "auto"

    try:
        resp = client.chat.completions.create(**kwargs)
        full_content = ""
        buf          = ""
        tc_fragments: dict[int, dict] = {}
        index_map: dict[int, int] = {}
        next_internal_idx = 0

        for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta  = getattr(choice, "delta", None)
            if not delta:
                continue

            text   = getattr(delta, "content", None) or ""
            full_content += text
            buf          += text

            # Accumulate sentence boundaries for streaming TTS
            while True:
                m = _SENT_END.search(buf)
                if not m:
                    break
                sentence = buf[: m.start() + 1].strip()
                buf      = buf[m.end():]
                if sentence:
                    yield {"type": "sentence", "text": sentence}

            # Accumulate streaming tool-call fragments
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    server_idx = getattr(tc, "index", 0)
                    tc_id      = getattr(tc, "id", None)

                    if tc_id:
                        # New tool call
                        internal_idx = next_internal_idx
                        next_internal_idx += 1
                        index_map[server_idx] = internal_idx
                        tc_fragments[internal_idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    else:
                        if server_idx not in index_map:
                            internal_idx = next_internal_idx
                            next_internal_idx += 1
                            index_map[server_idx] = internal_idx
                            tc_fragments[internal_idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                        else:
                            internal_idx = index_map[server_idx]

                    frag = tc_fragments[internal_idx]
                    
                    if getattr(tc, "id", None):
                        frag["id"] += tc.id
                    
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            frag["function"]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            frag["function"]["arguments"] += fn.arguments

        # Flush any trailing content
        if buf.strip():
            yield {"type": "sentence", "text": buf.strip()}

        # Parse accumulated tool-call argument strings → dicts
        tool_calls: list = []
        for idx in sorted(tc_fragments):
            frag = tc_fragments[idx]
            args_str = frag["function"]["arguments"]
            try:
                args = json.loads(args_str)
            except Exception:
                args = args_str
            tool_calls.append({
                "id":       frag.get("id") or f"call_{idx}",
                "function": {"name": frag["function"]["name"], "arguments": args},
            })

        yield {
            "type":       "done",
            "content":    full_content.strip(),
            "tool_calls": tool_calls,
        }

    except Exception as e:
        raise RuntimeError(f"OpenAI-compatible stream failed: {e}")


def call_llm_stream(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> Generator[dict, None, None]:
    """
    Streaming chat request.  Routes to Ollama or OpenAI-compatible backend.

    Yields:
        {"type": "sentence", "text": str}   — each complete sentence as it arrives
        {"type": "done", "content": str, "tool_calls": list}  — when stream ends

    Sentences are split on [.!?] + whitespace so TTS can start immediately.
    Tool calls always appear in the final "done" event.
    """
    provider = get_llm_provider()
    if provider == "openai":
        yield from _stream_openai(messages, tools, timeout)
        return

    url, model = get_llm_settings()
    endpoint   = f"{url}/api/chat"

    payload: dict = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "keep_alive": -1,
        # 150 tokens ≈ 100 words ≈ 3-4 sentences — enough for any voice reply.
        # num_gpu:99 pushes all layers to GPU; num_thread removed (Ollama auto-tunes).
        "options":    {"num_predict": 150, "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    def _do_stream() -> Generator[dict, None, None]:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            tool_calls:  list = []
            buf          = ""

            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg   = chunk.get("message", {})
                delta = msg.get("content") or ""

                full_content += delta
                buf          += delta

                # Yield complete sentences as they accumulate
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end() :]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                tc = msg.get("tool_calls")
                if tc:
                    tool_calls.extend(tc)

                if chunk.get("done"):
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}

                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

    try:
        yield from _do_stream()
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] Stream ConnectionError — trying to restart Ollama… ({e})")
        if ensure_ollama_running():
            yield from _do_stream()
            return
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama stream timed out.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        print(f"[LLM] Stream error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM stream failed: {e}")
