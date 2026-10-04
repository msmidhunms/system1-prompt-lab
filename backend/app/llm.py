"""LLM providers used by the Karpathy loop to propose new Laya prompts.

The provider, model and credentials are chosen in the frontend and stored in the
`app_settings` table. An API key that is not stored falls back to the provider's
environment variable.
"""

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import anthropic
import httpx
from sqlalchemy.orm import Session

from app.database import AppSetting

LLM_SETTINGS_KEY = "llm_config"
REQUEST_TIMEOUT = 600.0

# kind: "anthropic" (official SDK), "openai_compat" (/chat/completions), "cli" (local agent CLI)
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "label": "Claude (Anthropic API)",
        "kind": "anthropic",
        "default_model": "claude-opus-5-5",
        "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "claude-fable-5-1"],
        "key_env": "ANTHROPIC_API_KEY",
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "kind": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5",
        "models": ["gpt-5", "gpt-5-mini", "gpt-4.1", "o4-mini"],
        "key_env": "OPENAI_API_KEY",
        "needs_key": True,
    },
    "ollama": {
        "label": "Ollama (local)",
        "kind": "openai_compat",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "models": [],
        "key_env": None,
        "needs_key": False,
    },
    "gemini": {
        "label": "Google Gemini",
        "kind": "openai_compat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-pro",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "key_env": "GEMINI_API_KEY",
        "needs_key": True,
    },
    "openai_compatible": {
        "label": "OpenAI-compatible (OpenRouter, Groq, LM Studio, vLLM, ...)",
        "kind": "openai_compat",
        "base_url": "",
        "default_model": "",
        "models": [],
        "key_env": "OPENAI_COMPATIBLE_API_KEY",
        "needs_key": False,
    },
    "claude_cli": {
        "label": "Claude Code CLI (uses your claude login)",
        "kind": "cli",
        "command": "claude",
        "default_model": "",
        "models": ["opus", "sonnet", "haiku"],
        "key_env": None,
        "needs_key": False,
    },
    "codex_cli": {
        "label": "Codex CLI (uses your codex login)",
        "kind": "cli",
        "command": "codex",
        "default_model": "",
        "models": [],
        "key_env": None,
        "needs_key": False,
    },
}

DEFAULT_PROVIDER = "anthropic"


class LLMError(Exception):
    """An LLM call failed; the message is safe to show in the UI."""


# ---------------------------------------------------------------- settings

def _load_raw(db: Session) -> Dict[str, Any]:
    row = db.query(AppSetting).filter(AppSetting.key == LLM_SETTINGS_KEY).first()
    raw = dict(row.value) if row and row.value else {}
    raw.setdefault("active", DEFAULT_PROVIDER)
    raw.setdefault("providers", {})
    return raw


def _save_raw(db: Session, raw: Dict[str, Any]) -> None:
    row = db.query(AppSetting).filter(AppSetting.key == LLM_SETTINGS_KEY).first()
    if row:
        row.value = raw
    else:
        db.add(AppSetting(key=LLM_SETTINGS_KEY, value=raw))
    db.commit()


def resolve_config(db: Session, provider: Optional[str] = None) -> Dict[str, Any]:
    """Return the full config (including the API key) for a provider, defaults filled in."""
    raw = _load_raw(db)
    name = provider or raw["active"]
    if name not in PROVIDERS:
        raise LLMError(f"Unknown LLM provider '{name}'")
    preset = PROVIDERS[name]
    stored = raw["providers"].get(name, {})
    api_key = stored.get("api_key") or (os.getenv(preset["key_env"]) if preset["key_env"] else None)
    return {
        "provider": name,
        "kind": preset["kind"],
        "model": stored.get("model") or preset["default_model"],
        "base_url": (stored.get("base_url") or preset.get("base_url") or "").rstrip("/"),
        "api_key": api_key or None,
        "key_source": "stored" if stored.get("api_key") else ("env" if api_key else None),
    }


def public_config(db: Session) -> Dict[str, Any]:
    """Settings for the frontend: every provider's config, with keys reduced to a flag."""
    raw = _load_raw(db)
    providers = {}
    for name, preset in PROVIDERS.items():
        cfg = resolve_config(db, name)
        providers[name] = {
            "label": preset["label"],
            "kind": preset["kind"],
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "default_base_url": preset.get("base_url", ""),
            "models": preset["models"],
            "needs_key": preset["needs_key"],
            "key_env": preset["key_env"],
            "api_key_set": cfg["api_key"] is not None,
            "key_source": cfg["key_source"],
            "cli_found": shutil.which(preset["command"]) is not None if preset["kind"] == "cli" else None,
        }
    return {"active": raw["active"], "providers": providers}


def update_config(
    db: Session,
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    clear_api_key: bool = False,
    make_active: bool = True,
) -> None:
    """Store settings for one provider. A blank api_key leaves the stored key untouched."""
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown LLM provider '{provider}'")
    raw = _load_raw(db)
    providers = dict(raw["providers"])
    stored = dict(providers.get(provider, {}))
    if model is not None:
        stored["model"] = model.strip()
    if base_url is not None:
        stored["base_url"] = base_url.strip()
    if api_key and api_key.strip():
        stored["api_key"] = api_key.strip()
    if clear_api_key:
        stored.pop("api_key", None)
    providers[provider] = stored
    raw["providers"] = providers
    if make_active:
        raw["active"] = provider
    _save_raw(db, raw)


# ---------------------------------------------------------------- completion

def complete(cfg: Dict[str, Any], system: str, user: str) -> str:
    """Send one system+user prompt to the configured LLM and return its text reply."""
    kind = cfg["kind"]
    if kind == "anthropic":
        return _complete_anthropic(cfg, system, user)
    if kind == "openai_compat":
        return _complete_openai_compat(cfg, system, user)
    if kind == "cli":
        return _complete_cli(cfg, system, user)
    raise LLMError(f"Unknown provider kind '{kind}'")


def _anthropic_client(cfg: Dict[str, Any]) -> anthropic.Anthropic:
    # Without a stored key the SDK resolves credentials itself (env var or `ant auth login`).
    if cfg.get("api_key"):
        return anthropic.Anthropic(api_key=cfg["api_key"], timeout=REQUEST_TIMEOUT)
    return anthropic.Anthropic(timeout=REQUEST_TIMEOUT)


def _complete_anthropic(cfg: Dict[str, Any], system: str, user: str) -> str:
    try:
        response = _anthropic_client(cfg).messages.create(
            model=cfg["model"],
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError:
        raise LLMError("Anthropic rejected the API key")
    except anthropic.NotFoundError:
        raise LLMError(f"Anthropic model '{cfg['model']}' was not found")
    except anthropic.RateLimitError:
        raise LLMError("Anthropic rate limit hit")
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Could not reach the Anthropic API: {e}")
    except anthropic.AnthropicError as e:
        # Raised at client construction when no credentials can be resolved.
        raise LLMError(f"Anthropic client error: {e}")

    if response.stop_reason == "refusal":
        raise LLMError("Claude declined the request")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude's reply was cut off at max_tokens")
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise LLMError("Claude returned an empty reply")
    return text


def _openai_headers(cfg: Dict[str, Any]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}


def _complete_openai_compat(cfg: Dict[str, Any], system: str, user: str) -> str:
    if not cfg["base_url"]:
        raise LLMError("Base URL is not set for this provider")
    if not cfg["model"]:
        raise LLMError("Model is not set for this provider")
    try:
        response = httpx.post(
            f"{cfg['base_url']}/chat/completions",
            headers=_openai_headers(cfg),
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach {cfg['base_url']}: {e}")
    if response.status_code != 200:
        raise LLMError(f"{cfg['provider']} returned HTTP {response.status_code}: {response.text[:500]}")
    try:
        text = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected response from {cfg['provider']}: {response.text[:500]}")
    if not text or not text.strip():
        raise LLMError(f"{cfg['provider']} returned an empty reply")
    return text


def _complete_cli(cfg: Dict[str, Any], system: str, user: str) -> str:
    command = PROVIDERS[cfg["provider"]]["command"]
    if shutil.which(command) is None:
        raise LLMError(f"'{command}' was not found on PATH")
    prompt = f"{system}\n\n{user}"

    # Run in an empty directory so the agent CLI does not pick up this repo as context.
    with tempfile.TemporaryDirectory() as workdir:
        last_message = os.path.join(workdir, "last_message.txt")
        if command == "claude":
            args = ["claude", "-p", "--output-format", "text"]
            if cfg["model"]:
                args += ["--model", cfg["model"]]
        else:
            args = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
                    "--output-last-message", last_message]
            if cfg["model"]:
                args += ["--model", cfg["model"]]
            args.append("-")
        try:
            proc = subprocess.run(
                args, input=prompt, capture_output=True, text=True,
                cwd=workdir, timeout=REQUEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise LLMError(f"'{command}' timed out after {int(REQUEST_TIMEOUT)}s")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-500:]
            raise LLMError(f"'{command}' exited with code {proc.returncode}: {detail}")
        text = proc.stdout
        if command == "codex" and os.path.exists(last_message):
            with open(last_message, encoding="utf-8") as f:
                text = f.read()
    if not text.strip():
        raise LLMError(f"'{command}' returned an empty reply")
    return text


# ---------------------------------------------------------------- helpers

def list_models(cfg: Dict[str, Any]) -> List[str]:
    """Ask the provider which models it serves (falls back to the preset list)."""
    preset = PROVIDERS[cfg["provider"]]
    if cfg["kind"] == "anthropic":
        try:
            return [m.id for m in _anthropic_client(cfg).models.list()]
        except anthropic.AnthropicError as e:
            raise LLMError(f"Could not list Anthropic models: {e}")
    if cfg["kind"] == "openai_compat":
        if not cfg["base_url"]:
            raise LLMError("Base URL is not set for this provider")
        try:
            response = httpx.get(f"{cfg['base_url']}/models", headers=_openai_headers(cfg), timeout=20.0)
            response.raise_for_status()
            data = response.json().get("data", [])
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"Could not list models from {cfg['base_url']}: {e}")
        # Gemini's compatibility endpoint prefixes ids with "models/".
        return sorted(m["id"].removeprefix("models/") for m in data if m.get("id"))
    return preset["models"]


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (tolerates prose and code fences)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise LLMError("The LLM reply did not contain a JSON object")
