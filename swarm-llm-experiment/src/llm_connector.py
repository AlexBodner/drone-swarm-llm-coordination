# llm_connector.py
import os
import re
import time

# ── Model selection ────────────────────────────────────────────────────────
# llama-3.3-70b-versatile : 100K tokens/day  — more capable
# llama-3.1-8b-instant    : 500K tokens/day  — faster, higher limit
GROQ_MODEL = "llama-3.1-8b-instant"

# ── Option A: Groq (recommended for development) ──────────────────────────
def get_completion_groq(prompt: str, max_retries: int = 3, max_tokens: int = 4096) -> str:
    """
    Call Groq API with automatic retry on 429 rate-limit errors.
    Parses the 'Please try again in Xs' message to sleep the right amount,
    defaulting to 900s if not parseable.

    max_tokens: output token budget. Scale up for larger swarms to avoid
    truncated JSON. llama-3.1-8b-instant supports up to 8192 output tokens.
    """
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    client = Groq(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit_exceeded" in err_str:
                # Parse "Please try again in Xs" from error message
                match = re.search(r"try again in ([\d.]+)s", err_str)
                wait_s = float(match.group(1)) + 5 if match else 900.0
                wait_s = min(wait_s, 1800)  # cap at 30 min
                if attempt < max_retries - 1:
                    print(f"\n[Rate limit] Sleeping {wait_s:.0f}s then retrying "
                          f"(attempt {attempt+1}/{max_retries})...", flush=True)
                    time.sleep(wait_s)
                    continue
            raise  # re-raise on last attempt or non-rate-limit errors


# ── Option B: Gemini Flash (recommended for pilot experiment) ──────────────
def get_completion_gemini(prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={"temperature": 0},
    )
    response = model.generate_content(prompt)
    return response.text


# ── Option C: Claude (recommended for full experiment) ─────────────────────
def get_completion_claude(prompt: str, max_tokens: int = 4096) -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def get_available_connector() -> tuple:
    """Return (name, fn) for the first LLM with a valid API key."""
    checks = [
        (f"groq/{GROQ_MODEL}", "GROQ_API_KEY", get_completion_groq),
        ("gemini/gemini-2.0-flash", "GEMINI_API_KEY", get_completion_gemini),
        ("claude/claude-sonnet-4", "ANTHROPIC_API_KEY", get_completion_claude),
    ]
    for name, env_var, fn in checks:
        if os.environ.get(env_var):
            return name, fn
    return None, None


# ── Active connector (auto-selects based on available API key) ─────────────
def get_completion(prompt: str, max_tokens: int = 4096) -> str:
    name, fn = get_available_connector()
    if fn is None:
        raise RuntimeError(
            "No LLM API key found. Set one of: GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY"
        )
    try:
        return fn(prompt, max_tokens=max_tokens)
    except TypeError:
        # Gemini connector doesn't accept max_tokens yet — fall back gracefully
        return fn(prompt)


# ── Manual override: change this to force a specific model ─────────────────
# get_completion = get_completion_groq
# get_completion = get_completion_gemini
# get_completion = get_completion_claude


if __name__ == "__main__":
    name, fn = get_available_connector()
    if fn is None:
        print("No API key found in environment. Please set GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY.")
        exit(1)

    print(f"Testing LLM connector using: {name}")
    test_prompt = (
        "Write a Python function called `plan(state)` that returns a dict "
        "{0: (1.0, 0.0, 1.0), 1: (-1.0, 0.0, 1.0)}. "
        "Return ONLY a ```python ... ``` code block."
    )
    response = get_completion(test_prompt)
    print(f"Response:\n{response}")
    assert "```python" in response or "def plan" in response, "Response should contain code"
    print("\nCheckpoint PASSED: llm_connector.py works correctly.")
