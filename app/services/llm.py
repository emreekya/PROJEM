import json
import logging
import re
import requests
from collections import Counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config

_max_retries = 5
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEPRECATED_GEMINI_MODELS = {"gemini-pro", "gemini-1.0-pro"}
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 20
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond ENTIRELY in the same language as the video subject.
9. CRITICAL: use ONLY that single language. Never mix in English, Spanish or any foreign words. Every single word must be a correct, natural word of the subject's language. If the subject is Turkish, write 100% fluent Turkish with no foreign words.
10. HOOK: the very FIRST sentence must be a powerful, curiosity-provoking hook that instantly grabs attention — a surprising fact, a bold claim, or an intriguing question. Do NOT start with a flat definition or "X is ...". Make the viewer want to keep watching.
11. The HOOK itself MUST also be written in the subject's language — NEVER in English. For example, in Turkish begin with something like "Biliyor muydunuz ki..." or a bold Turkish statement; NEVER write "Did you know that" or any English phrase. Absolutely no English anywhere, including the opening.
12. Avoid repetition. Every sentence must add a new fact, example or insight instead of paraphrasing a previous sentence.
13. Prefer concrete, visually depictable statements. Avoid vague filler such as "it has mysterious features" unless the next sentence immediately explains a specific feature.
14. If the video subject promises a numbered list, such as "5 features" or "7 facts", deliver exactly that many clearly distinct points. Use natural spoken transitions in the subject's language so each point can become a separate visual scene.
15. For numbered-list subjects, explicitly signal every point with natural spoken markers such as "Birincisi", "İkincisi" or their equivalents in the subject's language.
16. Do not add historical, mythological or religious background unless the video subject explicitly asks for history, mythology or religion. Stay focused on the promised topic.
17. Avoid unsupported absolute claims and stereotypes such as "the only sign", "always", or presenting personality traits as guaranteed facts. Use natural qualifiers such as "often", "may" or "can".
18. Use natural contemporary wording, correct proper names and the standard date format of the subject's language. For Turkish, write dates such as "21 Mayıs", not "Mayıs 21", and prefer established Turkish names such as "Merkür".
19. Do not use insulting, stigmatizing or sensational labels as a hook. Explain nuances in neutral language.

## Viral short-form structure (CRITICAL for retention):
20. STRUCTURE: Follow this arc precisely — (a) HOOK: one punchy curiosity-driven opening line; (b) PROMISE/SETUP: one short line that tells the viewer what they will get and why staying is worth it; (c) BODY: the distinct points, each tight and concrete; (d) PAYOFF + CTA: a satisfying closing line followed by a short, natural call-to-action.
21. RETENTION BRIDGES: Between points, weave in short open-loop teasers in the subject's language so the viewer keeps watching — e.g. in Turkish "ama asıl sürpriz sonda", "birazdan gelen madde çoğu kişiyi şaşırtıyor", "bir saniye, en iyisi geliyor". Keep them natural, never clickbait-lies.
22. PACING: Write for spoken short-form video (TikTok/Reels/Shorts). Use short, punchy sentences. Avoid long subordinate clauses. Each sentence should be easy to say in one breath and map to one visual.
23. CTA: End with exactly one natural, language-appropriate call-to-action (e.g. in Turkish "Hangisi sana en yakın, yorumlara yaz", "Kaydet, sonra lazım olacak", "Takip et, yarın devamı geliyor"). Keep it to a single sentence. Never add hashtags, emojis, or links.
24. NO META: Do not narrate the structure (never literally write "hook", "intro", "call to action"); just deliver the content naturally.
""".strip()


def script_repetition_reason(script: str) -> str:
    """Return a diagnostic message when a script is trapped in a repetition loop."""
    text = re.sub(r"\s+", " ", (script or "")).strip()
    if len(text) < 300:
        return ""

    sentences = [
        re.sub(r"[^\w\s]", "", sentence.casefold()).strip()
        for sentence in re.split(r"[.!?。！？]+", text)
    ]
    sentences = [sentence for sentence in sentences if len(sentence) >= 24]
    sentence_counts = Counter(sentences)
    if sentence_counts and max(sentence_counts.values()) >= 3:
        return "the same sentence was repeated at least 3 times"

    words = re.findall(r"\w+", text.casefold())
    if len(words) >= 80:
        windows = Counter(tuple(words[i : i + 8]) for i in range(len(words) - 7))
        if windows and max(windows.values()) >= 5:
            return "the same 8-word phrase was repeated at least 5 times"
    return ""


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    content = content.strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    return content.replace("\n", "")


def _generate_via_pollinations(prompt: str) -> str:
    """Pollinations metin API'si ile yanıt üretir (ücretsiz, anahtarsız, sınırsız).

    Cloudflare LLM kotası bittiğinde otomatik YEDEK olarak kullanılır.
    """
    base_url = (
        config.app.get("pollinations_base_url", "")
        or "https://text.pollinations.ai/openai"
    )
    model_name = config.app.get("pollinations_model_name", "") or "openai-fast"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "seed": 101,
    }
    if config.app.get("pollinations_private"):
        payload["private"] = True
    if config.app.get("pollinations_referrer"):
        payload["referrer"] = config.app.get("pollinations_referrer")
    headers = {"Content-Type": "application/json"}
    token = config.app.get("pollinations_api_key")
    if isinstance(token, list):
        token = token[0] if token else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    result = response.json()
    if result and "choices" in result and len(result["choices"]) > 0:
        return _normalize_text_response(
            result["choices"][0]["message"]["content"], "pollinations"
        )
    raise Exception("[pollinations] geçersiz yanıt formatı")


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _generate_response(prompt: str) -> str:
    try:
        content = ""
        llm_provider = config.app.get("llm_provider", "openai")
        logger.info(f"llm provider: {llm_provider}")
        if llm_provider == "g4f":
            if not config.app.get("enable_g4f", False):
                raise ValueError(
                    "g4f provider is disabled by default because it relies on "
                    "reverse-engineered third-party endpoints. Set enable_g4f=true "
                    "in config.toml only if you understand and accept the security, "
                    "reliability, and legal risks."
                )

            logger.warning(
                "g4f provider is enabled. This provider may be unstable and carries "
                "supply-chain and terms-of-service risks. Prefer official providers, "
                "OpenAI-compatible APIs, LiteLLM, Ollama, or local inference for production."
            )
            try:
                import g4f
            except ImportError as e:
                raise ValueError(
                    "g4f package is not installed by default. Install the optional "
                    "dependency with `uv sync --extra g4f` only if you understand "
                    "and accept the provider risks."
                ) from e

            model_name = config.app.get("g4f_model_name", "")
            if not model_name:
                model_name = "gpt-3.5-turbo-16k-0613"
            content = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            api_version = ""  # for azure
            if llm_provider == "moonshot":
                api_key = config.app.get("moonshot_api_key")
                model_name = config.app.get("moonshot_model_name")
                base_url = "https://api.moonshot.cn/v1"
            elif llm_provider == "ollama":
                # api_key = config.app.get("openai_api_key")
                api_key = "ollama"  # any string works but you are required to have one
                model_name = config.app.get("ollama_model_name")
                base_url = config.app.get("ollama_base_url", "")
                if not base_url:
                    base_url = config.get_default_ollama_base_url()
            elif llm_provider == "openai":
                api_key = config.app.get("openai_api_key")
                model_name = config.app.get("openai_model_name")
                base_url = config.app.get("openai_base_url", "")
                if not base_url:
                    base_url = "https://api.openai.com/v1"
            elif llm_provider == "oneapi":
                api_key = config.app.get("oneapi_api_key")
                model_name = config.app.get("oneapi_model_name")
                base_url = config.app.get("oneapi_base_url", "")
            elif llm_provider == "azure":
                api_key = config.app.get("azure_api_key")
                model_name = config.app.get("azure_model_name")
                base_url = config.app.get("azure_base_url", "")
                api_version = config.app.get("azure_api_version", "2024-02-15-preview")
            elif llm_provider == "gemini":
                api_key = config.app.get("gemini_api_key")
                model_name = config.app.get("gemini_model_name")
                base_url = config.app.get("gemini_base_url", "")
                # Gemini 旧模型名已经陆续下线，这里自动兼容历史配置，
                # 避免用户沿用旧值时直接收到 404。
                if not model_name:
                    model_name = _DEFAULT_GEMINI_MODEL
                elif model_name in _DEPRECATED_GEMINI_MODELS:
                    logger.warning(
                        f"gemini model '{model_name}' is deprecated, fallback to '{_DEFAULT_GEMINI_MODEL}'"
                    )
                    model_name = _DEFAULT_GEMINI_MODEL
            elif llm_provider == "grok":
                api_key = config.app.get("grok_api_key")
                model_name = config.app.get("grok_model_name")
                base_url = config.app.get("grok_base_url", "")
                if not base_url:
                    base_url = "https://api.x.ai/v1"
            elif llm_provider == "qwen":
                api_key = config.app.get("qwen_api_key")
                model_name = config.app.get("qwen_model_name")
                base_url = "***"
            elif llm_provider == "cloudflare":
                api_key = config.app.get("cloudflare_api_key")
                model_name = config.app.get("cloudflare_model_name")
                account_id = config.app.get("cloudflare_account_id")
                base_url = "***"
            elif llm_provider == "minimax":
                api_key = config.app.get("minimax_api_key")
                model_name = config.app.get("minimax_model_name")
                base_url = config.app.get("minimax_base_url", "")
                if not base_url:
                    base_url = "https://api.minimax.io/v1"
            elif llm_provider == "mimo":
                api_key = config.app.get("mimo_api_key")
                model_name = config.app.get("mimo_model_name")
                base_url = config.app.get("mimo_base_url", "")
                # Xiaomi MiMo 官方文档说明其兼容 OpenAI Chat Completions 协议。
                # 这里使用独立 provider 保存默认地址和模型名，用户不用把 MiMo
                # 当作 OpenAI 自定义 base_url 配置，也便于后续继续接入 MiMo
                # 多模态或 TTS 能力时保持边界清晰。
                if not base_url:
                    base_url = "https://api.xiaomimimo.com/v1"
                if not model_name:
                    model_name = "mimo-v2.5-pro"
            elif llm_provider == "deepseek":
                api_key = config.app.get("deepseek_api_key")
                model_name = config.app.get("deepseek_model_name")
                base_url = config.app.get("deepseek_base_url")
                if not base_url:
                    base_url = "https://api.deepseek.com"
            elif llm_provider == "modelscope":
                api_key = config.app.get("modelscope_api_key")
                model_name = config.app.get("modelscope_model_name")
                base_url = config.app.get("modelscope_base_url")
                if not base_url:
                    base_url = "https://api-inference.modelscope.cn/v1/"
            elif llm_provider == "ernie":
                api_key = config.app.get("ernie_api_key")
                secret_key = config.app.get("ernie_secret_key")
                base_url = config.app.get("ernie_base_url")
                model_name = "***"
                if not secret_key:
                    raise ValueError(
                        f"{llm_provider}: secret_key is not set, please set it in the config.toml file."
                    )
            elif llm_provider == "pollinations":
                try:
                    base_url = config.app.get("pollinations_base_url", "")
                    if not base_url:
                        base_url = "https://text.pollinations.ai/openai"
                    model_name = config.app.get("pollinations_model_name", "openai-fast")
                   
                    # Prepare the payload
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "seed": 101  # Optional but helps with reproducibility
                    }
                    
                    # Optional parameters if configured
                    if config.app.get("pollinations_private"):
                        payload["private"] = True
                    if config.app.get("pollinations_referrer"):
                        payload["referrer"] = config.app.get("pollinations_referrer")
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    # Make the API request
                    response = requests.post(base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return _normalize_text_response(content, llm_provider)
                    else:
                        raise Exception(f"[{llm_provider}] returned an invalid response format")
                        
                except requests.exceptions.RequestException as e:
                    raise Exception(f"[{llm_provider}] request failed: {str(e)}")
                except Exception as e:
                    raise Exception(f"[{llm_provider}] error: {str(e)}")

            elif llm_provider == "litellm":
                model_name = config.app.get("litellm_model_name")

            if llm_provider not in ["pollinations", "ollama", "litellm"]:  # Skip validation for providers that don't require API key
                if not api_key:
                    raise ValueError(
                        f"{llm_provider}: api_key is not set, please set it in the config.toml file."
                    )
                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )
                if not base_url and llm_provider not in ["gemini"]:
                    raise ValueError(
                        f"{llm_provider}: base_url is not set, please set it in the config.toml file."
                    )

            if llm_provider == "qwen":
                import dashscope
                from dashscope.api_entities.dashscope_response import GenerationResponse

                dashscope.api_key = api_key
                response = dashscope.Generation.call(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, GenerationResponse):
                        status_code = response.status_code
                        if status_code != 200:
                            raise Exception(
                                f'[{llm_provider}] returned an error response: "{response}"'
                            )

                        content = response["output"]["text"]
                        return content.replace("\n", "")
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}"'
                        )
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            if llm_provider == "gemini":
                import google.generativeai as genai

                if not base_url:
                    genai.configure(api_key=api_key, transport="rest")
                else:
                    genai.configure(api_key=api_key, transport="rest", client_options={'api_endpoint': base_url})

                generation_config = {
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }

                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                try:
                    response = model.generate_content(prompt)
                    candidates = response.candidates
                    generated_text = candidates[0].content.parts[0].text
                except (AttributeError, IndexError) as e:
                    logger.warning(
                        f"gemini returned invalid response content: {str(e)}"
                    )
                    raise ValueError(
                        f"[{llm_provider}] returned invalid response content"
                    )

                return _normalize_text_response(generated_text, llm_provider)

            if llm_provider == "cloudflare":
                # account_id ve api_key tek string VEYA (görsel rotasyonuyla aynı) LİSTE
                # olabilir. Hesapları sırayla dener; biri başarısız olursa (kota vb.)
                # sıradakine geçer.
                _accs = account_id if isinstance(account_id, list) else [account_id]
                _keys = api_key if isinstance(api_key, list) else [api_key]
                _accs = [str(a).strip() for a in _accs if a and str(a).strip()]
                _keys = [str(k).strip() for k in _keys if k and str(k).strip()]
                _pairs = list(zip(_accs, _keys))
                if not _pairs:
                    raise ValueError(
                        "cloudflare: account_id/api_key ayarlı değil (config.toml)"
                    )
                _last_err = ""
                _gw = (config.app.get("cloudflare_ai_gateway") or "").strip()
                for _acc, _key in _pairs:
                    try:
                        if _gw:
                            _url = f"https://gateway.ai.cloudflare.com/v1/{_acc}/{_gw}/workers-ai/{model_name}"
                        else:
                            _url = f"https://api.cloudflare.com/client/v4/accounts/{_acc}/ai/run/{model_name}"
                        response = requests.post(
                            _url,
                            headers={"Authorization": f"Bearer {_key}"},
                            json={
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a friendly assistant",
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                "max_tokens": 2048,
                            },
                            timeout=120,
                        )
                        if response.status_code == 200:
                            result = response.json()
                            _raw = (result.get("result") or {}).get("response")
                            # Cloudflare bazı durumlarda 'response'u liste olarak döndürür
                            # (özellikle JSON dizisi istenince, ör. anahtar kelimeler).
                            # Yapıyı korumak için JSON string'e çeviriyoruz; çağıran taraf
                            # (script düz metin / terms JSON) uygun şekilde işler.
                            if isinstance(_raw, list):
                                _raw = json.dumps(_raw, ensure_ascii=False)
                            if _raw:
                                return _normalize_text_response(_raw, llm_provider)
                            _last_err = "boş yanıt (response yok)"
                            logger.warning(f"[cloudflare] {_last_err}: {str(result)[:150]}")
                            continue
                        _last_err = f"{response.status_code}: {response.text[:150]}"
                        logger.warning(f"[cloudflare] hesap denemesi başarısız: {_last_err}")
                    except Exception as e:
                        _last_err = str(e)
                        logger.warning(f"[cloudflare] istek hatası: {_last_err}")
                # Tüm Cloudflare hesaplarının kotası bitti -> Pollinations'a düş (yedek LLM).
                logger.warning(
                    f"[cloudflare] tüm hesaplar başarısız ({_last_err}) "
                    f"-> Pollinations'a düşülüyor (yedek LLM)"
                )
                try:
                    return _generate_via_pollinations(prompt)
                except Exception as _pe:
                    raise Exception(
                        f"[cloudflare] tüm hesaplar başarısız: {_last_err} | "
                        f"Pollinations yedeği de başarısız: {str(_pe)}"
                    )

            if llm_provider == "ernie":
                response = requests.post(
                    "https://aip.baidubce.com/oauth/2.0/token", 
                    params={
                        "grant_type": "client_credentials",
                        "client_id": api_key,
                        "client_secret": secret_key,
                    }
                )
                access_token = response.json().get("access_token")
                url = f"{base_url}?access_token={access_token}"

                payload = json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "penalty_score": 1,
                        "disable_search": False,
                        "enable_citation": False,
                        "response_format": "text",
                    }
                )
                headers = {"Content-Type": "application/json"}

                response = requests.request(
                    "POST", url, headers=headers, data=payload
                ).json()
                return _normalize_text_response(response.get("result"), llm_provider)

            if llm_provider == "litellm":
                import litellm

                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )

                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    drop_params=True,
                )

                if not response:
                    raise ValueError(f"[{llm_provider}] returned empty response")
                if not getattr(response, "choices", None):
                    raise ValueError(f"[{llm_provider}] returned empty response")

                return _extract_chat_completion_text(response, llm_provider)

            if llm_provider == "azure":
                # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
                # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
                # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
                # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
                logger.info(f"requesting azure chat completion, model: {model_name}")
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, ChatCompletion):
                        return _extract_chat_completion_text(response, llm_provider)
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                            f"connection and try again."
                        )
                else:
                    raise Exception(
                        f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                    )

            if llm_provider == "modelscope":
                content = ''
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                    stream=True
                )
                if response:
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content += delta.content
                    
                    if not content.strip():
                        raise ValueError("Empty content in stream response")
                    
                    return _normalize_text_response(content, llm_provider)
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            else:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )

            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        return _normalize_text_response(content, llm_provider)
    except Exception as e:
        return f"Error: {str(e)}"


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层已经用 Pydantic 做长度校验；这里继续兜底，是为了保护
    # WebUI 或内部服务直接调用 generate_script 时不会把超长提示词发送给模型，
    # 避免 token 成本异常和请求失败。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI 和 API 都会限制范围；这里兜底处理内部调用，避免异常参数直接扩大
        # LLM 生成成本或生成空结果。
        logger.warning(
            "script paragraph_number is out of range and will be clamped: "
            f"{value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def _dedupe_sentences(script: str) -> str:
    """Senaryodaki YİNELENEN cümleleri (ilkini koruyarak) atar.

    Kıyaslama gibi konularda model aynı cümleyi 2-3 kez yazabiliyor; bu da tekrar
    dedektörünü tetikleyip senaryoyu komple çöpe attırıyordu. Burada sadece birebir
    yinelenenleri eliyoruz; içerik/anlam korunur.
    """
    text = re.sub(r"\s+", " ", (script or "")).strip()
    if not text:
        return ""
    parts = re.findall(r"[^.!?]*[.!?]+|\S[^.!?]*$", text)
    out = []
    kept_keys = []
    kept_wordsets = []
    kept_prefix = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        key = re.sub(r"[^\w\s]", "", s.casefold()).strip()
        words = [w for w in key.split() if w]
        # Kısa bağlaç/geçiş cümleleri serbest.
        if len(key) < 12:
            out.append(s)
            continue
        # 1) Birebir tekrar.
        if key in kept_keys:
            continue
        wordset = set(words)
        prefix3 = tuple(words[:3])
        # 2) ŞABLON/yakın tekrar: önceki bir cümleyle yüksek kelime örtüşmesi.
        #    Aynı ilk 3 kelime ("Ronaldo 2 kez ...") + %70 örtüşme, ya da tek başına
        #    %88 örtüşme -> yinelenmiş say, ele.
        dup = False
        for kw, kp in zip(kept_wordsets, kept_prefix):
            if not kw:
                continue
            inter = len(wordset & kw)
            shorter = min(len(wordset), len(kw)) or 1
            ratio = inter / shorter
            if (kp == prefix3 and ratio >= 0.7) or ratio >= 0.88:
                dup = True
                break
        if dup:
            continue
        out.append(s)
        kept_keys.append(key)
        kept_wordsets.append(wordset)
        kept_prefix.append(prefix3)
    return " ".join(out).strip()


_META_SENTENCE_RE = re.compile(
    r"\b(bu\s+video|bu\s+yaz[ıi]|sizlerle\s+payla|izlemeye\s+devam|"
    r"videoya\s+ho[şs]\s+geldin|bu\s+içerik|aşağıdaki|yukarıdaki)",
    re.IGNORECASE,
)


def _strip_meta_sentences(script: str) -> str:
    """'Bu video ile...', 'sizlerle paylaşacağız' gibi meta cümleleri ayıklar."""
    text = re.sub(r"\s+", " ", (script or "")).strip()
    if not text:
        return script
    parts = re.findall(r"[^.!?]*[.!?]+|\S[^.!?]*$", text)
    kept = [s.strip() for s in parts if s.strip() and not _META_SENTENCE_RE.search(s)]
    result = " ".join(kept).strip()
    # Hepsi meta çıktıysa orijinali koru (boş senaryo riskini önle).
    return result if len(result) >= max(40, len(text) // 3) else text


def _ensure_cta(script: str) -> str:
    """Senaryo soru/CTA ile bitmiyorsa marka tonunda bir kapanış ekler."""
    s = (script or "").strip()
    if not s:
        return s
    tail = s[-70:].lower()
    if s.rstrip().endswith("?") or any(
        k in tail for k in ["yorumlara yaz", "yorumlarda", "takip et", "kaydet", "yorum yap"]
    ):
        return s
    return s + " Sen ne düşünüyorsun, yorumlara yaz abi."


def _is_zodiac_subject(video_subject: str) -> bool:
    """Konu bir ASTROLOJİ burcuyla mı ilgili? ('burç/burcu/burcunun' geçiyorsa)."""
    s = (video_subject or "").casefold()
    s = s.replace("ç", "c").replace("ı", "i").replace("İ".casefold(), "i")
    return "burc" in s


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # 将“脚本生成规则”和“运行时上下文”分开拼接。这样高级用户即使覆盖默认
    # system prompt，也不会漏掉视频主题、语言、段落数这些每次生成都必须带上的参数。
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    list_count_match = re.search(r"\b(\d{1,2})\b", video_subject or "")
    list_count = int(list_count_match.group(1)) if list_count_match else 0
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if list_count:
        prompt += (
            f"\n- numbered-list promise detected in title: deliver exactly {list_count} "
            "distinct, concrete and visually depictable points"
        )
    if _is_zodiac_subject(video_subject):
        prompt += (
            "\n- CRITICAL ASTROLOGY DISAMBIGUATION: this topic is about a ZODIAC SIGN and the "
            "people born under it. In Turkish many sign names are also everyday nouns/animals "
            "(Koç=ram, Boğa=bull, Yengeç=crab, Aslan=lion, Balık=fish, Oğlak=kid goat, "
            "Başak=ear of wheat, Terazi=scales, Akrep=scorpion, Yay=bow). You MUST treat the "
            "subject ONLY as the astrology sign / horoscope and the personality of people with "
            "that sign — NEVER describe the literal animal, object, its biology or habitat."
        )
    if language:
        prompt += f"\n- language: {language}"
    _brand = (config.app.get("brand_voice") or "").strip()
    if _brand:
        prompt += (
            "\n\n# Brand Voice (channel persona — the whole script MUST sound like this):\n"
            f"{_brand}"
        )
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    _last_nonempty = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            # Bazı sağlayıcılar düz senaryoyu JSON dizisi olarak döndürebilir;
            # öyleyse parçaları paragraflara birleştir (yoksa format_response [...] siler).
            if response and response.strip().startswith("["):
                try:
                    _parsed = json.loads(response)
                    if isinstance(_parsed, list):
                        response = "\n\n".join(
                            str(x).strip() for x in _parsed if str(x).strip()
                        )
                except Exception:
                    pass
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            repetition_reason = script_repetition_reason(final_script)
            if repetition_reason:
                logger.warning(f"rejected repetitive generated script: {repetition_reason}")
                # Önce tekrarları temizlemeyi dene; temizlenmiş hali sorunsuzsa kullan.
                deduped = _dedupe_sentences(final_script)
                if deduped and not script_repetition_reason(deduped):
                    logger.info("repetitive script auto-cleaned (yinelenen cümleler atıldı)")
                    final_script = deduped
                    break
                # Hâlâ tekrarlıysa son çare olarak sakla ve yeniden dene.
                if deduped:
                    _last_nonempty = deduped
                final_script = ""
                prompt += (
                    "\n\n# Correction Required:\n"
                    "The previous attempt repeated phrases. Rewrite from scratch. "
                    "Every sentence must add a distinct fact. Never repeat a sentence, "
                    "example or conclusion."
                )
                continue

            # g4f may return an error message
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    # Tüm denemeler tekrar yüzünden boş geldiyse: temizlenmiş son taslağı kullan
    # (birazcık tekrarlı senaryo, HİÇ video çıkmamasından çok daha iyidir).
    if not final_script.strip() and _last_nonempty.strip():
        logger.warning("tüm denemeler tekrarlı geldi -> temizlenmiş son taslak kullanılıyor")
        final_script = _last_nonempty
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        # HER ZAMAN tekrar temizliği: 2 kez geçen yinelenen cümleler tekrar-dedektörünü
        # tetiklemese bile burada ayıklanır.
        final_script = _dedupe_sentences(final_script)
        # Meta cümleleri ayıkla ("Bu video ile...") + kapanış CTA'sını garanti et.
        final_script = _strip_meta_sentences(final_script)
        final_script = _ensure_cta(final_script)
        # HOOK SEÇİMİ: ilk cümle (kanca) izlenme oranını en çok belirleyen kısımdır.
        # 3 alternatif kanca üretip en güçlüsünü seçip ilk cümleyi onunla değiştir.
        final_script = _improve_hook(video_subject, final_script, language)
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _split_first_sentence(text: str):
    """Metni (ilk cümle, kalan) olarak böler. Cümle sonu . ? ! ile belirlenir."""
    m = re.search(r"^(.*?[.!?])(\s+)(.*)$", text.strip(), re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(3).strip()
    return text.strip(), ""


def _hook_score(hook: str) -> float:
    """Bir kancanın 'merak/çatışma' gücünü kabaca puanlar (yüksek = daha iyi)."""
    h = (hook or "").strip()
    if not h:
        return -1.0
    words = re.findall(r"\w+", h)
    n = len(words)
    score = 0.0
    if "?" in h:
        score += 2.0  # soru = güçlü merak
    if re.search(r"\d", h):
        score += 1.2  # sayı = somutluk
    # İdeal uzunluk 4-12 kelime; çok kısa/uzun cezalı.
    if 4 <= n <= 12:
        score += 1.5
    elif n > 16 or n < 3:
        score -= 1.5
    # İngilizce sızıntısı cezası (kabaca).
    if re.search(r"\b(the|you|did|know|that|your|how|why|what)\b", h.lower()):
        score -= 3.0
    # Klişe/zayıf açılış cezası.
    if re.match(r"^\s*(merhaba|selam|bu videoda|bugün)\b", h.lower()):
        score -= 1.5
    return score


def _improve_hook(video_subject: str, script: str, language: str = "") -> str:
    """3 alternatif kanca üretip en güçlüsünü seçer ve senaryonun ilk cümlesini değiştirir.

    Herhangi bir hata/şüpheli durumda orijinal senaryo aynen korunur (best-effort).
    """
    try:
        original_hook, rest = _split_first_sentence(script)
        if not rest:
            return script  # tek cümlelik senaryoya dokunma
        lang_line = f"Language: {language}\n" if language else ""
        _brand = (config.app.get("brand_voice") or "").strip()
        brand_line = f"Channel brand voice (match this tone): {_brand}\n" if _brand else ""
        prompt = (
            "# Role: Short-form Video Hook Writer\n\n"
            f"{lang_line}"
            f"{brand_line}"
            f"Video subject: {video_subject}\n"
            f"Current opening line: {original_hook}\n\n"
            "Write 3 alternative FIRST-LINE hooks for this short video. Each hook must:\n"
            "- be in the SAME language as the subject (no English if subject is not English),\n"
            "- match the channel brand voice above if provided,\n"
            "- be one short spoken sentence (4-12 words),\n"
            "- spark strong curiosity, tension or surprise (a bold claim, a vivid question, "
            "or a counter-intuitive fact),\n"
            "- NOT be clickbait-lies and NOT reference 'this video'.\n\n"
            "Return ONLY a JSON array of exactly 3 strings, nothing else."
        )
        resp = _generate_response(prompt=prompt)
        if not resp:
            return script
        raw = resp.strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return script
        candidates = json.loads(raw[start : end + 1])
        candidates = [str(c).strip() for c in candidates if str(c).strip()]
        if not candidates:
            return script
        # Orijinal kancayı da yarışa kat; en yüksek puanlıyı seç.
        pool = candidates + [original_hook]
        best = max(pool, key=_hook_score)
        if best == original_hook or _hook_score(best) <= _hook_score(original_hook):
            return script  # orijinal zaten en az kadar iyi -> dokunma
        if not re.search(r"[.!?]$", best):
            best += "."
        logger.info(f"hook upgraded: {original_hook!r} -> {best!r}")
        return f"{best} {rest}".strip()
    except Exception as e:
        logger.warning(f"hook improvement skipped: {e}")
        return script


ZODIAC_SIGNS_TR = [
    "Koç", "Boğa", "İkizler", "Yengeç", "Aslan", "Başak",
    "Terazi", "Akrep", "Yay", "Oğlak", "Kova", "Balık",
]

TOPIC_CATEGORIES = {
    "karisik": "herhangi bir popüler, geniş kitleye hitap eden alan (sağlık, psikoloji, "
    "ilginç bilgiler, tarih, bilim, uzay, para, ilişkiler, spor, doğa, teknoloji) — "
    "alanı da sen rastgele ve özgün seç",
    "burc": "astroloji ve burçlar",
    "tarih": "az bilinen, çarpıcı, merak uyandıran tarihi olaylar ve hikayeler",
    "futbol": "futbol dünyası (efsane maçlar, transferler, rekorlar, oyuncular, ilginç hikayeler)",
    "saglik": "sağlık, beslenme, uyku ve fitness ipuçları",
    "bilim": "bilim, uzay ve teknolojiden şaşırtıcı gerçekler",
    "ilginc": "şaşırtıcı ve az bilinen ilginç bilgiler",
    "para": "para, tasarruf, yatırım ve finansal ipuçları",
    "motivasyon": "motivasyon, kişisel gelişim ve alışkanlıklar",
    "psikoloji": "psikoloji ve insan davranışları üzerine merak uyandıran konular",
}


def generate_topic_idea(
    category: str = "karisik",
    language: str = "",
    sign: str = "",
    mode: str = "viral",
    avoid: List[str] = None,
) -> str:
    """Kategoriye göre TEK, taze ve viral bir kısa-video konu başlığı üretir.

    `avoid`: daha önce üretilmiş konular -> aynılarını tekrar etmemesi için modele verilir.
    `category`: TOPIC_CATEGORIES anahtarı. `sign`: burç adı (category='burc' iken).
    `mode`: 'gunluk' (günlük burç yorumu) veya 'viral'.
    """
    import datetime
    import random

    avoid = [str(a).strip() for a in (avoid or []) if str(a).strip()][-40:]
    seed = random.randint(1, 9_999_999)
    today = datetime.date.today().strftime("%d.%m.%Y")
    lang = language or "Türkçe"

    if category == "burc":
        zodiac_note = (
            " (Bu bir ASTROLOJİ BURCUDUR, hayvan/nesne değil; başlıkta mutlaka "
            f"'{sign} Burcu' ifadesi geçsin.)" if sign else ""
        )
        if sign and mode == "gunluk":
            area = (
                f"{sign} burcunun {today} tarihli GÜNLÜK burç yorumu "
                f"(bugüne özel aşk, kariyer, para).{zodiac_note}"
            )
        elif sign:
            area = (
                f"{sign} burcu ile ilgili viral, ilgi çekici bir konu "
                f"(kişilik özellikleri, uyumlu/uyumsuz burçlar, az bilinen gerçekler vb.).{zodiac_note}"
            )
        else:
            area = "astroloji ve burçlarla ilgili viral bir konu"
    else:
        area = TOPIC_CATEGORIES.get(category, category)

    avoid_block = ""
    if avoid:
        avoid_block = (
            "\n\nŞU KONULARA BENZEMEYECEK ve KESİNLİKLE TEKRAR ETMEYECEK "
            "(farklı bir açı/konu bul):\n- " + "\n- ".join(avoid)
        )

    prompt = (
        "# Rol: Viral Kısa Video Konu Üreticisi (TikTok / YouTube Shorts)\n\n"
        f"Alan/Kategori: {area}.\n"
        f"Bugünün tarihi: {today}. Çeşitlilik tohumu: {seed} "
        "(bu tohum her seferinde farklı; ona göre TAZE ve özgün bir fikir üret).\n"
        f"Dil: {lang} — konu başlığı tamamen bu dilde olmalı.\n\n"
        "GÖREV: İzlenme potansiyeli yüksek TEK bir video KONU BAŞLIĞI üret.\n"
        "Kurallar:\n"
        "- Kısa ve vurucu (en fazla 10 kelime).\n"
        "- Güçlü merak/viral potansiyeli; somut ve net olsun.\n"
        "- Uygunsa numaralı liste vaadi kullan (örn. '...5 şaşırtıcı gerçek').\n"
        "- Klişeden kaçın, özgün ol; her çağrıda belirgin biçimde FARKLI bir fikir ver.\n"
        "- Çıktıda açıklama, tırnak, numara veya etiket OLMASIN; SADECE konu başlığı."
        f"{avoid_block}"
    )

    try:
        resp = _generate_response(prompt=prompt) or ""
    except Exception as e:
        logger.warning(f"topic idea generation failed: {e}")
        return ""
    topic = resp.strip().strip('"').strip("'").split("\n")[0].strip()
    topic = re.sub(r"^(konu|başlık|topic|title)\s*[:\-–]\s*", "", topic, flags=re.IGNORECASE).strip()
    topic = topic.strip('"').strip("'").strip()
    # Burç güvenliği: başlıkta 'burç' bağlamı yoksa ekle (Oğlak=hayvan karışmasın).
    if category == "burc" and sign and topic and "burc" not in topic.casefold().replace("ç", "c"):
        topic = f"{sign} Burcu: {topic}"
    return topic[:120]


def generate_terms(video_subject: str, video_script: str, amount: int = 5) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-4 words.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
6. each term must represent a DIFFERENT concrete visual concept from the script.
7. do not return near-synonyms or minor variations of the same phrase. For example, never return traits, characteristics, personality and facts as separate terms for the same subject.
8. prefer visually searchable scenes, objects, locations or actions over abstract category labels.
9. include the main subject only where it improves relevance; do not mechanically repeat it in every term.
10. use natural, grammatically correct English search phrases. Avoid incomplete phrases such as "people meeting new".

## Output Example:
["Gemini constellation", "friends talking cafe", "creative writing desk", "new city exploration", "multiple project planning"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# Görsel üretiminde tek bir segment için kullanılacak yedek (fallback) görsel istemi son eki.
# material.py'deki zenginleştirme ile aynı tutulur.
_IMAGE_PROMPT_STYLE_SUFFIX = (
    "cinematic photography, highly detailed, dramatic lighting, "
    "professional, ultra realistic, 4k"
)
_SENSITIVE_VISUAL_TERMS = (
    "warrior",
    "soldier",
    "army",
    "battle",
    "sword",
    "weapon",
    "armor",
    "armour",
    "shield",
    "helmet",
    "mosque",
    "church",
    "temple",
    "cathedral",
    "minaret",
    "cross",
    "crusader",
    "ottoman",
    "medieval",
    "ancient costume",
    "historical costume",
)
_SENSITIVE_TOPIC_TERMS = (
    "history",
    "historical",
    "war",
    "battle",
    "soldier",
    "army",
    "weapon",
    "religion",
    "religious",
    "mosque",
    "church",
    "temple",
    "islam",
    "christian",
    "ottoman",
    "medieval",
    "conquest",
    "sultan",
    "tarih",
    "fetih",
    "fethi",
    "savaş",
    "asker",
    "silah",
    "din",
    "dini",
    "cami",
    "kilise",
    "tapınak",
    "osmanlı",
)
_STRICT_VISUAL_EXCLUSIONS = (
    "STRICT EXCLUSIONS: no text, letters, numbers, captions, logos, watermarks, "
    "or writing inside the image. Do not introduce unrelated religious, military, "
    "historical, fantasy, mythological or landmark imagery. Use an ordinary contemporary "
    "setting and keep every object directly relevant."
)


def _topic_allows_sensitive_visuals(video_subject: str) -> bool:
    subject = (video_subject or "").casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", subject)
        for term in _SENSITIVE_TOPIC_TERMS
    )


def _contains_sensitive_visuals(prompt: str) -> bool:
    candidate = (prompt or "").casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", candidate)
        for term in _SENSITIVE_VISUAL_TERMS
    )


def _safe_scene_prompt(
    segment_text: str, video_subject: str = "", visual_world: str = ""
) -> str:
    world = f" Visual world (MANDATORY, every scene belongs to it): {visual_world}." if visual_world else ""
    return (
        f"Scene for the video topic '{video_subject}'.{world} "
        f"Directly illustrate this complete narration idea: '{segment_text}', "
        "but ALWAYS keep it clearly belonging to the topic's visual world above. "
        "Avoid generic stock-photo portraits disconnected from the topic. "
        f"{_IMAGE_PROMPT_STYLE_SUFFIX}."
    )


def _sanitize_generated_image_prompt(
    candidate: str,
    segment_text: str,
    video_subject: str = "",
    visual_world: str = "",
    allow_sensitive: bool = None,
) -> str:
    """Reject prompt drift before sending a scene to the image provider.

    `allow_sensitive`: None -> konu başlığındaki anahtar kelimelere göre karar verilir
    (eski davranış). True/False verilirse (konu analizinden) o kullanılır; böylece
    Medusa/mitoloji gibi anahtar-kelime listesinde olmayan konular da doğru ele alınır.
    """
    candidate = (candidate or "").strip().strip('"').strip()
    if not candidate:
        candidate = _safe_scene_prompt(segment_text, video_subject, visual_world)

    if allow_sensitive is None:
        allows_sensitive = _topic_allows_sensitive_visuals(video_subject)
    else:
        allows_sensitive = bool(allow_sensitive)
    if not allows_sensitive and _contains_sensitive_visuals(candidate):
        logger.warning("discarded unsafe or off-topic scene prompt")
        candidate = _safe_scene_prompt(segment_text, video_subject, visual_world)

    if allows_sensitive:
        return (
            f"{candidate}. No text, captions, logos or watermarks. Keep all historical, "
            "religious or military details accurate and directly required by the narration."
        )
    return f"{candidate}. {_STRICT_VISUAL_EXCLUSIONS}"


def generate_cover_image_prompt(video_subject: str) -> str:
    """Başlığa göre tek bir profesyonel kapak arka planı sanat yönetimi istemi üretir."""
    fallback = (
        f"a contemporary editorial visual directly representing {video_subject}, one clear focal "
        "subject, polished social-media thumbnail composition, premium lighting, clean center space"
    )
    prompt = f"""
# Role: Creative Director for Short-Video Thumbnails

## Goal:
Write ONE English image-generation prompt for a premium, cinematic, vertical BACKGROUND IMAGE
that directly represents the video topic below. A separate software layer will add the headline text.

## Constraints:
1. Return ONLY the raw image prompt, nothing else.
2. Infer the correct visual domain from the topic: for example technology, finance, health,
   travel, history, psychology, astrology or education. Use domain-specific imagery only when relevant.
3. Build a clean cinematic composition: dark rich background, one strong topic-specific focal
   subject in the lower two-thirds, and calm empty negative space in the upper third (for a
   separate headline overlay). It is a photographic/illustrative SCENE, NOT a poster or flyer.
4. CRITICAL — ZERO TEXT: never describe a poster, flyer, magazine, book, sign, banner, label,
   badge, decorative border/frame with writing, or anything that implies letters. The image must
   contain NO text, letters, words, captions, numbers, logos or watermarks of any kind.
5. Do not introduce mythology, fantasy characters, warriors, weapons, historical costumes,
   battles, temples, mosques, churches or random landmarks unless the topic explicitly requires them.
6. Match the visual language to the topic. For astrology, prefer an elegant celestial SCENE with
   subtle gold zodiac/constellation motifs, deep navy or black sky and soft stars — but still a
   clean wordless image, NOT a decorated poster. For other topics, use an equally polished,
   domain-appropriate but text-free visual language.
7. Keep the prompt under 110 words.

## Video Topic:
{video_subject}
""".strip()
    try:
        response = _generate_response(prompt)
        response = response.strip().strip('"').strip()
        return _sanitize_generated_image_prompt(
            response or fallback,
            video_subject,
            video_subject,
        )
    except Exception as e:
        logger.warning(f"failed to generate cover image prompt: {str(e)}")
        return _sanitize_generated_image_prompt(fallback, video_subject, video_subject)


def _fallback_image_prompt(segment_text: str, video_subject: str = "") -> str:
    segment_text = (segment_text or "").strip()
    if not segment_text:
        segment_text = "cinematic scene"
    return _sanitize_generated_image_prompt(
        _safe_scene_prompt(segment_text, video_subject),
        segment_text,
        video_subject,
    )


def analyze_topic(video_subject: str, video_script: str = "") -> dict:
    """Konuyu anlayıp tüm sahnelerin bağlı kalacağı 'görsel dünyayı' çıkarır.

    Döner: {domain, visual_world, entities[list], allow_sensitive(bool)}.
    Görsel dünya, her sahne görseline zorunlu çapa olarak verilir; böylece "Başak +
    yalnızlık" gibi soyut cümleler bile astroloji bağlamında, "Medusa" mitoloji
    bağlamında, "Fatih'in fethi" tarihî bağlamda görselleşir — jenerik kaymaz.
    """
    fallback = {
        "domain": "",
        "visual_world": "",
        "entities": [],
        "allow_sensitive": _topic_allows_sensitive_visuals(video_subject),
    }
    if not (video_subject or "").strip():
        return fallback

    script_excerpt = (video_script or "").strip().replace("\n", " ")[:600]
    prompt = (
        "# Rol: Kısa Video Görsel Yönetmeni (Visual Director)\n\n"
        "Aşağıdaki video KONUSUNU (ve varsa senaryo özetini) analiz et ve TÜM sahne "
        "görsellerinin bağlı kalacağı tutarlı bir GÖRSEL DÜNYA tanımla. Amaç: her kare "
        "şüphesiz bu konuya ait görünsün; soyut cümleler bile konunun dünyasına bağlansın.\n\n"
        "SADECE şu JSON nesnesini döndür (başka hiçbir şey yok):\n"
        "{\n"
        '  "domain": "<kısa etiket: astroloji, tarih, mitoloji, futbol, sağlık, bilim, '
        'finans, psikoloji, doğa, teknoloji, genel ...>",\n'
        '  "visual_world": "<TEK zengin İNGİLİZCE cümle: her sahnenin ait olacağı tutarlı '
        "görsel dünya — dönem, mekan, tekrar eden görsel motifler, palet ve ruh hali. "
        "Soyut kavramları konunun dünyasına bağla (örn. astrolojide yalnızlık = yıldızlı "
        'gökyüzü altında yalnız bir siluet, hafif burç takımyıldızı).>",\n'
        '  "entities": ["<konuya özgü kilit kişi/yer/nesne, İngilizce>"],\n'
        '  "allow_sensitive": <true: konu özünde tarihî/mitolojik/dinî/askeri ise ve bu '
        "tür görseller konuyu doğru anlatmak için GEREKLİYSE; aksi halde false>\n"
        "}\n\n"
        f"KONU: {video_subject}\n"
        f"SENARYO (özet, isteğe bağlı): {script_excerpt}"
    )
    try:
        resp = _generate_response(prompt=prompt) or ""
        start, end = resp.find("{"), resp.rfind("}")
        if start == -1 or end == -1:
            return fallback
        data = json.loads(resp[start : end + 1])
        if not isinstance(data, dict):
            return fallback
        out = {
            "domain": str(data.get("domain", "")).strip(),
            "visual_world": str(data.get("visual_world", "")).strip()[:400],
            "entities": [str(e).strip() for e in (data.get("entities") or []) if str(e).strip()][:8],
            "allow_sensitive": bool(data.get("allow_sensitive", fallback["allow_sensitive"])),
        }
        logger.info(
            f"topic profile: domain={out['domain']!r}, allow_sensitive={out['allow_sensitive']}, "
            f"visual_world={out['visual_world'][:80]!r}"
        )
        return out
    except Exception as e:
        logger.warning(f"analyze_topic failed, using fallback: {e}")
        return fallback


def generate_image_prompts_for_segments(
    segments: List[str],
    video_subject: str = "",
    visual_world: str = "",
    allow_sensitive: bool = None,
) -> List[str]:
    """Her seslendirme segmenti (cümle) için Pollinations-uyumlu, İngilizce bir görsel istemi üretir.

    Tüm segmentler TEK bir LLM çağrısında işlenir; model, segment sırasına birebir
    karşılık gelen bir JSON dizisi döndürür. Başarısızlık halinde her segment için
    güvenli bir yedek istem (segment metni + sinematik son ek) kullanılır.
    Çıktı listesi DAİMA `len(segments)` uzunluğundadır.
    """
    if not segments:
        return []

    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(segments))

    # Konu analizinden gelen ZORUNLU görsel dünya çapası: her sahne buna bağlı kalır.
    if visual_world:
        world_block = (
            "## Visual World (MANDATORY — applies to EVERY image):\n"
            f"{visual_world}\n"
            "Every single prompt MUST visibly belong to this world and to the video subject. "
            "Even abstract narration (feelings, concepts, connectors) must be expressed THROUGH "
            "this world's motifs/setting — never as a generic, topic-less stock photo.\n\n"
        )
    else:
        world_block = ""

    # Hassas görsel kuralı: konu analizine göre (tarih/mitoloji/din/askeri uygun mu).
    if allow_sensitive:
        rule_9 = (
            "9. The topic REQUIRES authentic historical / mythological / religious / period "
            "imagery — use it faithfully and accurately (correct era, costumes, setting, "
            "architecture, artifacts). Do NOT sanitize it into a generic modern scene."
        )
    else:
        rule_9 = (
            "9. Never add unrelated mythology, fantasy characters, warriors, weapons, historical "
            "costumes, battles, temples, mosques, churches, religious buildings or random landmarks. "
            "Use those elements only when the narration or overall topic explicitly requires them."
        )

    prompt = f"""
# Role: Image Prompt Generator for Short-Video Scenes

## Goal:
For EACH numbered narration segment below, write ONE vivid English image-generation prompt
that directly represents the exact meaning of that segment WHILE staying inside the mandatory
visual world below, so every image unmistakably belongs to the video subject.

{world_block}## Constraints:
1. Return ONLY a JSON array of strings, nothing else.
2. The array length MUST equal the number of segments, in the SAME order.
3. Each prompt must be in ENGLISH, vivid and detailed, including subject, mood, lighting and composition.
4. Each prompt must depict the meaning of its own segment, not a generic portrait.
5. CONSISTENT STYLE: keep ONE coherent visual language across ALL prompts — the same
   photographic style, color palette, lighting mood and lens feel. State this shared style
   briefly in every prompt (e.g. "cinematic editorial photography, warm natural light,
   shallow depth of field, muted modern palette") so the whole video feels like one piece.
6. VISUAL VARIETY: consecutive prompts MUST be visually DISTINCT from each other. Never
   repeat the same scene, subject, pose or setting in adjacent prompts. Vary the subject,
   environment, camera angle and composition each time, even when two segments are about
   the same idea. A short connector segment (e.g. "Secondly", "İkincisi") must share ONE
   distinct scene with the explanation that follows it — do not invent a near-duplicate.
7. Prefer contemporary editorial visuals and clear symbolic details.
8. For astrology or zodiac topics, use modern celestial motifs, constellations and subtle zodiac symbolism.
{rule_9}
10. CRITICAL — ZERO TEXT: the image must contain NO text of any kind — no letters, words,
    captions, labels, numbers, signs, logos, watermarks or typography. To guarantee this,
    NEVER depict text-bearing objects: no books/pages with visible writing, no charts,
    diagrams, infographics, posters, signs, screens with UI, newspapers, packaging or
    bottles with labels. Replace any such idea with a clean, wordless real-world visual
    (e.g. instead of "a book of rules" use "fresh colorful ingredients arranged on a table").
11. Keep each prompt under 70 words. Do not number the prompts, just the raw strings.

## Output Example (for 2 segments):
["a confident young professional speaking naturally with friends in a modern cafe, warm editorial lighting, candid composition", "two elegant human silhouettes connected by subtle constellation lines against a sophisticated midnight-blue celestial background, premium editorial style"]

## Context:
### Video Subject
{video_subject}

### Narration Segments
{numbered}
""".strip()

    n = len(segments)
    prompts: List[str] = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate image prompts: {response}")
                break
            prompts = json.loads(response)
            if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
                break
        except Exception as e:
            logger.warning(f"failed to parse image prompts: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        prompts = json.loads(match.group())
                        if isinstance(prompts, list):
                            break
                    except Exception as e2:
                        logger.warning(f"failed to parse image prompts json: {str(e2)}")
        if i < _max_retries:
            logger.warning(f"retry image prompts... {i + 1}")

    if not isinstance(prompts, list):
        prompts = []

    # Çıktı uzunluğunu segment sayısına eşitle: eksikleri yedek istemle doldur,
    # fazlaları kırp. Boş gelen istemleri de yedekle değiştir.
    # Ayrıca her sahneye DÖNÜŞÜMLÜ bir çekim tipi ekle: konu benzer olsa bile
    # ardışık kareler farklı kompozisyonda (geniş/yakın/tepeden/alçak açı) çıksın
    # -> "aynı görsel tekrarı" hissi kırılır.
    _SHOTS = [
        "cinematic wide establishing shot",
        "extreme close-up macro detail",
        "dramatic low-angle shot",
        "overhead top-down flat-lay view",
        "medium side-profile shot with shallow depth of field",
        "dynamic over-the-shoulder perspective",
    ]
    normalized: List[str] = []
    for idx in range(n):
        candidate = prompts[idx] if idx < len(prompts) else ""
        candidate = candidate.strip() if isinstance(candidate, str) else ""
        _clean = _sanitize_generated_image_prompt(
            candidate,
            segments[idx],
            video_subject,
            visual_world=visual_world,
            allow_sensitive=allow_sensitive,
        )
        normalized.append(f"{_clean} | {_SHOTS[idx % len(_SHOTS)]}")

    logger.success(f"generated {len(normalized)} segment image prompts")
    return normalized


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
    
