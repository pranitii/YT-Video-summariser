# app.py
import os
import re
import time
import tempfile
import subprocess
from typing import List, Optional
from urllib.parse import urlparse, parse_qs
from xml.etree.ElementTree import ParseError

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, NotFound

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    TooManyRequests,
    CouldNotRetrieveTranscript,
)

# ---------------- Config ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY not set. Put it in a .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=API_KEY)

DEFAULT_LANG_PREF = ["en", "en-US", "en-GB", "hi", "en-IN"]

PROMPT_PREFIX = (
    "You're a YouTube video summariser. Summarise the transcript in 200–250 words "
    "as bullet points with key ideas and facts.\n\nTranscript:\n"
)

# ---------------- Gemini helpers ----------------
@st.cache_resource(show_spinner=False)
def get_available_text_models() -> list[str]:
    """
    Query the API for all models that support generateContent with this key.
    We will show only these in the sidebar so we never pick an invalid id.
    """
    try:
        models = genai.list_models()
        names = [
            m.name
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        # Prefer a stable order: flash-ish first, then others
        names.sort(key=lambda n: ("flash" not in n, "pro" not in n, n))
        return names or ["models/gemini-1.5-flash"]
    except Exception:
        # Fallback if list_models fails for some reason
        return ["models/gemini-1.5-flash"]


def summarize_with_gemini(model_name: str, transcript_text: str) -> str:
    """
    Call Gemini with the chosen model. If quota is exhausted, surface a clear error.
    """
    st.caption(f"Using Gemini model: `{model_name}`")
    model = genai.GenerativeModel(model_name)

    try:
        resp = model.generate_content(PROMPT_PREFIX + transcript_text)
        return (resp.text or "").strip()
    except NotFound:
        st.error(
            f"The selected model (`{model_name}`) is not available for this API key. "
            "Please go to the sidebar and pick one of the listed models again. "
            "If none work, check your Google AI Studio / API key configuration."
        )
        return ""
    except ResourceExhausted:
        st.error(
            "Gemini quota is exhausted for this model / project. "
            "Try again later or switch to a lighter model (e.g. a flash / 8b variant) "
            "in Google AI Studio."
        )
        return ""
    except Exception as e:
        st.error(f"Unexpected Gemini error: {e}")
        return ""


# ---------------- Transcript helpers ----------------
def extract_video_id(url: str) -> str:
    """Support watch?v=ID, youtu.be/ID, /embed/ID, /shorts/ID, and raw 11-char IDs."""
    if not url:
        return ""
    part = url.strip().split("&")[0]
    if len(part) == 11 and "/" not in part and " " not in part:
        return part

    p = urlparse(url)
    if p.netloc.endswith("youtu.be"):
        return p.path.lstrip("/").split("?")[0]
    if "youtube.com" in p.netloc:
        qs = parse_qs(p.query)
        if "v" in qs:
            return qs["v"][0]
        parts = [s for s in p.path.split("/") if s]
        if parts:
            return parts[-1].split("?")[0]
    return ""


def list_available_transcripts(video_id: str, cookies_path: Optional[str] = None):
    info = {"manual": [], "generated": []}
    try:
        ts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)
        info["manual"] = [t.language_code for t in ts._manually_created_transcripts.values()]
        info["generated"] = [t.language_code for t in ts._generated_transcripts.values()]
    except Exception as e:
        info["error"] = str(e)
    return info


def fetch_transcript_text_api(
    video_id: str,
    languages: List[str],
    cookies_path: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Primary path: youtube-transcript-api with retries and language negotiation.
    Normalize empty/HTML responses to NoTranscriptFound so caller can fallback.
    """
    for attempt in range(max_retries):
        try:
            data = YouTubeTranscriptApi.get_transcript(
                video_id, languages=languages, cookies=cookies_path
            )
            txt = " ".join(c["text"] for c in data).strip()
            if txt:
                return txt
        except NoTranscriptFound:
            pass
        except TooManyRequests:
            time.sleep(2 * (attempt + 1))
            continue
        except (CouldNotRetrieveTranscript, ValueError, ParseError):
            time.sleep(1.5 * (attempt + 1))
            continue

        # Try via list_transcripts
        try:
            ts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)
            try:
                t = ts.find_transcript(languages)
            except Exception:
                t = next(iter(ts._manually_created_transcripts.values()), None) \
                    or next(iter(ts._generated_transcripts.values()), None)
                if t is None:
                    raise NoTranscriptFound(video_id, languages, {})
            txt = " ".join(c["text"] for c in t.fetch()).strip()
            if txt:
                return txt
        except TooManyRequests:
            time.sleep(2 * (attempt + 1))
            continue
        except (CouldNotRetrieveTranscript, ValueError, ParseError):
            time.sleep(1.5 * (attempt + 1))
            continue
        except NoTranscriptFound:
            break

    raise NoTranscriptFound(video_id, languages, {})


def _strip_vtt_to_text(vtt_text: str) -> str:
    """Convert WebVTT to plain text."""
    lines = []
    for line in vtt_text.splitlines():
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}", line):
            continue
        if line.strip().isdigit():
            continue
        lines.append(line.strip())
    return " ".join(lines).strip()


def _run_ytdlp(cmd: list) -> subprocess.CompletedProcess:
    """Run yt-dlp; if binary missing on PATH, try 'python -m yt_dlp'."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return subprocess.run(
            ["python", "-m", "yt_dlp", *cmd[1:]],
            capture_output=True,
            text=True,
        )


def fetch_transcript_text_ytdlp(
    video_id: str, lang: str = "en", cookies_path: Optional[str] = None
) -> str:
    """
    Fallback: use yt-dlp to download auto-subs as VTT and parse to text.
    Requires: pip install yt-dlp
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            url,
            "--write-auto-subs",
            "--sub-lang",
            lang,
            "--skip-download",
            "--sub-format",
            "vtt",
            "-o",
            outtmpl,
            "--quiet",
            "--no-warnings",
        ]
        if cookies_path:
            cmd.extend(["--cookies", cookies_path])

        _ = _run_ytdlp(cmd)

        vtt_path = os.path.join(tmpdir, f"{video_id}.{lang}.vtt")
        if not os.path.exists(vtt_path):
            candidates = [p for p in os.listdir(tmpdir) if p.endswith(".vtt")]
            if not candidates:
                return ""
            vtt_path = os.path.join(tmpdir, candidates[0])

        with open(vtt_path, "r", encoding="utf-8") as f:
            vtt_text = f.read()
        return _strip_vtt_to_text(vtt_text)


# ---------------- UI ----------------
st.title("YouTube Transcript to Detailed Notes Converter")

available_models = get_available_text_models()
with st.sidebar:
    st.subheader("Settings")
    st.caption("Models detected for this API key:")
    model_name = st.selectbox("Gemini model", available_models, index=0)
    lang_pref = st.multiselect(
        "Preferred caption languages (order matters)",
        options=[
            "en", "en-US", "en-GB", "hi", "en-IN", "es", "fr", "de", "it", "pt", "ru",
            "zh", "ja", "ko", "ar", "tr", "vi", "id", "bn", "ta", "te", "mr",
        ],
        default=DEFAULT_LANG_PREF,
    )
    st.markdown("**(Optional)** Pass YouTube cookies for age/region-restricted videos.")
    cookies_mode = st.radio(
        "Provide cookies via", ["None", "Upload file", "Path on disk"], index=0
    )
    cookies_path = None
    if cookies_mode == "Upload file":
        up = st.file_uploader(
            "Upload Netscape-format cookies file", type=["txt", "cookies"]
        )
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            tmp.write(up.read())
            tmp.flush()
            cookies_path = tmp.name
            st.caption(f"Using uploaded cookies: {cookies_path}")
    elif cookies_mode == "Path on disk":
        cookies_path = st.text_input("Absolute path to cookies file", value="")
        if cookies_path:
            st.caption(f"Using cookies: {cookies_path}")

youtube_link = st.text_input("Enter YouTube video link:")
vid = extract_video_id(youtube_link) if youtube_link else ""
if vid:
    st.image(f"https://img.youtube.com/vi/{vid}/0.jpg", use_container_width=True)

# Manual transcript path
with st.expander("Or paste a transcript manually"):
    manual_txt = st.text_area("Paste transcript text here")
    if st.button("Summarize pasted transcript"):
        if manual_txt.strip():
            summary = summarize_with_gemini(model_name, manual_txt.strip())
            if summary:
                st.markdown("## Detailed Notes:")
                st.write(summary)
        else:
            st.warning("Please paste some transcript text.")

# Automatic transcript path
if st.button("Get Detailed Notes from YouTube") and vid:
    dbg = list_available_transcripts(vid, cookies_path=cookies_path)
    st.caption(f"Available transcripts: {dbg}")

    text = ""
    try:
        text = fetch_transcript_text_api(
            vid, languages=lang_pref, cookies_path=cookies_path, max_retries=3
        )
    except (NoTranscriptFound, ParseError, TypeError) as e:
        st.warning(f"API path failed: {e}")
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except TooManyRequests:
        st.error("YouTube is rate-limiting transcript requests (HTTP 429). Try again later.")
    except VideoUnavailable:
        st.error("The video is unavailable or age/region/members restricted.")
    except Exception as e:
        st.warning(f"API path failed: {e}")

    if not text:
        st.warning("Falling back to yt-dlp auto-captions…")
        text = fetch_transcript_text_ytdlp(
            vid, lang=(lang_pref[0] if lang_pref else "en"), cookies_path=cookies_path
        )

    if not text:
        st.error(
            "Could not retrieve any captions. Common causes: Shorts, some music videos, "
            "or heavily restricted videos. Provide cookies, try another language, or paste a transcript manually."
        )
    else:
        summary = summarize_with_gemini(model_name, text)
        if summary:
            st.markdown("## Detailed Notes:")
            st.write(summary)
