# app.py
import os
import re
import time
import json
import tempfile
import subprocess
from typing import List, Optional
from urllib.parse import urlparse, parse_qs
from xml.etree.ElementTree import ParseError

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

try:
    from youtube_transcript_api import TooManyRequests  # type: ignore
except Exception:
    try:
        from youtube_transcript_api._errors import TooManyRequests  # type: ignore
    except Exception:
        class TooManyRequests(Exception):  # fallback shim
            """Compatibility shim when the package doesn't provide TooManyRequests."""
            pass

#  Config 
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY not set. Put it in a .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=API_KEY)
STABLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]
DEFAULT_LANG_PREF = ["en", "en-US", "en-GB", "hi", "en-IN"]

PROMPT_PREFIX = (
    "You're a YouTube video summariser. Summarise the transcript in 200–250 words "
    "as bullet points with key ideas and facts.\n\nTranscript:\n"
)

#Helpers 
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
        info["manual"] = [t.language_code for t in getattr(ts, "_manually_created_transcripts", {}).values()]
        info["generated"] = [t.language_code for t in getattr(ts, "_generated_transcripts", {}).values()]
    except Exception as e:
        info["error"] = str(e)
    return info

def _strip_vtt_to_text(vtt_text: str) -> str:
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

import sys

def _run_ytdlp(cmd: list) -> subprocess.CompletedProcess:
    """Run yt-dlp; if binary missing on PATH, run the module with THIS Python."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return subprocess.run(
            [sys.executable, "-m", "yt_dlp", *cmd[1:]],
            capture_output=True, text=True, check=False
        )


def ytdlp_list_subs(video_id: str, cookies_path: Optional[str] = None) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ["yt-dlp", "--skip-download", "--dump-json", url, "--quiet", "--no-warnings"]
    if cookies_path:
        cmd.extend(["--cookies", cookies_path])
    proc = _run_ytdlp(cmd)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        info = json.loads(proc.stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    out = {"manual": {}, "auto": {}}
    for kind in ("subtitles", "automatic_captions"):
        if kind in info and isinstance(info[kind], dict):
            bucket = "manual" if kind == "subtitles" else "auto"
            for lang, entries in info[kind].items():
                exts = sorted({e.get("ext") for e in entries if isinstance(e, dict) and "ext" in e})
                if exts:
                    out[bucket][lang] = exts
    return out
def _choose_best_lang(available: dict, prefs: List[str]) -> Optional[tuple]:
    manual = available.get("manual", {})
    auto = available.get("auto", {})
    for p in prefs:
        if p in manual: return ("manual", p)
        if p in auto:   return ("auto", p)
    if any(p.startswith("en") for p in prefs):
        for k in manual:
            if k.startswith("en"): return ("manual", k)
        for k in auto:
            if k.startswith("en"): return ("auto", k)
    if manual:
        k = sorted(manual.keys())[0]; return ("manual", k)
    if auto:
        k = sorted(auto.keys())[0]; return ("auto", k)
    return None
def fetch_transcript_text_ytdlp(
    video_id: str, prefs: List[str], cookies_path: Optional[str] = None
) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    avail = ytdlp_list_subs(video_id, cookies_path=cookies_path)
    if not avail or (not avail.get("manual") and not avail.get("auto")):
        cand_langs = [*prefs, "en", "en-US", "en-GB"]
        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
            for l in cand_langs:
                for flag in (["--write-subs"], ["--write-auto-subs"]):
                    cmd = ["yt-dlp", url, *flag, "--sub-lang", l,
                           "--skip-download", "--sub-format", "vtt", "-o", outtmpl,
                           "--quiet", "--no-warnings"]
                    if cookies_path: cmd.extend(["--cookies", cookies_path])
                    proc = _run_ytdlp(cmd)
                    if proc.returncode != 0 and proc.stderr:
                        st.caption(f"yt-dlp stderr: {proc.stderr[:4000]}")
                    vtts = [p for p in os.listdir(tmpdir) if p.endswith(".vtt")]
                    if vtts:
                        with open(os.path.join(tmpdir, vtts[0]), "r", encoding="utf-8") as f:
                            return _strip_vtt_to_text(f.read())
        return ""
    choice = _choose_best_lang(avail, prefs)
    if not choice:
        return ""
    bucket, lang = choice
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        flags = ["--write-subs"] if bucket == "manual" else ["--write-auto-subs"]
        cmd = ["yt-dlp", url, *flags, "--sub-lang", lang,
               "--skip-download", "--convert-subs", "vtt", "-o", outtmpl,
               "--quiet", "--no-warnings"]
        if cookies_path:
            cmd.extend(["--cookies", cookies_path])
        proc = _run_ytdlp(cmd)
        if proc.returncode != 0 and proc.stderr:
            st.caption(f"yt-dlp stderr: {proc.stderr[:4000]}")
        vtts = [p for p in os.listdir(tmpdir) if p.endswith(".vtt")]
        if not vtts:
            return ""
        with open(os.path.join(tmpdir, vtts[0]), "r", encoding="utf-8") as f:
            return _strip_vtt_to_text(f.read())

def fetch_transcript_text_api(
    video_id: str,
    languages: List[str],
    cookies_path: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            data = YouTubeTranscriptApi.get_transcript(
                video_id, languages=languages, cookies=cookies_path
            )
            txt = " ".join(c.get("text", "") for c in data).strip()
            if txt:
                return txt
        except NoTranscriptFound:
            pass
        except TooManyRequests:
            time.sleep(2 * (attempt + 1)); continue
        except (CouldNotRetrieveTranscript, ValueError, ParseError):
            time.sleep(1.5 * (attempt + 1)); continue
        try:
            ts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)
            try:
                t = ts.find_transcript(languages)
            except Exception:
                t = next(iter(getattr(ts, "_manually_created_transcripts", {}).values()), None) \
                    or next(iter(getattr(ts, "_generated_transcripts", {}).values()), None)
                if t is None:
                    raise NoTranscriptFound(video_id, languages, {})
            txt = " ".join(c.get("text", "") for c in t.fetch()).strip()
            if txt:
                return txt
        except TooManyRequests:
            time.sleep(2 * (attempt + 1)); continue
        except (CouldNotRetrieveTranscript, ValueError, ParseError):
            time.sleep(1.5 * (attempt + 1)); continue
        except NoTranscriptFound:
            break
    try:
        ts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path)
        t = next(iter(getattr(ts, "_manually_created_transcripts", {}).values()), None) \
            or next(iter(getattr(ts, "_generated_transcripts", {}).values()), None)
        if t:
            txt = " ".join(c.get("text", "") for c in t.fetch()).strip()
            if txt:
                return txt
    except Exception:
        pass
    raise NoTranscriptFound(video_id, languages, {})
def summarize_with_gemini(model_name: str, transcript_text: str) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(PROMPT_PREFIX + transcript_text)
    return resp.text

# UI 
st.title("YouTube Transcript to Detailed Notes Converter")

with st.sidebar:
    st.subheader("Settings")
    model_name = st.selectbox("Gemini model", STABLE_MODELS, index=0)
    lang_pref = st.multiselect(
        "Preferred caption languages (order matters)",
        options=[
            "en","en-US","en-GB","hi","en-IN","es","fr","de","it","pt","ru","zh",
            "ja","ko","ar","tr","vi","id","bn","ta","te","mr"
        ],
        default=DEFAULT_LANG_PREF,
    )
    st.markdown("**(Optional)** Pass YouTube cookies for age/region-restricted videos.")
    cookies_mode = st.radio("Provide cookies via", ["None", "Upload file", "Path on disk"], index=0)
    cookies_path = None
    if cookies_mode == "Upload file":
        up = st.file_uploader("Upload Netscape-format cookies file", type=["txt","cookies"])
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            tmp.write(up.read()); tmp.flush()
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

with st.expander("Or paste a transcript manually"):
    manual_txt = st.text_area("Paste transcript text here")
    if st.button("Summarize pasted transcript"):
        if manual_txt.strip():
            st.markdown("## Detailed Notes:")
            st.write(summarize_with_gemini(model_name, manual_txt.strip()))
        else:
            st.warning("Please paste some transcript text.")

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
            vid, prefs=(lang_pref or ["en"]), cookies_path=cookies_path
        )

    if not text:
        st.error(
            "Could not retrieve any captions. Common causes: Shorts, some music videos, "
            "or heavily restricted videos. Provide cookies, try another language, or paste a transcript manually."
        )
    else:
        summary = summarize_with_gemini(model_name, text)
        st.markdown("## Detailed Notes:")
        st.write(summary)
