#!/usr/bin/env python3
"""
JINGLU v29 - Asterisk AudioSocket <-> Sarvam (STT/TTS) <-> Groq (LLM) voice agent.

Dialplan:  AudioSocket(${LEADID},127.0.0.1:9092)

Hard-won lessons baked in:
  v2  reader thread ALWAYS drains the socket (Asterisk pushes a frame every
      20 ms; if we stop reading while we talk its writes block).
  v3  writer thread ALWAYS sends a frame every 20 ms - silence when we have
      nothing. app_audiosocket.c: "Reached timeout after 2000 ms of no
      activity" kills the call otherwise, and STT+LLM+TTS take 5-8 s a turn.
  v4  opening greeting is pre-rendered at startup  -> talk ~0.2 s after answer
      replies are TTS'd sentence-by-sentence       -> first words play sooner
      10 s of caller silence                       -> polite goodbye + hang up
      User-Agent set on every call (Groq/Cloudflare 403s bare urllib)
  v5  language is detected ONCE then LOCKED - "unknown" per turn made Sarvam
      flip te->bn->ta->gu->bn on one call and Jinglu replied in Bengali to a
      Coimbatore hotel owner. The locked code is also fed back to STT, which
      lifts transcript quality on 8 kHz audio.
      LLM errors no longer kill the call (a Groq 429 used to hang up mid-turn).

Wire format: [1B type][2B len BE][payload]
  0x00 terminate | 0x01 uuid(16B) | 0x03 dtmf | 0x10 audio | 0xff error
  audio = signed linear 16-bit LE, 8 kHz, mono, 320 B = 20 ms
"""
import base64, io, json, logging, os, queue, re, socket, struct, sys, threading, time, unicodedata, uuid, wave
import numpy as np
from collections import deque
import urllib.request
import sqlite3, urllib.error
from datetime import datetime, timedelta, timezone

SUPA       = os.environ.get("SUPABASE_URL", "https://xuxkfjiaoifnfugwkbpj.supabase.co")
SUPA_KEY   = os.environ.get("SUPABASE_KEY", "")
SARVAM_KEY = os.environ.get("SARVAM_KEY", "")
GROQ_KEY   = os.environ.get("GROQ_KEY", "")

LISTEN_HOST = os.environ.get("JINGLU_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("JINGLU_PORT", "9092"))
PROMPT_FILE = os.environ.get("JINGLU_PROMPT", "/etc/ailc/jinglu_prompt.md")
CRM_URL     = os.environ.get("JINGLU_CRM_URL", "http://127.0.0.1:5999/api/calls/report")

CRM_SCRIPT_URL = os.environ.get("JINGLU_CRM_SCRIPT",
                                "http://127.0.0.1:5999/api/ai/script")
DB_PATH = os.environ.get("JINGLU_DB_PATH",
                          "/var/www/hotel-lead-crm/prisma/prod.db")  # v36: read-only, for greeting pre-render


_SETTINGS_TENANT_ID = "cmsf0p7az0000lvypq8v2r3o4"  # microgenn (the live production tenant)

def _refresh_settings_from_db():
    """v69: pull tenant-scoped settings from the CRM website's Setting table
    and hot-apply them to this process before the call starts. Runs once per
    call, from the top of Call.__init__. Wrapped so ANY failure (DB missing,
    locked, bad value, whatever) just leaves every existing value exactly as
    it was -- this function must never be able to break a call.

    Deliberately excludes callTtsModel/callTtsSpeaker: prerender() bakes the
    TTS voice into fixed cached audio at boot, so changing those here mid-run
    would desync cached phrases from a live voice. Those two stay restart-only.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
        cur = conn.cursor()
        cur.execute(
            "SELECT tenantId, key, value FROM Setting WHERE tenantId IS NULL OR tenantId=?",
            (_SETTINGS_TENANT_ID,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        try:
            log.warning(f"_refresh_settings_from_db: DB read failed, keeping current values: {e}")
        except Exception:
            pass
        return

    vals = {}
    for tid, k, v in rows:
        if tid is None:
            vals.setdefault(k, v)
    for tid, k, v in rows:
        if tid is not None:
            vals[k] = v  # tenant-specific overrides win over global defaults

    def _num(key, caster, cur_val):
        v = vals.get(key)
        if v is None or str(v).strip() == "":
            return cur_val
        try:
            return caster(v)
        except Exception:
            return cur_val

    def _str(key, cur_val):
        v = vals.get(key)
        if v is None or str(v).strip() == "":
            return cur_val
        return v

    global SILENCE_RMS, NOISE_MULT, NOISE_MAX, MIN_VOICE_FRAMES, VOICE_RELAX_MIN, VOICE_RATIO
    global SILENCE_MS, DEAD_AIR_MS, SETTLE_MS, CALIB_MS, DEFAULT_LANG, MAX_TOKENS, SPEAK_MAX
    global MIN_FIRST_CHARS, LANG_Q, LANG_TRIES, STT_MODEL, LLM_MODEL, GROQ_KEY, DECLINE_MARKERS
    try:
        SILENCE_RMS = _num("callSilenceRms", int, SILENCE_RMS)
        NOISE_MULT = _num("callNoiseMult", float, NOISE_MULT)
        NOISE_MAX = _num("callNoiseMax", int, NOISE_MAX)
        MIN_VOICE_FRAMES = _num("callMinVoiceFrames", int, MIN_VOICE_FRAMES)
        VOICE_RATIO = _num("callVoiceRatio", float, VOICE_RATIO)
        VOICE_RELAX_MIN = _num("callVoiceRelaxMin", float, VOICE_RELAX_MIN)
        SILENCE_MS = _num("callSilenceMs", int, SILENCE_MS)
        DEAD_AIR_MS = _num("callDeadAirMs", int, DEAD_AIR_MS)
        SETTLE_MS = _num("callSettleMs", int, SETTLE_MS)
        CALIB_MS   = max(400, min(1000, _num("callCalibMs", int, CALIB_MS)))  # v80: safety-clamped
        DEFAULT_LANG = _str("callDefaultLang", DEFAULT_LANG)
        MAX_TOKENS = _num("callMaxTokens", int, MAX_TOKENS)
        SPEAK_MAX = _num("callSpeakMax", int, SPEAK_MAX)
        MIN_FIRST_CHARS = _num("callMinFirstChars", int, MIN_FIRST_CHARS)
        LANG_Q = _str("callLangQ", LANG_Q)
        LANG_TRIES = _num("callLangTries", int, LANG_TRIES)
        _decline_raw = _str("callDeclineMarkers", None)
        if _decline_raw:
            _decline_parsed = tuple(p.strip().lower() for p in _decline_raw.split(",") if p.strip())
            if _decline_parsed:
                DECLINE_MARKERS = _decline_parsed
        STT_MODEL = _str("callSttModel", STT_MODEL)
        LLM_MODEL = _str("callLlmModel", LLM_MODEL)
        GROQ_KEY = _str("callLlmGroqKey", GROQ_KEY)

        _provider = _str("callLlmProvider", None)
        if _provider:
            os.environ["JINGLU_LLM_PROVIDER"] = _provider
        _okey = _str("callLlmOpenaiKey", None)
        if _okey:
            os.environ["OPENAI_KEY"] = _okey
        _omodel = _str("callLlmOpenaiModel", None)
        if _omodel:
            os.environ["JINGLU_LLM_OPENAI"] = _omodel
        _gkey = _str("callLlmGeminiKey", None)
        if _gkey:
            os.environ["GEMINI_KEY"] = _gkey
        _gmodel = _str("callLlmGeminiModel", None)
        if _gmodel:
            os.environ["JINGLU_LLM_GEMINI"] = _gmodel
    except Exception as e:
        try:
            log.warning(f"_refresh_settings_from_db: apply failed, some values may be stale: {e}")
        except Exception:
            pass


def fetch_crm_script(lead_id):
    """v32: the CRM decides which script calls this lead, from its category -
    Hotels leads get the Hotels script, Sports leads the Sports one. Any
    failure returns None and we keep the file prompt: a live call is worth
    more than the perfect script."""
    try:
        req = urllib.request.Request("%s?uuid=%s" % (CRM_SCRIPT_URL, lead_id))
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log.info("CRM script lookup failed (%s) - using the file prompt", e)
        return None


def load_active_scripts():
    """v36: read every active CallScript directly from the CRM's DB so we can
    pre-render each one's own greeting at boot. Read-only, best-effort - an
    empty list just means calls fall back to the generic greeting.
    v51: [ ] in the greeting is a placeholder for the script's agentName -
    substitute it here so the pre-rendered audio says the real name."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            rows = conn.execute(
                "select id, greeting, agentName from CallScript where active=1").fetchall()
        finally:
            conn.close()
        return [(r[0], (r[1] or "").replace("[ ]", r[2] or "Jinglu"))
                for r in rows if r[1]]
    except Exception as e:
        log.warning("could not read CallScript table for pre-render: %s", e)
        return []



TTS_MODEL   = os.environ.get("JINGLU_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.environ.get("JINGLU_SPEAKER", "shruti")   # v7: the one Dhina picked
STT_MODEL   = os.environ.get("JINGLU_STT_MODEL", "saaras:v3")
LLM_MODEL   = os.environ.get("JINGLU_LLM", "llama-3.3-70b-versatile")

RATE, FRAME_BYTES = 8000, 320
# v15: SILENCE_RMS alone is a FIXED gate, and a hotel lobby, a road, or a TV in
# the room sits well above it - we were feeding that straight to STT as "speech"
# and getting garbage transcripts back. So it is now only the FLOOR of the gate:
# we measure the room's own noise continuously and demand that real speech stand
# NOISE_MULT times above it. NOISE_MAX stops a very loud line making us deaf.
SILENCE_RMS   = int(os.environ.get("JINGLU_SILENCE_RMS", "500"))
NOISE_MULT    = float(os.environ.get("JINGLU_NOISE_MULT", "2.5"))
ECHO_GRACE_MS = int(os.environ.get("JINGLU_ECHO_GRACE_MS", "300"))  # v37: freeze noise-learning this long after our own TTS ends
NOISE_MAX     = int(os.environ.get("JINGLU_NOISE_MAX", "2500"))
NOISE_INIT    = int(os.environ.get("JINGLU_NOISE_INIT", "150"))
TRIM_PAD      = int(os.environ.get("JINGLU_TRIM_PAD", "12"))
# v26: measured - "heard 0.7s speech in 12.3s" then "stt for 12.1s audio".
# Sporadic above-gate blips kept extending the voice span so the trim
# covered the whole window. Nobody answers a yes/no question for eight
# seconds; past that it is mostly room, and his ANSWER is at the END.
TRIM_MAX_FRAMES = int(os.environ.get("JINGLU_TRIM_MAX_FRAMES", "1000")) # 20.0s  # v37: was 8s, cut off long customer replies mid-sentence
# v19: a door, a click, a horn crosses the gate for one or two 20ms frames.
# Human speech does not - the shortest syllable runs 100ms+. So a lone spike
# is noise, and must not start a turn or extend one. This is what stopped us
# shipping 14s of room to STT to get back the word "Hello".
MIN_VOICE_FRAMES = int(os.environ.get("JINGLU_MIN_VOICE_FRAMES", "5"))   # 100ms
# v24: the noise floor alone is not enough. A real call logged "noise=8
# gate=500" - the line was near silent so the adaptive part did nothing and the
# gate sat on its floor. Someone talking across the room reads 600-1500 RMS,
# clears 500, and 6.5s of room went to STT to come back as garbage. The only
# reliable reference is the CALLER'S OWN voice - the man on the handset is far
# louder than the room. Median of his frames, not max: one loud word must not
# deafen us to him for the rest of the call.
VOICE_RATIO = float(os.environ.get("JINGLU_VOICE_RATIO", "0.30"))
# Asymmetric like an AGC: believe a DROP in his voice fast (follow him down or
# go deaf), a RISE slowly (one shout must not lock out ten quiet answers).
VOICE_EMA_UP   = float(os.environ.get("JINGLU_VOICE_EMA_UP", "0.35"))
VOICE_EMA_DOWN = float(os.environ.get("JINGLU_VOICE_EMA_DOWN", "0.70"))
# A gate learned only upward is a trap - we learn from frames that already
# passed it, so once he is under it he can never get back. When nothing clears
# it the gate stands down, but never below VOICE_RELAX_MIN of his level (the
# room lives under that). Re-arms once vlevel actually reflects him again.
VOICE_RELAX_MS  = int(os.environ.get("JINGLU_VOICE_RELAX_MS", "600"))
VOICE_RELAX_MIN = float(os.environ.get("JINGLU_VOICE_RELAX_MIN", "0.35"))
# v6: 900 -> 650, v8: -> 500. Pure dead waiting after they stop talking, paid
# on every single turn before STT even starts. 500ms still clears a normal
# mid-sentence breath; below that we start cutting people off.
SILENCE_MS    = int(os.environ.get("JINGLU_SILENCE_MS", "500"))
BARGE_RMS     = int(os.environ.get("JINGLU_BARGE_RMS", "1200"))
PREROLL_FRAMES  = int(os.environ.get("JINGLU_PREROLL_FRAMES", "20"))  # v85: ~400ms of pre-barge audio kept (20ms/frame)
BARGE_FRAMES  = 10
MIN_SPEECH_MS = 300
# v26: 20s. After the closing line the caller says nothing and we waited out
# the full turn AND the dead-air timer on top of it.
MAX_TURN_MS   = int(os.environ.get("JINGLU_MAX_TURN_MS", "12000"))
DEAD_AIR_MS   = int(os.environ.get("JINGLU_DEAD_AIR_MS", "10000"))   # v4: 10 s -> close
MAX_CALL_SEC  = int(os.environ.get("JINGLU_MAX_CALL", "240"))
SETTLE_MS     = int(os.environ.get("JINGLU_SETTLE_MS", "2500"))   # v78: cap on extra wait once real speech is already confirmed
CALIB_MS       = int(os.environ.get("JINGLU_CALIB_MS", "600"))    # v80: calib
# window ms, website-configurable (callCalibMs), clamped 400-1000ms below
IST           = timezone(timedelta(hours=5, minutes=30))

# v5: 8 kHz telephony audio makes Sarvam's "unknown" auto-detect flip languages
# every single turn - one real call went te-IN -> bn-IN -> ta-IN -> gu-IN -> bn-IN
# and Jinglu answered a Coimbatore hotel owner in Bengali. So: detect ONCE, lock,
# and pass the locked code to STT from then on (an explicit hint also lifts
# accuracy a lot). Anything outside our sales geography is never a real answer.
ALLOWED_LANGS = {"ta-IN", "en-IN", "te-IN", "kn-IN", "ml-IN", "hi-IN", "mr-IN", "bn-IN", "gu-IN", "pa-IN", "or-IN"}
# v25: a real call locked hi-IN off the transcript 'saamne mein bas rahaa' -
# Sarvam's own text was gibberish and its detector guessed Hindi FROM that
# gibberish. The whole call then ran in Hindi at a Tamil speaker, who asked
# for Tamil three times and was ignored because the lock was already set.
# The detector is a guess on 8 kHz audio; it may only pick languages this
# business sells into. Hindi stays ALLOWED - if he SAYS hindi we speak it -
# but it can never be INFERRED. Unsure -> our home market, not a guess.
DETECT_TRUST = {"ta-IN", "en-IN"}   # v52: te/kn/ml removed - see below
TAMIL_LO, TAMIL_HI = 0x0B80, 0x0BFF     # v54: Tamil script block
TAMIL_MIN = int(os.environ.get("JINGLU_TAMIL_MIN", "4"))
EN_MARKERS = set(("yes yeah yep no nope hello hi hey ok okay sure proceed can "
                  "could you your please thank thanks tell me what which how "
                  "when where why speak english talk say said good morning "
                  "evening sir madam am is are do does going want need company "
                  "from about").split())


def tamil_chars(s):
    return sum(1 for ch in s if TAMIL_LO <= ord(ch) <= TAMIL_HI)


def en_markers(s):
    return len({w for w in re.findall(r"[A-Za-z]+", s.lower()) if w in EN_MARKERS})

ROMAN_TAMIL = ("anna", "anne", "aama", "aamaa", "illa", "enna", "sari",
               "seri", "vanakkam", "thambi", "ayya", "sollunga", "venum",
               "irukku", "evlo", "epdi", "eppadi", "nalla", "romba",
               "konjam", "thaan", "dhaan", "puriyala", "theriyala")
def roman_tamil(text):
    # v61: "Yes. Yes Anna. Hello." hit en_markers>=2 ("yes","hello") and locked
    # English off a caller who was saying Tamil words in Latin letters in the
    # same breath. Two English filler words is not English if a Tamil word
    # rode along with them.
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    return any(w in words for w in ROMAN_TAMIL)
# v52: 78 real calls showed EVERY te/kn/ml auto-detection was a Tamil
# caller rendered in the wrong script on 8 kHz audio, never a real
# Malayalam/Telugu/Kannada speaker. Trusting them locked 2 calls into a
# language the caller does not speak, and TTS only has ta/en voices
# anyway (see GOODBYE). A genuine speaker of those languages still gets
# them by NAMING the language - lang_from_words/ALLOWED_LANGS is
# untouched. Replaying the 78 calls, only those 2 change; the other 76
# lock exactly as before.
DEFAULT_LANG  = os.environ.get("JINGLU_DEFAULT_LANG", "ta-IN")
MAX_LLM_FAILS = int(os.environ.get("JINGLU_MAX_LLM_FAILS", "2"))
# v6: 160 -> 110. The prompt already caps replies at two sentences; the extra
# headroom only ever bought us a longer TTS render.
# v20: 110 tokens of Tamil renders to ~10 SECONDS of speech, and the caller
# sits through every one of them - measured 11-12s between turns while
# time-to-first-word was only 1.7s. On a phone call one short sentence per
# turn is the whole budget.
MAX_TOKENS      = int(os.environ.get("JINGLU_MAX_TOKENS", "90"))  # v41: 70 still truncating
# v26: a TOKEN cap does not bound how long a reply takes to SAY. Thanglish
# mixes English words - cheap in tokens, long in characters. 38 tokens still
# produced a 90-character reply = 7.3 SECONDS of the customer waiting.
# Seconds are what he feels, so cap what maps to seconds: characters.
SPEAK_MAX = int(os.environ.get("JINGLU_SPEAK_MAX", "58"))
# v17: the closing verdict is ~20 JSON fields - 110 tokens cut it off at
# '"in_target_area' and _json_slice then died on the missing brace, so nothing
# was ever saved. The verdict is one call per lead; give it room.
VERDICT_TOKENS = int(os.environ.get("JINGLU_VERDICT_TOKENS", "400"))
# v18: "substring not found" was not truncation - the model replied in prose
# and never emitted a brace. Asking inside the conversation was too soft.
VERDICT_SYS = ("Reply with ONE JSON object and NOTHING else. No greeting, no "
               "explanation, no markdown fence, no text before or after. Your "
               "reply must start with { and end with }.")
# v11: measured, not guessed. STT came in at 0.24s, so the wait is LLM+TTS, and
# TTS scales with how much text we hand it. Ship the first clause the moment we
# have one - the rest renders while they are already hearing it.
MIN_FIRST_CHARS = int(os.environ.get("JINGLU_MIN_FIRST_CHARS", "18"))
FIRST_MAX       = int(os.environ.get("JINGLU_FIRST_MAX", "28"))
# v12: name -> unix ts to stop skipping it. Filled from the provider's own
# "try again in ..." so a parked provider un-parks itself, no restart needed.
_LLM_COOLDOWN = {}

UA = "Mozilla/5.0 (X11; Linux x86_64) Jinglu/1.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jinglu")

# Opening lines are fixed, so we render them ONCE at boot and replay the bytes.
# This is what takes "answer -> first word" from ~7 s down to ~0.2 s.
#
# v7: the opening is now Thanglish, not English. Every lead is a Tamil Nadu
# hotel owner who enquired with us; greeting him in English and only switching
# after he answers wastes the first turn and sounds like a call centre. So the
# language is decided BEFORE the call - JINGLU_DEFAULT_LANG, ta-IN - and his
# first reply can still flip it to English (or te/kn/ml/hi), once, then it locks.
#
# v9: the greeting ASKS which language instead of guessing. Sarvam's detector on
# 8 kHz audio locked te-IN off a single "த்" - one syllable decided a whole call.
# Asking costs nothing (it is pre-rendered, same clip) and the answer is a fact
# instead of a guess. Their reply is matched on keywords FIRST, detector second.
#
# v10: greet in ENGLISH. It is the neutral opener for a cold call anywhere in
# South India - Tamil at a Bangalore or Kochi lead would be worse than English
# at a Coimbatore one. Ask, then run the whole call in whatever they answer.
OPENING_LANG = "en-IN"
# v13: open-ended - "what is your preferred language" instead of offering two.
# A caller from Kochi or Hyderabad should be able to just say Malayalam or Telugu
# without us having pre-decided the menu for him.
LANG_Q = os.environ.get("JINGLU_LANG_Q", "Which language sir?")
LANG_REPEAT = "Which language would you prefer?"
LANG_NO_SUPPORT = ("Sorry sir, we do not have support for that language. "
                   "Can we continue in English?")
TURN2_FALLBACK = {                                     # v33: spoken only if the
    "en-IN": "You enquired with us sir - what were you looking for?",
    "ta-IN": "நீங்க எங்ககிட்ட enquiry பண்ணிருந்தீங்க sir - எதுக்காக பாத்திருந்தீங்க?",
}                                                       # LLM Turn 2 call fails
# v18: was 2 (=3 asks). A real call burned 55 seconds - 08:45:18 to 08:46:13 -
# just re-asking, because Sarvam kept hearing Gujarati. One retry, then take
# the default and get on with the sale.
LANG_TRIES = int(os.environ.get("JINGLU_LANG_TRIES", "2"))  # v44: 1 was locking wrong language too fast

# v21: humans leave ~200ms between turns. We had 1.6s, split four ways with no
# single fat piece to cut. But people do not sit silent while they think - they
# say "mm", "sure", "right" and THEN answer. That backchannel is why a human
# conversation feels instant even though the real answer takes just as long.
# So: the moment STT returns, play a short filler while the LLM is still
# thinking. Pre-rendered, so it costs nothing and starts in milliseconds.
# Rotated and never repeated back-to-back, or it turns into a verbal tic.
THINKING = {
    "en-IN": ["Sure sir,", "Okay,", "Right,", "Got it,"],
    "ta-IN": ["சரி சார்,", "ம்ம்,", "ஓகே,", "புரியுது,"],
    "te-IN": ["సరే సార్,", "అలాగే,"],
    "kn-IN": ["ಸರಿ ಸರ್,", "ಆಯ್ತು,"],
    "ml-IN": ["ശരി സാർ,", "ഓകെ,"],
    "hi-IN": ["जी सर,", "ठीक है,"],
}
THINK_PCM = {}
# v22: default OFF. It fills the wait, but it also ADDS ~0.6s of talking to
# every turn, and on a real call that read as more lag, not less. Set
# JINGLU_BACKCHANNEL=1 to try it again once the reply length is under control.
BACKCHANNEL = os.environ.get("JINGLU_BACKCHANNEL", "1") == "1"  # v39/A1: on by default
OPENINGS = {
    # v32: matches the v3 call script verbatim - "Never good-morning or
    # good-evening." No time-of-day variant anymore; all three slots
    # speak the same line so the rest of the slot-picking code below
    # does not need to change.
    "morning":   "Hello sir, Microgenn - Tamil or English?",
    "afternoon": "Hello sir, Microgenn - Tamil or English?",
    "evening":   "Hello sir, Microgenn - Tamil or English?",
}

# What they might say when asked. Checked before the detector, because "English"
# spoken in a Tamil accent is exactly what the detector gets wrong.
LANG_WORDS = [
    # v17: a real caller said "Tamil" and Sarvam transcribed it in DEVANAGARI as
    # "तमिल", detected hi-IN, and locked the whole call to Hindi. The keyword list
    # only knew "tamil"/"தமிழ்", so the wrong guess won. Whatever script the STT
    # decides to write in, the WORD is still the answer - so match all of them.
    ("en-IN", ("english", "englsh", "inglish", "ingles", "ஆங்கில", "இங்கிலீ",
               "अंग्रेज", "इंग्लि", "आंग्ल", "ఇంగ్లీ", "ಇಂಗ್ಲಿ", "ഇംഗ്ലീ")),
    ("ta-IN", ("tamil", "thamil", "tamizh", "thamizh", "தமிழ்", "தமில",
               "तमिल", "तमिळ", "तामिल", "తమిళ", "ತಮಿಳ", "തമിഴ")),
    ("te-IN", ("telugu", "తెలుగు", "தெலுங்", "तेलुगु", "तेलगु")),
    ("kn-IN", ("kannada", "ಕನ್ನಡ", "கன்னட", "कन्नड")),
    ("ml-IN", ("malayalam", "മലയാളം", "மலையாள", "मलयालम")),
    ("hi-IN", ("hindi", "हिंदी", "हिन्दी", "இந்தி")),
    ("bn-IN", ("bengali", "bangla", "வங்காள")),
    ("mr-IN", ("marathi", "மராத்தி")),
    ("gu-IN", ("gujarati", "குஜராத்தி")),
    ("pa-IN", ("punjabi", "பஞ்சாபி")),
    ("ur-IN", ("urdu", "உருது")),
    ("or-IN", ("odia", "oriya")),
    ("as-IN", ("assamese",)),
    ("fr-FR", ("french",)),
    ("es-ES", ("spanish",)),
    ("ar-AE", ("arabic",)),
]


# The three Tamil L-sounds are one sound through a phone codec and Sarvam
# swaps them freely - a caller's Tamil came back with the wrong one. Fold
# them together so the consonant skeleton survives a bad transcript.
_FOLD = {"\u0bb4": "\u0bb2", "\u0bb3": "\u0bb2",
         "\u0933": "\u0932", "\u0934": "\u0932"}


def _squash(t):
    """v25: strip every vowel sign before matching. Sarvam wrote a spoken
    "Tamil" with the wrong matra on a real call, an exact match missed it,
    and he stayed stuck in Hindi for the rest of the call."""
    out = []
    for c in unicodedata.normalize("NFD", (t or "").lower()):
        # NB: unicodedata.combining() is ZERO for Indic matras - they carry
        # a combining class of 0 despite being marks. Test the CATEGORY.
        if unicodedata.category(c) in ("Mn", "Mc"):
            continue
        out.append(_FOLD.get(c, c))
    return "".join(out)


# v38: keyword cross-check for "requirement" -- mirrors LANG_WORDS/lang_from_words.
REQUIREMENT_WORDS = [
    ("hotel software", ("hotel software", "hotel management", "hotel app",
                         "property management", "pms", "hms")),
    ("website", ("website", "webiste", "web site", "webpage", "web page",
                 "web development")),
    ("mobile app", ("mobile app", "mobile application", "android app",
                     "ios app", "app develop")),
    ("ERP", ("erp", "enterprise resource")),
    ("CRM", ("crm", "customer relationship")),
    ("custom software", ("custom software", "custom app", "custom development",
                          "software development")),
]

def requirement_from_words(text):
    """Customer's own words. Returns a canonical requirement string if a known
    keyword is found, else None (fall back to the LLM's free-text verdict)."""
    low = _squash(text)
    for canonical, words in REQUIREMENT_WORDS:
        for w in words:
            if _squash(w) in low:
                return canonical
    return None

# v41: the LLM was re-asking "what do you need?" after the customer had
# already answered and moved on to city - it was tracking flow state in its
# own head, in-context, and losing it. Track the two key slots ourselves
# and hand the model a deterministic status line every turn instead of
# trusting it to remember.
NEED_WORDS = {
    "hotel":   ["hotel", "ஹோட்டல்", "room", "ரூம்"],
    "website": ["website", "வெப்சைட்", "site"],
    "app":     ["app", "ஆப்", "mobile", "மொபைல்"],
    "erp":     ["erp"],
    "crm":     ["crm"],
    "software":["software", "சாப்ட்வேர்", "சாஃப்ட்வேர்"],
}
# v62: only 15 TN cities were ever matchable - a caller from ANY other town
# (even other well-known TN towns) got city=None or a wrong difflib fuzzy match
# onto one of these 15, since the fuzzy fallback below searches this same list.
# Broadened to the state's major district headquarters/towns - still TN-only
# (this business calls TN leads), never removes an existing entry.
TN_CITIES = [
    "coimbatore", "chennai", "madurai", "salem", "trichy", "tiruchirapalli",
    "erode", "tirupur", "tiruppur", "vellore", "karur", "namakkal", "dindigul",
    "thanjavur", "kovai", "cbe", "tirunelveli", "nellai", "thoothukudi",
    "tuticorin", "nagercoil", "kanyakumari", "cuddalore", "kanchipuram",
    "kancheepuram", "kumbakonam", "sivakasi", "rajapalayam", "pollachi",
    "hosur", "nagapattinam", "karaikudi", "neyveli", "krishnagiri",
    "villupuram", "viluppuram", "virudhunagar", "ambur", "pudukkottai",
    "ooty", "udhagamandalam", "theni", "tiruvannamalai", "tindivanam",
    "gudiyatham", "ariyalur", "perambalur", "sankarankovil", "aruppukkottai",
    "paramakudi", "arakkonam", "vaniyambadi", "chengalpattu", "chengalpet",
    "ranipet", "tambaram", "avadi", "attur", "gobichettipalayam", "palani",
    "mettur", "pattukkottai", "rajapalaiyam", "sivaganga", "thiruvarur",
    "tiruvarur", "valparai", "udumalpet", "udumalaipettai",
]
def extract_need(text):
    low = text.lower()
    for need, words in NEED_WORDS.items():
        for w in words:
            if w in low:
                return need
    return None
def extract_city(text):
    low = text.lower()
    for city in TN_CITIES:
        if city in low:
            return city
    import difflib
    words = [w.strip(".,?!") for w in text.split() if len(w.strip(".,?!")) >= 4]
    for w in words:
        match = difflib.get_close_matches(w.lower(), TN_CITIES, n=1, cutoff=0.55)
        if match:
            return match[0]
    return None

# v73: Sarvam writes a sub-second answer in whatever Indic script it guesses.
# One caller who said "Tamil" came back as Gujarati on try 1 and Bengali on
# try 2, so lang_from_words() saw no language name and we defaulted. The Indic
# Unicode blocks share one layout, so the offset inside the block gives the
# phoneme regardless of which script was picked -- transliterate to Latin and
# look for the language name there. Checked against 104 logged answers:
# 2 new locks, 14 agreements, 0 conflicts with what the code already decided.
_IND_C = {0x05:"a",0x06:"aa",0x07:"i",0x08:"ii",0x09:"u",0x0A:"uu",
          0x0F:"e",0x10:"ai",0x13:"o",0x14:"au",
          0x15:"ka",0x16:"kha",0x17:"ga",0x18:"gha",0x19:"nga",
          0x1A:"cha",0x1B:"chha",0x1C:"ja",0x1D:"jha",0x1E:"nya",
          0x1F:"ta",0x20:"tha",0x21:"da",0x22:"dha",0x23:"na",
          0x24:"ta",0x25:"tha",0x26:"da",0x27:"dha",0x28:"na",
          0x2A:"pa",0x2B:"pha",0x2C:"ba",0x2D:"bha",0x2E:"ma",
          0x2F:"ya",0x30:"ra",0x31:"ra",0x32:"la",0x33:"la",0x34:"la",
          0x35:"va",0x36:"sha",0x37:"sha",0x38:"sa",0x39:"ha"}
_IND_V = {0x3E:"aa",0x3F:"i",0x40:"ii",0x41:"u",0x42:"uu",
          0x47:"e",0x48:"ai",0x4B:"o",0x4C:"au",0x4D:""}
_LAT_LANG = (("ta-IN", ("tamil","tamiz","thamil","thamiz","damil")),
             ("en-IN", ("english","engli","ingli","inglis","ingkili","ingili")),
             ("hi-IN", ("hindi","hindhi")),
             ("te-IN", ("telugu","thelugu")),
             ("kn-IN", ("kannada","kanada")),
             ("ml-IN", ("malayalam","malayala")))


def indic_to_latin(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if cp < 0x0900 or cp > 0x0D7F:
            out.append(ch.lower())
            continue
        off = (cp - 0x0900) % 0x80
        if off in _IND_V:
            if out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]
            out.append(_IND_V[off])
        elif off in _IND_C:
            out.append(_IND_C[off])
    s, res = "".join(out), []
    for c in s:
        if res and res[-1] == c and c in "aeiou":
            continue
        res.append(c)
    return "".join(res)


def lang_from_latin(lat):
    for code, pats in _LAT_LANG:
        for p in pats:
            if p in lat:
                return code
    return None


def lang_from_words(text):
    """Their answer to LANG_Q. Returns a code - which may be one we do NOT
    support - or None if they said something that isn't a language at all.

    v53: a confused sentence that names TWO languages ("I speak English,
    Tamil-la thaan pesittu irundhen") used to return the FIRST match found -
    on a real call that falsely triggered a mid-call switch to English off
    a sentence that was not a language request at all, and the LLM lost its
    place in the flow trying to process the switch. If more than one
    language is named, treat it as ambiguous confusion, not a request."""
    low = _squash(text)
    hits = set()
    for code, words in LANG_WORDS:
        for w in words:
            if _squash(w) in low:
                hits.add(code)
                break
    if len(hits) == 1:
        return hits.pop()
    return None


# v10: goodbye follows whatever they chose. Signing off in English after they
# asked for Tamil is exactly the inconsistency we just spent a day removing.
GOODBYE = {
    "en-IN": "Thank you for your time sir. Have a good day.",
    "ta-IN": "நேரம் ஒதுக்கினதுக்கு ரொம்ப thanks சார். நல்ல நாளா இருக்கட்டும்.",
    "te-IN": "మీ సమయానికి ధన్యవాదాలు సార్. మంచి రోజు గడపండి.",
    "kn-IN": "ನಿಮ್ಮ ಸಮಯಕ್ಕೆ ಧನ್ಯವಾದಗಳು ಸರ್. ಒಳ್ಳೆಯ ದಿನ ಕಳೆಯಿರಿ.",
    "ml-IN": "നിങ്ങളുടെ സമയത്തിന് നന്ദി സാർ. നല്ല ദിവസം ആശംസിക്കുന്നു.",
    "hi-IN": "आपके समय के लिए धन्यवाद सर। आपका दिन शुभ हो।",
    "mr-IN": "तुमच्या वेळेबद्दल धन्यवाद सर. तुमचा दिवस चांगला जावो.",
    "bn-IN": "আপনার সময়ের জন্য ধন্যবাদ স্যার। আপনার দিন শুভ হোক।",
    "gu-IN": "તમારા સમય માટે આભાર સર. તમારો દિવસ સારો રહે.",
    "pa-IN": "ਤੁਹਾਡੇ ਸਮੇਂ ਲਈ ਧੰਨਵਾਦ ਸਰ। ਤੁਹਾਡਾ ਦਿਨ ਚੰਗਾ ਲੰਘੇ।",
    "or-IN": "ଆପଣଙ୍କ ସମୟ ପାଇଁ ଧନ୍ୟବାଦ ସାର୍। ଆପଣଙ୍କର ଦିନ ଶୁଭ ହେଉ।",
}
OPENING_PCM = {}
GOODBYE_PCM = {}
GOODBYE_PCM_MAM = {}  # v86: gendered goodbye variant, see prerender()
FIXED_PCM = {}          # v13: LANG_REPEAT / LANG_NO_SUPPORT, rendered once at boot
SCRIPT_GREETING_PCM = {}  # v36: per-CallScript greeting, keyed by scriptId


def slot_now():
    h = datetime.now(IST).hour
    return "morning" if h < 12 else ("afternoon" if h < 17 else "evening")


# ------------------------------------------------------------------ helpers
def http_json(url, payload, headers, timeout=20):
    headers = dict(headers)
    headers.setdefault("User-Agent", UA)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %s %s :: %s" % (
            e.code, url.rsplit("/", 1)[-1], e.read()[:300].decode(errors="ignore"))) from None
    return json.loads(body) if body else None


def rpc(fn, payload=None, timeout=15):
    return http_json(f"{SUPA}/rest/v1/rpc/{fn}", payload or {},
                     {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}",
                      "Content-Type": "application/json"}, timeout)


def pitch_gender(pcm):
    """v35: rough voice-pitch classifier from raw PCM (16-bit mono).
    Autocorrelation-based F0 estimate. Male speech ~85-180Hz, female ~165-255Hz.
    This is a HEURISTIC on noisy 8kHz telephony audio -- treat as best-effort,
    not a guarantee. Returns "sir", "mam", or None (inconclusive)."""
    try:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        if len(arr) < RATE // 20:   # need at least 50ms of audio
            return None
        arr = arr - arr.mean()
        if np.abs(arr).max() < 200:  # too quiet, likely silence/noise
            return None
        corr = np.correlate(arr, arr, mode="full")[len(arr) - 1:]
        min_lag = int(RATE / 260)   # ~260Hz upper bound
        max_lag = int(RATE / 80)    # ~80Hz lower bound
        if max_lag >= len(corr):
            return None
        seg = corr[min_lag:max_lag]
        if len(seg) == 0 or seg.max() <= 0:
            return None
        peak_lag = min_lag + int(np.argmax(seg))
        f0 = RATE / peak_lag
        if 80 <= f0 < 165:
            return "sir"
        if 165 <= f0 <= 255:
            return "mam"
        return None
    except Exception:
        return None


def rms(pcm):
    n = len(pcm) // 2
    if not n:
        return 0
    total = 0
    for s in struct.unpack("<%dh" % n, pcm[:n * 2]):
        total += s * s
    return int((total / n) ** 0.5)


def pcm_to_wav(pcm, rate=RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm(b):
    with wave.open(io.BytesIO(b), "rb") as w:
        if w.getframerate() != RATE:
            log.warning("TTS returned %d Hz, expected %d", w.getframerate(), RATE)
        return w.readframes(w.getnframes())


def sentences(text):
    """Split a reply so the first words reach the caller while the rest renders."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    out = []
    for p in parts:                       # keep chunks meaty enough to sound natural
        if out and (len(out) > 1 or len(out[-1]) >= 20) and len(out[-1]) < 40:
            out[-1] += " " + p
        else:
            out.append(p)
    return out or [text]


# ------------------------------------------------------------------ Sarvam
def sarvam_stt(pcm, lang=None):
    """lang=None -> let Sarvam guess (first turn only). Otherwise pass the locked
    code: it stops the per-turn language flip-flop AND improves the transcript."""
    wav = pcm_to_wav(pcm)
    b = "----jinglu" + uuid.uuid4().hex
    parts = []

    def field(k, v):
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())

    field("model", STT_MODEL)
    field("language_code", lang or "unknown")
    field("mode", "transcribe")
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())

    req = urllib.request.Request(
        "https://api.sarvam.ai/speech-to-text", data=b"".join(parts),
        headers={"api-subscription-key": SARVAM_KEY, "User-Agent": UA,
                 "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("STT HTTP %s :: %s" % (e.code, e.read()[:200].decode(errors="ignore"))) from None
    return (d.get("transcript") or "").strip(), d.get("language_code") or "en-IN"


def sarvam_tts(text, lang="en-IN"):
    # v62: Sarvam TTS rejects "or-IN" (Validation Error 400) -- it wants "od-IN" for Odia.
    # The rest of the codebase (ALLOWED_LANGS, lang detection, GOODBYE keys) standardises on
    # "or-IN", so translate only at the Sarvam TTS boundary rather than renaming everywhere.
    if lang == "or-IN":
        lang = "od-IN"
    d = http_json("https://api.sarvam.ai/text-to-speech",
                  {"text": text[:2400], "target_language_code": lang,
                   "speaker": TTS_SPEAKER, "model": TTS_MODEL,
                   "speech_sample_rate": RATE, "pace": 1.05},
                  {"api-subscription-key": SARVAM_KEY,
                   "Content-Type": "application/json"}, timeout=25)
    return wav_to_pcm(base64.b64decode(d["audios"][0]))


SCRIPT_RANGE = {
    "ta-IN": (0x0B80, 0x0BFF), "hi-IN": (0x0900, 0x097F),
    "mr-IN": (0x0900, 0x097F), "bn-IN": (0x0980, 0x09FF),
    "pa-IN": (0x0A00, 0x0A7F), "gu-IN": (0x0A80, 0x0AFF),
    "or-IN": (0x0B00, 0x0B7F), "te-IN": (0x0C00, 0x0C7F),
    "kn-IN": (0x0C80, 0x0CFF), "ml-IN": (0x0D00, 0x0D7F),
}

# v62: SCRIPT_RANGE tells us what script a reply IS in. This dict names the
# script so we can tell the model, up front, what script to WRITE in -- see
# the _lock() note below on why that's the only place this can be fixed.
SCRIPT_NAME = {
    "ta-IN": "Tamil", "hi-IN": "Devanagari (Hindi)", "mr-IN": "Devanagari (Marathi)",
    "bn-IN": "Bengali", "pa-IN": "Gurmukhi (Punjabi)", "gu-IN": "Gujarati",
    "or-IN": "Odia", "te-IN": "Telugu", "kn-IN": "Kannada", "ml-IN": "Malayalam",
}


FILLER = ("புரியல", "புரியலை", "என்", "ஹலோ", "hello", "sorry",
          "என்ன சொன்னீங்க", "மீண்டும்", "puri", "puriya", "puriyala",
          "puriyalai", "purila", "i don't understand", "i dont understand")


def is_filler(text):
    """v60: a short reply that is only confusion or noise, never an answer.

    v51 already flagged these for the model, but extract_need/extract_city ran
    BEFORE that check and could bank the confusion word itself as the answer.
    On a real call "புரியல" (I do not understand) came back from STT as
    "Puriya", went into the city slot, and the bot spent four turns and forty
    five seconds asking "are you from Puri?". A non-answer must fill nothing."""
    low = (text or "").strip().lower()
    return len(low) <= 20 and any(k in low for k in FILLER)


def script_ok(text, locked):
    """v58: does this transcript look like the language we locked?

    On 8 kHz audio Sarvam sometimes returns a turn written in a completely
    different script. A real call locked to ta-IN came back in Devanagari and,
    because the model mirrors whatever it sees in the transcript, the rest of
    the call ran in Hindi at a Tamil speaker. Latin-only text passes in every
    call - English words are normal in both - so this only catches a wholesale
    swap into another Indic script."""
    want = SCRIPT_RANGE.get(locked)
    if want is None:
        return True
    ok = other = 0
    for ch in text:
        c = ord(ch)
        if want[0] <= c <= want[1]:
            ok += 1
        elif 0x0900 <= c <= 0x0DFF:
            other += 1
    if ok == 0 and other == 0:
        return True
    return ok >= other


def tts_lang(text, locked):
    """v56: the model breaks the language lock in about one reply in eight - it
    writes Tamil script while the call is locked to en-IN (52 of 182 English
    replies in the log), or romanised Tamil while locked to ta-IN. Handing Tamil
    script to an English voice is exactly what the caller hears as garbled,
    unintelligible speech. Speak the script that is actually in front of us; the
    lock still drives the prompt, the flow and the sign-off. Latin-only text is
    left on the locked voice, so this can only ever fix, never change, a reply
    that already obeyed the rule."""
    for ch in text:
        if 0x0B80 <= ord(ch) <= 0x0BFF:
            return "ta-IN"
    return locked


def groq_chat(messages, timeout=20, max_tokens=None):
    """v55: this is the VERDICT call. It was still hard-wired to Groq while the
    conversation path had long since moved to llm_providers(). When the Groq key
    stopped working, every verdict failed - 28 of 77 logged calls ended
    qualified=None, so the lead was never scored and no summary was written.
    Walk the same provider chain as the conversation, primary first."""
    last = "no provider"
    for name, url, key, model in llm_providers():
        payload = {"model": model, "messages": messages,
                   "temperature": 0.4, "max_tokens": max_tokens or MAX_TOKENS}
        if model.startswith("qwen/"):
            payload["reasoning_effort"] = "none"   # v31: qwen3 thinking mode off
        try:
            d = http_json(url, payload,
                          {"Authorization": "Bearer " + key,
                           "Content-Type": "application/json"}, timeout)
            return d["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last = "%s :: %s" % (name, str(exc)[:120])
            log.warning("verdict LLM %s failed: %s", name, str(exc)[:120])
    raise RuntimeError("all LLM providers failed :: %s" % last)


def _retry_after(msg):
    """Groq says 'try again in 10m35.04s'. Believe it - no point hammering."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg or "")
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 5
    return 300.0


def llm_providers():
    """v13: Groq -> Gemini -> OpenAI by default. All three speak the OpenAI
    chat-completions wire format, so one code path serves all of them; only
    url/key/model differ. A provider with no key configured is simply skipped.
    v38: JINGLU_LLM_PROVIDER (from the CRM "Primary AI provider" setting) moves
    that provider to the front of the chain - the other two remain as fallback,
    order unchanged among themselves."""
    chain = [
        ("groq", "https://api.groq.com/openai/v1/chat/completions",
         GROQ_KEY, LLM_MODEL),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
         os.environ.get("GEMINI_KEY", ""), os.environ.get("JINGLU_LLM_GEMINI", "gemini-2.0-flash")),
        ("openai", "https://api.openai.com/v1/chat/completions",
         os.environ.get("OPENAI_KEY", ""), os.environ.get("JINGLU_LLM_OPENAI", "gpt-4o-mini")),
    ]
    primary = os.environ.get("JINGLU_LLM_PROVIDER", "groq").strip().lower()
    chain.sort(key=lambda p: p[0] != primary)   # primary first, stable order otherwise
    out = []
    for name, url, key, model in chain:
        if key:
            out.append((name, url, key, model))
    return out


def groq_stream(messages, on_sentence, timeout=25):
    """v12: try each provider in turn. A 429 parks that one for exactly as long
    as it asked for, then it comes back on its own - no manual switching.

    Once a provider has already spoken a sentence we can NOT fail over: the
    caller would hear the reply start twice. At that point we surface the error."""
    chain = llm_providers()
    if not chain:
        raise RuntimeError("no LLM key configured (GROQ_KEY / OPENAI_KEY / GEMINI_KEY)")
    spoke = [False]

    def once(part):
        spoke[0] = True
        on_sentence(part)

    last = None
    for name, url, key, model in chain:
        cool = _LLM_COOLDOWN.get(name, 0)
        if cool > time.time():
            continue
        try:
            return _stream_one(url, key, model, messages, once, timeout, name)
        except Exception as e:
            last = e
            msg = str(e)
            if spoke[0]:                      # already talking - cannot restart
                raise
            if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
                wait = _retry_after(msg)
                _LLM_COOLDOWN[name] = time.time() + wait
                log.warning("LLM %s rate-limited, parked %.0fs -> falling back", name, wait)
            else:
                log.warning("LLM %s failed (%s) -> falling back", name, msg[:120])
    raise RuntimeError("all LLM providers failed :: %s" % last)


def _stream_one(url, key, model, messages, on_sentence, timeout, name):
    """One provider. Each finished clause goes to TTS the moment it lands, so the
    caller hears clause 1 while the model is still writing clause 2.

    The closing verdict is JSON and must never be spoken, so if the reply opens
    with '{' we collect silently and speak nothing. Returns the full text."""
    payload = {"model": model, "messages": messages, "temperature": 0.4,
               "max_tokens": MAX_TOKENS, "stream": True}
    if name == "groq" and model.startswith("qwen/"):
        payload["reasoning_effort"] = "none"  # v31: qwen3 thinking mode off - <think> tags were reaching TTS
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", "User-Agent": UA})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %s %s :: %s" % (
            e.code, name, e.read()[:300].decode(errors="ignore"))) from None

    # v17: `spoken` is how many chars of the reply have already gone to TTS.
    # Everything from the first '{' onward is the closing verdict and must never
    # be voiced - a real caller heard "location null, in target area null" read
    # out because the old check only looked at startswith("{").
    full, spoken, first = "", 0, True

    def speakable():
        cut = full.find("{")
        return full if cut == -1 else full[:cut]

    with r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                tok = json.loads(chunk)["choices"][0]["delta"].get("content") or ""
            except (ValueError, KeyError, IndexError):
                continue
            if not tok:
                continue
            full += tok
            buf = speakable()[spoken:]
            if not buf.strip():
                continue
            # the FIRST chunk breaks at any clause boundary, not just a full stop:
            # it is the only chunk the caller waits on.
            if first:
                # v22: search for the break AT OR AFTER MIN_FIRST_CHARS, not
                # from position 0. Searching from 0 found the FIRST comma and
                # m.end() stayed pinned there, so a reply opening "சார்," shipped
                # a 4-char chunk - measured (5ch)(4ch)(5ch)(5ch) on a real call.
                # The caller heard "சார்," then half a second of nothing then the
                # rest. That stutter is what "too much lag" actually was.
                m = re.compile(r"[.!?,;:](\s|$)").search(buf, MIN_FIRST_CHARS)
                if not m and len(buf) >= FIRST_MAX:
                    m = re.compile(r"\s").search(buf, FIRST_MAX)
                ok = bool(m)
            else:
                m = re.search(r"[.!?](\s|$)", buf)
                ok = m and len(buf.strip()) >= 20
            if ok:
                on_sentence(buf[:m.end()].strip())
                spoken += m.end()
                first = False

    tail = speakable()[spoken:].strip()
    if tail:
        on_sentence(tail)
    if name != "groq":
        log.info("LLM served by %s", name)
    return full.strip()


# ------------------------------------------------------------------ AudioSocket
class AudioSocket:
    TERMINATE, UUID, DTMF, AUDIO, ERROR = 0x00, 0x01, 0x03, 0x10, 0xff

    def __init__(self, conn):
        self.c = conn
        self.c.settimeout(60)
        self.closed = False
        self.uuid = None
        self.speaking = False
        self.live_level = 0
        self._preroll = []   # v85: rolling pre-barge audio, filled while speaking
        self.q = queue.Queue(maxsize=3000)
        self.out_q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._writer, daemon=True).start()

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.c.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _reader(self):
        try:
            while True:
                head = self._recv_exact(3)
                if not head:
                    break
                t, ln = head[0], struct.unpack(">H", head[1:3])[0]
                payload = self._recv_exact(ln) if ln else b""
                if payload is None or t == self.TERMINATE:
                    break
                if t == self.UUID and self.uuid is None:
                    self.uuid = payload
                    continue
                if t == self.AUDIO and self.speaking:
                    self.live_level = rms(payload)
                    # v85: keep a short rolling pre-roll of audio heard while we were
                    # still talking -- a barge-in only confirms after several loud
                    # frames in a row, so the caller's first words were always being
                    # thrown away here. Handed back to listen() on a confirmed barge-in.
                    self._preroll.append(payload)
                    if len(self._preroll) > PREROLL_FRAMES:
                        del self._preroll[0]
                    continue
                try:
                    self.q.put_nowait((t, payload))
                except queue.Full:
                    try:
                        self.q.get_nowait()
                        self.q.put_nowait((t, payload))
                    except (queue.Empty, queue.Full):
                        pass
        finally:
            self.closed = True
            try:
                self.q.put_nowait((None, None))
            except queue.Full:
                pass

    def _writer(self):
        """A frame every 20 ms, always. Silence counts as activity - dead air
        for 2 s makes app_audiosocket.c hang the call up."""
        silence = b"\x00" * FRAME_BYTES
        hdr = struct.pack(">BH", self.AUDIO, FRAME_BYTES)
        nxt = time.time()
        while not self.closed:
            try:
                frame = self.out_q.get_nowait()
            except queue.Empty:
                frame = silence
            try:
                self.c.sendall(hdr + frame)
            except OSError:
                self.closed = True
                return
            nxt += 0.02
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()

    def read(self, timeout=30):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def enqueue(self, pcm):
        for i in range(0, len(pcm), FRAME_BYTES):
            f = pcm[i:i + FRAME_BYTES]
            if len(f) < FRAME_BYTES:
                f += b"\x00" * (FRAME_BYTES - len(f))
            self.out_q.put(f)

    def flush_out(self):
        while True:
            try:
                self.out_q.get_nowait()
            except queue.Empty:
                return

    def wait_drain(self, stop_check=None):
        while not self.closed:
            if self.out_q.empty():
                return True
            if stop_check and stop_check():
                self.flush_out()
                return False
            time.sleep(0.02)
        return False

    def take_preroll(self):
        """v85: hand back (and clear) audio captured while we were still
        speaking, so a confirmed barge-in can hand the caller's own
        first words to listen() instead of losing them."""
        frames = self._preroll
        self._preroll = []
        return frames

    def hangup(self):
        try:
            self.c.sendall(struct.pack(">BH", self.TERMINATE, 0))
        except OSError:
            pass
        self.closed = True


# ------------------------------------------------------------------ the call
class Call:
    def __init__(self, sock, lead_id, prompt):
        _refresh_settings_from_db()  # v69: hot-apply website settings for this call
        self.a = sock
        self.lead_id = lead_id
        self.started = time.time()
        self.lang = OPENING_LANG     # v10: greeting is English; their answer sets the rest
        self.lang_locked = None      # v5: None until the first transcript decides it
        self.llm_fails = 0           # v5: consecutive LLM errors
        self.t_stt = 0.0             # v8: last STT round-trip, for the TURN log
        self.noise = NOISE_INIT      # v15: this line's own ambient level, learned live
        self.done = False            # v26: verdict emitted -> hang up
        self.vlevel = 0.0            # v24: how loud this caller actually is
        self.relax  = 1.0            # v24: gate stand-down when he goes unheard
        self._think_i = -1           # v21: which filler we used last
        self.lvls = deque(maxlen=150)  # v15: last 3s of frame levels, across turns
        self.warm = False            # v15: has this line been calibrated yet
        self.gender = None                      # v35: 'sir'/'mam' guessed from voice pitch
        self.turns = 0
        self.slots = {"need": None, "city": None}       # v41: deterministic flow state
        self.asked = {"need": 0, "city": 0}              # v41: how many times we asked each
        self.once_more_used = False      # v50: "Sorry sir, once more?" allowed ONCE per call
        self.transcript = []
        self.msgs = [{"role": "system", "content": prompt}]
        # v32: per-lead script from the CRM, keyed on the lead's category.
        self.script_name = "file"
        self._script_id = None            # v36: which pre-rendered greeting to use
        self._script_greeting = None
        _cs = fetch_crm_script(lead_id)
        if _cs and _cs.get("script"):
            self.msgs = [{"role": "system", "content": _cs["script"]}]
            self.script_name = "%s (%s)" % (_cs.get("name") or "crm",
                                            _cs.get("reason") or "")
            self._script_id = _cs.get("scriptId")
            self._script_greeting = _cs.get("greeting")
            # v51: [ ] in the greeting is a placeholder for agentName -
            # look it up once and substitute before it is ever spoken.
            if self._script_greeting and self._script_id:
                try:
                    _c2 = sqlite3.connect(DB_PATH, timeout=5)
                    try:
                        _row = _c2.execute(
                            "select agentName from CallScript where id=?",
                            (self._script_id,)).fetchone()
                    finally:
                        _c2.close()
                    if _row and _row[0]:
                        self._script_greeting = self._script_greeting.replace(
                            "[ ]", _row[0])
                except Exception as e:
                    log.warning("[%s] agentName lookup failed (%s)",
                                str(lead_id)[:8], e)
        log.info("[%s] script: %s", str(lead_id)[:8], self.script_name)

    def elapsed(self):
        return time.time() - self.started

    def load_lead(self):
        try:
            d = rpc("lead_context", {"p_lead_id": self.lead_id}) or {}
            self.msgs.append({"role": "system",
                              "content": "You are calling %s (phone %s)."
                                         % (d.get("name") or "sir", d.get("phone") or "")})
            return d.get("name") or "sir"
        except Exception as e:
            log.warning("lead_context failed: %s", e)
            return "sir"

    def _barge_checker(self):
        # v15: barge-in gets the same treatment. A fixed 1200 meant a noisy
        # lobby "interrupted" Jinglu every few seconds and cut her off mid-word.
        hot = [0]

        def check():
            bar = max(BARGE_RMS, int(self.noise * NOISE_MULT * 1.6))
            if self.a.live_level > bar:
                hot[0] += 1
            else:
                hot[0] = 0
            return hot[0] >= BARGE_FRAMES
        return check


    def say_pcm(self, text, pcm):
        """Play already-rendered audio (pre-cached greeting / goodbye)."""
        self.transcript.append({"role": "jinglu", "text": text, "ms": int(self.elapsed() * 1000)})
        log.info("[%s] JINGLU: %s", self.lead_id[:8], text)
        self.a.speaking = True
        self.a.live_level = 0
        self.a.enqueue(pcm)
        ok = self.a.wait_drain(self._barge_checker())
        self.a.speaking = False
        self.a.quiet_until = time.time() + ECHO_GRACE_MS / 1000.0
        _pre = self.a.take_preroll()
        if ok:
            self.a.drain()
        else:
            # v85: they were already talking when the barge-in cut us off --
            # give listen() the words it would otherwise have missed.
            for _p in _pre:
                try:
                    self.a.q.put_nowait((self.a.AUDIO, _p))
                except queue.Full:
                    pass
        return ok

    def think(self, customer_text=None):
        """v21: say something human the instant we know what they said, so the
        LLM's thinking time lands under our own voice instead of under silence.
        v39/A1: skip when he just asked a direct question - a filler before an
        answer sounds evasive. Returns True if a filler was actually spoken,
        so the caller can tell the LLM not to open with another acknowledgment.
        v39/A1-fix: unlike say()/say_pcm()/stream_say(), this must return
        instantly (the real reply streams right behind it), so no wait_drain()
        here - but leaving speaking=True forever froze the mic and silently
        broke every turn after this one, so reset it immediately, still
        honouring the echo-fix grace window."""
        if not BACKCHANNEL:
            return False
        if customer_text:
            t = customer_text.strip()
            low = t.lower()
            if t.endswith("?") or any(k in low for k in
                    ("price", "cost", "who are you", "which company", "how much")):
                return False
        pool = THINK_PCM.get(self.lang) or THINK_PCM.get(OPENING_LANG) or []
        if not pool:
            return False
        i = (self._think_i + 1) % len(pool)      # never the same one twice
        self._think_i = i
        self.a.speaking = True
        self.a.live_level = 0
        self.a.enqueue(pool[i])
        self.a.speaking = False
        self.a.quiet_until = time.time() + ECHO_GRACE_MS / 1000.0
        return True

    def bye(self):
        """v10: sign off in the language they picked, from the pre-rendered clip."""
        # v53: bye() has four call sites and had no guard. The dead-air path
        # spoke the sign-off then broke out of the loop, and the post-loop
        # "call limit reached" branch spoke it again - real calls ended with
        # "Thank you for your time sir. Have a good day." twice, back to back.
        if getattr(self, "_said_bye", False):
            return
        self._said_bye = True
        code = self.lang if self.lang in GOODBYE_PCM else OPENING_LANG
        # v86: prefer the "madam" clip when we guessed the caller is a
        # woman and one exists for this language -- the fixed goodbye
        # clip used to say "sir" unconditionally, even on calls where
        # every LLM-generated line correctly said "madam" throughout.
        pcm = GOODBYE_PCM.get(code)
        if self.gender == "mam" and GOODBYE_PCM_MAM.get(code):
            pcm = GOODBYE_PCM_MAM[code]
        if pcm:
            self.say_pcm(GOODBYE[code], pcm)

    def say(self, text):
        """TTS sentence by sentence - the writer plays chunk 1 while chunk 2 renders."""
        if not text:
            return True
        self.transcript.append({"role": "jinglu", "text": text, "ms": int(self.elapsed() * 1000)})
        log.info("[%s] JINGLU: %s", self.lead_id[:8], text)
        barge = self._barge_checker()
        self.a.speaking = True
        self.a.live_level = 0
        t0 = time.time()
        first = True
        for part in sentences(text):
            if self.a.closed:
                break
            try:
                pcm = sarvam_tts(part, tts_lang(part, self.lang))
            except Exception as e:
                log.error("TTS failed: %s", e)
                break
            if first:
                log.info("[%s] first audio in %.2fs", self.lead_id[:8], time.time() - t0)
                first = False
            self.a.enqueue(pcm)
            if barge():                      # they cut in while we were rendering
                self.a.flush_out()
                break
        ok = self.a.wait_drain(barge)
        self.a.speaking = False
        self.a.quiet_until = time.time() + ECHO_GRACE_MS / 1000.0
        _pre = self.a.take_preroll()
        if ok:
            self.a.drain()
        else:
            log.info("[%s] barge-in", self.lead_id[:8])
            for _p in _pre:
                try:
                    self.a.q.put_nowait((self.a.AUDIO, _p))
                except queue.Full:
                    pass
        return ok

    def stream_say(self):
        """v6: one turn, LLM streamed straight into TTS. Returns the full reply
        (which may be the verdict JSON -- in that case nothing was spoken).
        v36: TTS synthesis now runs on a background worker thread, decoupled
        from LLM token streaming, so sentence N+1's TTS can start rendering
        while sentence N is still playing -- instead of blocking the LLM
        reader on each TTS network call."""
        barge = self._barge_checker()
        self.a.speaking = True
        self.a.live_level = 0
        t0 = time.time()
        st = {"first": True, "stop": False, "llm": 0.0, "tts": 0.0, "chars": 0,
              "pcm": 0, "said": 0}

        text_q = queue.Queue()
        SENTINEL = object()

        def tts_worker():
            while True:
                part = text_q.get()
                if part is SENTINEL:
                    return
                if self.a.closed or st["stop"]:
                    continue
                t_tts = time.time()
                if st["first"]:
                    st["llm"] = t_tts - t0
                    st["chars"] = len(part)
                try:
                    pcm = sarvam_tts(part, tts_lang(part, self.lang))
                except Exception as e:
                    log.error("TTS failed: %s", e)
                    st["stop"] = True
                    continue
                if st["first"]:
                    st["tts"] = time.time() - t_tts
                    llm_tts = time.time() - t0
                    log.info("[%s] TURN %.2fs = %.2f silence + %.2f stt + %.2f llm + %.2f tts (%dch)",
                            self.lead_id[:8], SILENCE_MS / 1000.0 + self.t_stt + llm_tts,
                            SILENCE_MS / 1000.0, self.t_stt, st["llm"], st["tts"], st["chars"])
                    st["first"] = False
                st["pcm"] += len(pcm)
                st["said"] += len(part)
                self.a.enqueue(pcm)
                if st["said"] >= SPEAK_MAX:                   # v26: enough for one turn
                    log.info("[%s] reply capped at %dch", self.lead_id[:8], st["said"])
                    st["stop"] = True
                if barge():                                    # they cut in mid-render
                    self.a.flush_out()
                    st["stop"] = True

        worker = threading.Thread(target=tts_worker, daemon=True)
        worker.start()

        def on_sentence(part):
            if self.a.closed or st["stop"]:
                return
            text_q.put(part)

        try:
            full = groq_stream(self.msgs, on_sentence)
        finally:
            text_q.put(SENTINEL)
            worker.join(timeout=10)
            ok = self.a.wait_drain(barge)
            self.a.speaking = False
            self.a.quiet_until = time.time() + ECHO_GRACE_MS / 1000.0
            _pre = self.a.take_preroll()
            if ok:
                self.a.drain()
            else:
                log.info("[%s] barge-in", self.lead_id[:8])
                for _p in _pre:
                    try:
                        self.a.q.put_nowait((self.a.AUDIO, _p))
                    except queue.Full:
                        pass
        # v26: the model emits the verdict JSON when IT decides the call is
        # over. That is our only end-of-call signal and we ignored it -- on a
        # real call Jinglu said "Good day!" then sat through the 20s turn
        # timeout AND the 10s dead-air timer, said goodbye a SECOND time,
        # and burned 46 seconds of billed call after the conversation ended.
        if full and "{" in full:
            self.done = True
        if full and not full.lstrip().startswith("{"):
            self.transcript.append({"role": "jinglu", "text": full,
                                     "ms": int(self.elapsed() * 1000)})
            # v20: the number nobody was measuring. TURN only ever timed the
            # FIRST word. A real call showed TURN 1.7s but 11-12s between turns,
            # because she then talked for ten more seconds. THIS is the lag.
            log.info("[%s] SPOKE %.1fs (%dch) | JINGLU: %s", self.lead_id[:8],
                    st["pcm"] / (RATE * 2.0), st["said"], full)
        return full

    def pick_language(self):
        """v13: the greeting ended with 'what is your preferred language sir?'.
        Resolve it before the sales conversation starts, so the whole call runs
        in one language. Three outcomes, all honest:
          - they name a language we speak      -> lock it
          - they name one we do NOT speak      -> say so, offer English, lock en
          - we cannot tell what they said      -> apologise, ask again
        Returns False if the call died while we were asking."""
        for attempt in range(max(1, LANG_TRIES)):
            pcm = self.listen()
            if self.a.closed:
                return False

            if self.gender is None:                          # v35: try once, first turn
                g = pitch_gender(pcm)
                if g:
                    self.gender = g
                    log.info("[%s] gender guess -> %s", self.lead_id[:8], g)
            if not pcm:                                   # silence, not confusion
                log.info("[%s] %ds silence at language question",
                         self.lead_id[:8], DEAD_AIR_MS // 1000)
                self.bye()
                return False

            t0 = time.time()
            try:
                text, det = sarvam_stt(pcm, None)         # no hint - they may be anyone
            except Exception as e:
                log.error("STT failed: %s", e)
                text, det = "", None
            self.t_stt = time.time() - t0
            said = lang_from_words(text)
            log.info("[%s] LANG try %d: %r -> said=%s det=%s",
                     self.lead_id[:8], attempt + 1, text[:40], said, det)

            if said in ALLOWED_LANGS:
                return self._lock(said, "asked", text)
            if said:                                       # a real language, just not ours
                log.info("[%s] %s requested - not supported, offering English",
                         self.lead_id[:8], said)
                self.say_fixed(LANG_NO_SUPPORT)
                return self._lock("en-IN", "unsupported:%s" % said, text)
        if not said:
            # v73: the script came back wrong -- read it phonetically instead.
            said_t = lang_from_latin(indic_to_latin(text))
            if said_t in ALLOWED_LANGS:
                return self._lock(said_t, "asked-translit", text)
            # v18: this used to demand len(text) >= 12, so a caller who simply
            # said "Hello" (det=en-IN, 5 chars) was thrown away and the call
            # defaulted to Tamil. If the detector names a language we speak,
            # that IS an answer - short or not.
            # v26: kn-IN got locked off the transcript 'houm houm' - two
            # words, said=None, detector guessed Kannada. Same bug as the
            # Hindi one, just a language that WAS on the trusted list. If he
            # did not NAME a language he has not answered; ask again. Only
            # on the last try is a guess better than nothing.
            # v40: trusting the detector only on the LAST try meant a caller
            # who clearly said "Hello" (det=en-IN) still got "sorry, did not
            # catch that" on try 1 - every real call paid that tax. Trust it
            # starting try 2 instead of only the final try; try 1 stays
            # strict so a single misheard word (the v26 "houm houm" -> kn-IN
            # case) still cannot lock the wrong language off one guess.
            # v45: a caller who says a single unambiguous English word
            # ("Yes", "Hello", "Hi", "OK") on the very FIRST try was still
            # told "sorry, did not catch that" - the v26 guard against
            # ambiguous garbled text ("houm houm") was blocking clean short
            # replies too. Trust det=en-IN on try 1 ONLY when the whole
            # utterance is nothing but one of these words - anything longer
            # or messier still waits for try 2+ like before.
            # v61: yes/yeah/yep dropped. Every logged "hello" clear-ack (7 calls) was a
            # real English speaker; the one "Yes." clear-ack (call 2881420e) locked English
            # off a woman who then never spoke again -- looks like a hang-up caused by the
            # wrong lock. Tamil callers say "yes" as a reflex backchannel while meaning to
            # answer in Tamil. Losing this only costs one extra re-ask for a real "Yes" caller.
            CLEAR_EN_ACK = {"hello", "hi", "hey", "ok", "okay", "sure"}
            clean = text.strip().strip(".,!?").lower()
            if det == "en-IN" and clean in CLEAR_EN_ACK:
                return self._lock(det, "clear-ack", text)
            # v54: a first answer WRITTEN IN TAMIL SCRIPT is direct evidence;
            # Sarvam language label is only a guess on 8 kHz audio. Refusing it
            # on attempt 0 cost whole calls: four logged calls answered in Tamil
            # script on try 1, were asked again, then locked en-IN off a garbled
            # one-word "Hello" / "Comment" - a Tamil caller ran the rest of the
            # call in English. Replaying the logged calls this fires on exactly
            # those four and corrects all four, and drops the "did not catch
            # that" repeat from 81 calls to 62.
            if tamil_chars(text) >= TAMIL_MIN:
                return self._lock("ta-IN", "ta-script", text)
            # Two or more distinct English marker words is real English, not the
            # one-word garble the attempt-0 guard was written for. Real call:
            # "Yeah, you can. Yes, proceed. Hello?" was refused and 25 s were
            # burned re-asking. Zero language changes across the logged calls.
            if det == "en-IN" and en_markers(text) >= 2 and not roman_tamil(text):
                return self._lock(det, "detected-en", text)
            if det in DETECT_TRUST and (attempt >= 1 or len(text.strip()) >= 12):
                return self._lock(det, "detected", text)
            if det:
                log.info("[%s] det=%s NOT trusted (guess on 8kHz) - asking again",
                         self.lead_id[:8], det)   # they answered in the language
            if attempt < max(1, LANG_TRIES) - 1:
                self.say_fixed(LANG_REPEAT)

        log.info("[%s] no clear answer after %d tries - defaulting",
                 self.lead_id[:8], max(1, LANG_TRIES))
        return self._lock(DEFAULT_LANG, "default", "")

    def _lock(self, code, how, text):
        self.lang = self.lang_locked = code
        self.lang_lock_text = text  # v74: what the customer actually said
        log.info("[%s] language LOCKED -> %s (%s)", self.lead_id[:8], code, how)
        # v59: _lock never told the model which language it had settled on -
        # only the late fallback path did. A real ta-IN call opened with
        # "Got it, sir, which city are you calling from?" in English because
        # nothing in the prompt said otherwise.
        self.msgs.append({"role": "system", "content":
                          "The customer's language is %s. Speak ONLY in it "
                          "from now on, per the language rules above." % code})
        # v62: tts_lang() can only ever relabel a reply that is ALREADY written
        # in native script -- a reply typed in romanised/Latin letters ("Romba
        # nalla irukku sir") can't be fixed after the fact by swapping the TTS
        # voice, because the voice would still be reading Latin letters, not
        # the target language's phonetics (tts_lang's own docstring says as
        # much: Latin-only text is deliberately left alone, since we can't
        # tell romanised-target-language apart from legitimate English). The
        # only real fix is stopping the model writing romanised text in the
        # first place, so name the script explicitly here.
        script_name = SCRIPT_NAME.get(code)
        if script_name:
            self.msgs.append({"role": "system", "content":
                               "Write every reply in %s script (Unicode), "
                               "never in romanised/Latin-letter spelling. "
                               "Numbers, proper nouns and brand names "
                               "(WhatsApp, CRM, Jinglu) may stay in English "
                               "letters; every other word must be in %s "
                               "script." % (script_name, script_name)})
        if self.gender:
            # v61: old text said "Address him as 'mam'" -- self-contradicting (him +
            # mam) and "mam" is not a word the model was ever told to speak. A live call
            # with gender="mam" still got answered "Sure sir". Match the script's own
            # madam/sir, மேடம்/சார் vocabulary instead.
            if self.gender == "mam":
                self.msgs[0]["content"] += (
                    "\n\nVoice-pitch guess (best effort, not certain): the customer "
                    "sounds like a woman. Say 'madam' / 'மேடம்' instead of 'sir' / "
                    "'சார்' in every line."
                )
            else:
                self.msgs[0]["content"] += (
                    "\n\nVoice-pitch guess (best effort, not certain): the customer "
                    "sounds like a man. Say 'sir' / 'சார்' in every line."
                )
        if text.strip():
            self.transcript.append({"role": "customer", "text": text,
                                    "ms": int(self.elapsed() * 1000)})
        return True

    def say_fixed(self, text):
        """A pre-rendered English line - no LLM, no TTS wait."""
        pcm = FIXED_PCM.get(text)
        if pcm:
            self.say_pcm(text, pcm)

    def _learn(self, vlv):
        """v24: how loud THIS caller is, from the median of the frames we
        just accepted as his speech."""
        if len(vlv) < MIN_VOICE_FRAMES:
            return
        med = sorted(vlv)[len(vlv) // 2]
        if not self.vlevel:
            self.vlevel = med
        else:
            a = VOICE_EMA_DOWN if med < self.vlevel else VOICE_EMA_UP
            self.vlevel = (1.0 - a) * self.vlevel + a * med
        self.relax = 1.0        # vlevel now reflects him - safe to re-arm

    def _trim(self, frames, v0, v1):
        """v19: only the stretch that actually contained voice, plus TRIM_PAD
        frames either side so we never clip the first or last syllable."""
        if not frames:
            return b""
        if v0 is None:
            return b""                      # nothing ever crossed the gate
        a = max(0, v0 - TRIM_PAD)
        b = min(len(frames), (v1 or v0) + TRIM_PAD)
        if b - a > TRIM_MAX_FRAMES:              # v26: keep the tail
            log.info("[%s] span %.1fs -> keeping last %.1fs",
                     self.lead_id[:8], (b - a) * 0.02, TRIM_MAX_FRAMES * 0.02)
            a = b - TRIM_MAX_FRAMES
        return b"".join(frames[a:b])

    def _echoing(self):
        """v37: true while our own TTS audio (or its tail) is still in the room."""
        return self.a.speaking or time.time() < getattr(self.a, "quiet_until", 0)

    def gate(self):
        """v15: the level real speech must clear right now on THIS line.

        The room is the 20th percentile of the last 3s of frames - not "the
        quiet frames", because on a loud line there are none: every frame reads
        as speech, so the floor never learns and the gate never rises. A
        percentile always finds the room, even mid-sentence, because speech has
        gaps. Floor at SILENCE_RMS, capped at NOISE_MAX so we never go deaf."""
        if len(self.lvls) >= 25:
            q = sorted(self.lvls)
            self.noise = q[len(q) // 5]
        return min(NOISE_MAX, max(SILENCE_RMS,
                                  int(self.noise * NOISE_MULT),
                                  int(self.vlevel * VOICE_RATIO * self.relax)))

    def listen(self, max_wait_ms=None):
        # v19: we used to hand STT the whole buffer. A real turn read
        # "stt 0.29s for 14.0s audio -> Hello" - 0.4s of speech and 13.6s of
        # room noise, because an occasional blip kept resetting the silence
        # counter. Sarvam bills by the hour, so that was money, latency and
        # transcript quality all thrown away. Track where the voice actually
        # is and send only that, plus a little padding.
        speech, silence_ms, spoke_ms, waited = [], 0, 0, 0
        vlv, stale = [], 0
        v0 = v1 = None
        run = 0                      # consecutive frames currently above the gate
        last_voice_ms = 0      # v78: wall-clock ms at last confirmed voice frame
        if not self.warm:
            # v15: 600ms of pure calibration, once per call, right after a 12s
            # greeting - the caller is not mid-word. Must be >= the 25 frames
            # gate() needs for its percentile, or the gate stays at its floor
            # for the first 300ms and a loud room walks straight in as "speech".
            for _ in range(max(1, CALIB_MS // 20)):
                t, p = self.a.read(timeout=10)
                if t != self.a.AUDIO:
                    continue
                if not self._echoing():
                    self.lvls.append(rms(p))
                lvl = rms(p)
                if lvl > BARGE_RMS:
                    run += 1
                    speech.append(p)
                    if run >= MIN_VOICE_FRAMES:
                        if v0 is None:
                            v0 = len(speech) - run
                        spoke_ms += 20
                        silence_ms = 0
                        v1 = len(speech)
                        vlv.append(lvl)
                elif speech:
                    run = 0
                    speech.append(p)
                    silence_ms += 20
            self.warm = True
            log.info("[%s] line calibrated: noise=%d gate=%d",
                     self.lead_id[:8], int(self.noise), self.gate())
            if spoke_ms:
                log.info("[%s] caller was already talking during calibration -- captured %.1fs",
                         self.lead_id[:8], spoke_ms / 1000.0)
        while True:
            t, payload = self.a.read(timeout=10)
            if t is None:
                return self._trim(speech, v0, v1)
            if t == self.a.DTMF:
                log.info("[%s] DTMF %s", self.lead_id[:8], payload)
                continue
            if t != self.a.AUDIO:
                continue

            waited += 20
            if max_wait_ms is not None and waited >= max_wait_ms:
                self.last_spoke_ms = spoke_ms
                return self._trim(speech, v0, v1)
            lvl = rms(payload)
            if not self._echoing():
                self.lvls.append(lvl)
            if lvl > self.gate():
                run += 1
                speech.append(payload)
                if run >= MIN_VOICE_FRAMES:          # a real syllable, not a click
                    if v0 is None:
                        v0 = len(speech) - run       # back-date to where it began
                    spoke_ms += 20
                    silence_ms = 0
                    v1 = len(speech)
                    vlv.append(lvl)                  # v24: learn his level
                    stale = 0
                    last_voice_ms = waited
                    log.info("[%s] voice-pulse: lvl=%d run=%d waited=%.1fs spoke=%.1fs gate=%d relax=%.2f",
                                self.lead_id[:8], lvl, run, waited / 1000.0, spoke_ms / 1000.0, self.gate(), self.relax)
            elif True:
                if not spoke_ms:                     # v24: nothing cleared yet
                    stale += 20
                    if stale >= VOICE_RELAX_MS and self.relax > VOICE_RELAX_MIN:
                        self.relax = max(VOICE_RELAX_MIN, self.relax * 0.6)
                        stale = 0
                if not speech:
                    continue
                run = 0
                speech.append(payload); silence_ms += 20
                if silence_ms >= SILENCE_MS and spoke_ms >= MIN_SPEECH_MS:
                    log.info("[%s] heard %.1fs speech in %.1fs (noise=%d gate=%d)",
                             self.lead_id[:8], spoke_ms / 1000.0,
                             len(speech) * 0.02, int(self.noise), self.gate())
                    self._learn(vlv)
                    self.last_spoke_ms = spoke_ms
                    return self._trim(speech, v0, v1)


            if spoke_ms >= MAX_TURN_MS:
                self.last_spoke_ms = spoke_ms
                return self._trim(speech, v0, v1)
            if spoke_ms >= MIN_SPEECH_MS and waited >= MAX_TURN_MS:
                # v85: hard wall-clock backstop. A real reply is in progress but
                # total elapsed time -- not just confirmed-voice time -- has hit
                # the MAX_TURN_MS ceiling. Pauses between bursts of speech keep
                # resetting the settle-cap below, so a multi-burst rambling reply
                # could otherwise run far longer than MAX_TURN_MS (seen: 26s).
                self.last_spoke_ms = spoke_ms
                return self._trim(speech, v0, v1)
            if spoke_ms >= MIN_SPEECH_MS and last_voice_ms and (waited - last_voice_ms) >= SETTLE_MS:
                # v78: we already have a real, confirmed reply -- an
                # occasional blip after it (line noise, a breath) was
                # re-arming silence_ms and holding the turn open far
                # longer than needed (same failure mode the v19 note
                # above already diagnosed for STT cost/quality; this
                # caps the wait itself, not just what gets sent to STT).
                log.info("[%s] settle-cap: heard %.1fs speech, %.1fs quiet-ish -> answering",
                         self.lead_id[:8], spoke_ms / 1000.0, (waited - last_voice_ms) / 1000.0)
                self._learn(vlv)
                self.last_spoke_ms = spoke_ms
                return self._trim(speech, v0, v1)

            # v56: "not speech" was unreliable - a single noise blip (click,
            # breath) that crossed the gate for even one frame made `speech`
            # non-empty forever, so this dead-air escape hatch never fired.
            # Real calls showed 21s, 22.6s waits after the opening question -
            # the caller was silent, but one stray frame blocked the wrap-up
            # for the rest of the call, all the way to MAX_CALL_SEC. Check
            # spoke_ms instead - it only counts frames that cleared
            # MIN_VOICE_FRAMES, a real syllable, not a click.
            if spoke_ms == 0 and waited >= DEAD_AIR_MS:      # v4: 10 s -> wrap up
                self.last_spoke_ms = spoke_ms
                return b""
            if self.elapsed() > MAX_CALL_SEC:
                self.last_spoke_ms = spoke_ms
                return self._trim(speech, v0, v1)

    def run(self):
        self.load_lead()
        slot = slot_now()
        # v36: speak this call's own script greeting if we managed to
        # pre-render it; otherwise fall back to the generic one so a call
        # is never left waiting on a script that failed to load.
        if self._script_id and self._script_id in SCRIPT_GREETING_PCM:
            opening = self._script_greeting
            opening_pcm = SCRIPT_GREETING_PCM[self._script_id]
        elif self._script_greeting:
            # v42: the CRM answered with a real greeting (e.g. the system
            # default script, or a script saved after the last boot) but it
            # was never pre-rendered, so the old code silently dropped it and
            # spoke the hardcoded OPENINGS line instead - the caller heard
            # something different from what the CRM UI showed. Synthesize it
            # live instead; costs ~1-2s only on this uncached path.
            opening = self._script_greeting
            try:
                opening_pcm = sarvam_tts(opening, OPENING_LANG)
            except Exception as e:
                log.warning("[%s] live greeting TTS failed (%s) - using generic",
                            self.lead_id[:8], e)
                opening = OPENINGS[slot]
                opening_pcm = OPENING_PCM[slot]
        else:
            opening = OPENINGS[slot]
            opening_pcm = OPENING_PCM[slot]
        self.msgs.append({"role": "assistant", "content": opening})
        self.say_pcm(opening, opening_pcm)

        # v75: wait for the customer to actually respond before asking
        # which language he wants.
        pcm0 = self.listen(max_wait_ms=800)  # v85: greeting-reply capped at 0.8s, continue regardless
        if self.a.closed:
            self.wrap_up()
            self.a.hangup()
            return
        _spoke_ms = 0  # v85-fix: default when pcm0 empty (0.2/0.5s cap can return no audio)
        if pcm0:
            # v80: play the nod the INSTANT speech ends, before STT runs.
            # STT costs ~1s and the caller heard nothing during it.
            _fast_ack = FIXED_PCM.get("__confirm_ack__")
            # v83: only 300ms is needed to clear the speech gate in listen -- a
            # click or breath can cross that. Require a slightly longer
            # capture before playing the filler "Ok,", so it only fires on
            # audio long enough to plausibly be a real word.
            _spoke_ms = getattr(self, "last_spoke_ms", None)
            if _spoke_ms is None:
                _spoke_ms = len(pcm0) / (RATE * 2.0) * 1000.0
            if _fast_ack and _spoke_ms >= 450:
                self.say_pcm("Ok,", _fast_ack)
            elif _fast_ack:
                log.info("[%s] skipped fast-ack, only %.0fms confirmed speech (too short to be a real reply)", self.lead_id[:8], _spoke_ms)
        
        if LANG_Q_PCM:
            self.say_pcm(LANG_Q_TEXT, LANG_Q_PCM)               # ~0.2 s after answer

        try:
            text0, _ = sarvam_stt(pcm0, None)
        except Exception as e:
            text0 = ""
            log.warning("[%s] pre-lang greeting-reply STT failed: %s",
                        self.lead_id[:8], e)
        text0 = (text0 or "").strip()
        if _spoke_ms < 450 and text0:
            log.info("[%s] discarding greeting-reply text %r, only %.0fms confirmed speech (unreliable STT)", self.lead_id[:8], text0[:60], _spoke_ms)
            text0 = ""
        if text0:
            log.info("[%s] greeting reply: %r -> decline=%s",
                      self.lead_id[:8], text0[:60], sounds_like_decline(text0))
            if sounds_like_decline(text0):
                self.msgs.append({"role": "user", "content": text0})
                self.bye()
                self.wrap_up()
                self.a.hangup()
                return
            if lang_from_words(text0) is None:
                self.msgs.append({"role": "user", "content": text0})
                if sounds_like_confirm(text0) and not (_fast_ack and _spoke_ms >= 450):
                    # v79: "Yes/ok" to "good time to speak" was silently
                    # dropped and the call jumped straight to the language
                    # question with no acknowledgment - confusing for the
                    # caller who just answered. Give a one-word nod first.
                    ack = FIXED_PCM.get("__confirm_ack__")
                    if ack:
                        self.say_pcm("Ok,", ack)

            else:
                # v85: greeting-reply already named a language and passed the
                # reliability bar -- lock it now instead of asking again and
                # re-listening from scratch.
                _said0 = lang_from_words(text0)
                if _said0 in ALLOWED_LANGS:
                    self._lock(_said0, "greeting-reply", text0)
        # v13: settle the language first. Everything after this runs in one
        # language - no drifting, no guessing mid-sale.
        if not (getattr(self, "lang_locked", None) or self.pick_language()):  # v85: skip re-asking if greeting-reply already locked it
            self.wrap_up()
            self.a.hangup()
            return
        heard = getattr(self, "lang_lock_text", "") or ""
        heard = heard.strip()
        if heard and lang_from_words(heard) is None:
            self.msgs.append({"role": "user", "content": heard})
        self.msgs.append({"role": "user",
                "content": "[He was greeted. Do not greet him again. Reply only in the language HE used, and never mention language.]"})

        # v33: speak Turn 2 right away instead of waiting in silence for the
        # customer to talk first - that was a 20-30s dead gap on real calls.
        # LLM-generated so per-tenant scripts still vary; falls back to a
        # pre-rendered fixed line only if the LLM call itself errors out.
        try:
            self.think()
            reply = self.stream_say()
            if reply:
                self.msgs.append({"role": "assistant", "content": reply})
                if reply.lstrip().startswith("{"):
                    return self.finish(reply)
        except Exception as e:
            log.error("[%s] Turn 2 LLM failed (%s) - using fallback line",
                       self.lead_id[:8], e)
            fallback = TURN2_FALLBACK.get(self.lang, TURN2_FALLBACK["en-IN"])
            self.say_fixed(fallback)
            self.msgs.append({"role": "assistant", "content": fallback})

        while not self.a.closed and self.elapsed() < MAX_CALL_SEC:
            pcm = self.listen()
            if self.a.closed:
                break
            if not pcm:
                log.info("[%s] %ds silence - closing", self.lead_id[:8], DEAD_AIR_MS // 1000)
                self.bye()
                break

            # v8: measure STT. "first audio in" only ever covered LLM+TTS, so the
            # number looked fine (1.1s) while the caller was really waiting
            # 650ms silence + STT + 1.1s. You cannot fix what you do not measure.
            t_stt = time.time()
            try:
                text, det = sarvam_stt(pcm, self.lang_locked)
            except Exception as e:
                log.error("STT failed: %s", e)
                continue
                if text and not script_ok(text, self.lang_locked):
                    # v58: wrong script for this call. Retry the turn once on
                    # auto-detect; if that is no better, drop it to the repair
                    # ladder rather than let another language into the call.
                    try:
                        alt, _alt_det = sarvam_stt(pcm, None)
                    except Exception:
                        alt = ""
                    if alt and script_ok(alt, self.lang_locked):
                        log.info("[%s] script rescue %r -> %r",
                                 self.lead_id[:8], text[:40], alt[:40])
                        text = alt
                    else:
                        log.warning("[%s] off-script turn dropped: %r",
                                    self.lead_id[:8], text[:60])
                        text = ""
            self.t_stt = time.time() - t_stt
            log.info("[%s] stt %.2fs for %.1fs audio", self.lead_id[:8],
                     self.t_stt, len(pcm) / (RATE * 2.0))
            if not text:
                continue

            if self.lang_locked is None:                  # v5: decide once, then stick
                said = lang_from_words(text)              # v9: what they ANSWERED wins
                if said:
                    self.lang_locked, how = said, "asked"
                elif det in DETECT_TRUST and len(text.strip()) >= 15:
                    # v31: Sarvam renders short Tamil as English ('Aa, that's
                    # it.') and we locked en-IN off it. Short answers are not
                    # evidence - fall through to the default instead.
                    self.lang_locked, how = det, "detected"
                else:
                    if det:
                        log.warning("[%s] STT guessed %s - outside our geography",
                                    self.lead_id[:8], det)
                    self.lang_locked, how = DEFAULT_LANG, "default"
                self.lang = self.lang_locked
                log.info("[%s] language LOCKED -> %s (%s)", self.lead_id[:8], self.lang, how)
                self.msgs.append({"role": "system", "content":
                                  "The customer's language is %s. Speak ONLY "
                                  "in it from now on, per the language rules "
                                  "above." % self.lang})


            # v49: mid-call switching restored, but ONLY on an explicit
            # keyword match (lang_from_words - he says "Tamil" or "தமிழ்"
            # by name). v48 removed this entirely after a call drifted to
            # Malayalam mid-conversation, but that drift came from trusting
            # SARVAM's auto-detected script guess elsewhere, not from this
            # keyword check - lang_from_words only fires when he names the
            # language outright, so it is safe to restore. A customer saying
            # "I speak in Tamil" must be able to switch immediately, without
            # an extra confirmation turn - that costs more than it protects.
            want = lang_from_words(text)
            if want and want != self.lang_locked:
                if want in ALLOWED_LANGS:
                    log.info("[%s] customer asked for %s mid-call -> SWITCHING",
                             self.lead_id[:8], want)
                    self.lang = self.lang_locked = want
                    self.msgs.append({"role": "system", "content":
                                      "The customer just asked you to speak %s. "
                                      "Switch to it now and stay in it. Do not "
                                      "comment on the switch." % want})
                else:
                    log.info("[%s] asked for %s - not supported",
                             self.lead_id[:8], want)
                    self.say_fixed(LANG_NO_SUPPORT)
                    self.lang = self.lang_locked = "en-IN"

            log.info("[%s] CUSTOMER (%s): %s", self.lead_id[:8], self.lang, text)
            self.transcript.append({"role": "customer", "text": text,
                                    "ms": int(self.elapsed() * 1000)})
            self.turns += 1
            n = extract_need(text)
            if n and not self.slots["need"] and not is_filler(text):
                self.slots["need"] = n
            c = extract_city(text)
            if c and not self.slots["city"] and not is_filler(text):
                self.slots["city"] = c
            # v51: "புரியல", "hello", "sorry" etc are NOT answers - they are
            # the customer signalling confusion. A real call showed the LLM
            # treating these as "he answered, move to the next question",
            # skipping through five topics on pure filler. Flag it so the
            # state message can tell the LLM explicitly: repeat the SAME
            # question, do not advance the flow.
            # v55: SILENCE_MS being too short (350ms) cut real words
            # mid-syllable - "புரியலை" (I don't understand) came back from
            # STT as fragments like "Puri", "Puriya", "Puri" which the bot
            # then treated as a CITY NAME ("Puri, correct sir?"). Catch the
            # romanized fragments too, not just the full Tamil word.
        # v64: NON_ANSWER (dead since v60) removed -- is_filler()/FILLER
        # above is what actually flags these now; pyflakes confirmed
        # NON_ANSWER and clean_low were assigned but never read.
            is_non_answer = is_filler(text)
            self.msgs.append({"role": "user", "content": text})
            if is_non_answer:
                self.msgs.append({"role": "system", "content":
                    "[That reply was confusion/filler (\"puriyala\", "
                    "\"hello\", \"sorry\" etc), NOT an answer to your "
                    "question. Do NOT treat it as answered and do NOT move "
                    "to the next question or topic. Repeat your SAME last "
                    "question, shorter or as a two-way choice, per the "
                    "repair ladder rules.]"})
            else:
                # v52: reminder on every turn, not just once in the system
                # prompt - a long call drowns out a rule stated only at the
                # top. This is what stops the "Vellore" bug: he said
                # something unclear, the LLM invented a city he never said
                # instead of asking again.
                self.msgs.append({"role": "system", "content":
                    "[Before replying: are you CONFIDENT you understood "
                    "what he just said? If his words are garbled, unclear, "
                    "or you are only guessing at the meaning, do NOT invent "
                    "or assume an answer (a city, a need, a number he did "
                    "not clearly say). Instead confirm your best guess as a "
                    "short question, or ask again, per the repair ladder.]"})
            # v54: real calls locked to ta-IN kept replying almost entirely
            # in English ("Which city sir, Coimbatore or another place?") -
            # confirmation lines and script wording were being spoken
            # literally instead of translated. Remind every single turn,
            # not just at lock time.
            if self.lang == "ta-IN":
                self.msgs.append({"role": "system", "content":
                    "[The locked language is Tamil. Your reply must be "
                    "SPOKEN TAMIL IN TAMIL SCRIPT, not English sentences "
                    "with a couple of Tamil words tacked on. Confirmations "
                    "like \"correct sir?\" become \"சரிதானா sir?\", "
                    "\"how many rooms\" becomes \"எத்தனை rooms\", city/"
                    "number confirmations get said in தமிழ் too - only "
                    "product words (website, software, WhatsApp) and "
                    "numbers-as-digits stay in English letters.]"})
            self.msgs.append({"role": "system", "content":
                "[State: need=%s, city=%s, asked_counts: need=%d city=%d, "
                "once_more_already_used=%s. RULE: if a slot already has a "
                "value here, it is ANSWERED - do not ask for it again, move "
                "to the next unknown slot or the next stage of the flow. If "
                "a slot is unknown and its asked_count >= 2, mark it "
                "unanswered and move on - never ask a third time. RULE: if "
                "once_more_already_used is True, you have ALREADY used "
                "'Sorry sir, once more?' this call - do NOT say it or any "
                "variant of it again, no matter how garbled his answer was. "
                "Instead pick your best guess and confirm it, or move to the "
                "next question.]" % (self.slots["need"] or "unknown", self.slots["city"] or "unknown",
                            self.asked["need"], self.asked["city"],
                            self.once_more_used)})
            if self.think(text):         # v21/A1: talk while we think
                self.msgs.append({"role": "system", "content":
                                  "An acknowledgment word was already spoken - "
                                  "begin your reply directly with the content, "
                                  "no opening word."})

            # v5: a 429 or any blip used to kill the whole call mid-conversation.
            # v6: streams the reply into TTS as it is written.
            try:
                reply = self.stream_say()
                if self.done:                # v26: it just closed the call
                    log.info("[%s] verdict emitted - hanging up",
                             self.lead_id[:8])
                    break
            except Exception as e:
                self.llm_fails += 1
                log.error("[%s] LLM failed (%d/%d): %s",
                          self.lead_id[:8], self.llm_fails, MAX_LLM_FAILS, e)
                if self.llm_fails >= MAX_LLM_FAILS:
                    self.bye()
                    break
                continue
            self.llm_fails = 0
            if not reply:
                continue
            self.msgs.append({"role": "assistant", "content": reply})
            rlow = reply.lower()
            if not self.slots["need"] and any(k in rlow for k in
                    ("என்ன தேவை", "what do you need", "what service", "hotel software, or")):
                self.asked["need"] += 1
            if not self.slots["city"] and any(k in rlow for k in
                    ("நகர", "எந்த ஊர்", "which city", "city are you")):
                self.asked["city"] += 1
            # v50: track whether "once more?" (in any language) has already
            # been used this call, so the state message can tell the LLM to
            # stop reaching for it - the prompt rule alone was not holding,
            # a real call used it 5+ times.
            if any(k in rlow for k in
                    ("once more", "ஒரு முறை", "மீண்டும்", "again sir")):
                self.once_more_used = True
            if reply.lstrip().startswith("{"):
                return self.finish(reply)

        # v47: falling out of the while-loop on MAX_CALL_SEC used to hang up
        # with no goodbye - the call went dead on the caller mid-sentence.
        # bye() speaks the pre-rendered sign-off in whatever language was
        # locked, so this stays professional and in-language.
        if not self.a.closed:
            log.info("[%s] call limit reached - saying goodbye", self.lead_id[:8])
            self.bye()

        self.wrap_up()
        self.a.hangup()

    def finish(self, raw):
        try:
            self.save(json.loads(_json_slice(raw)))
        except Exception as e:
            log.error("verdict parse failed: %s", e)
            return self.wrap_up()
        self.a.hangup()

    def wrap_up(self):
        if not self.transcript or self.turns == 0:
            # v46: removed dead Supabase record_call_result call - CRM save via
            # CRM_URL (app/api/calls/report) already handles this by callUuid lookup.
            return
        msgs = self.msgs + [
            {"role": "system", "content": VERDICT_SYS},
            {"role": "user", "content": "[CALL ENDED. Output the verdict JSON now; "
                                        "null for anything you did not learn.]"},
        ]
        try:
            res = json.loads(_json_slice(groq_chat(msgs, max_tokens=VERDICT_TOKENS)))
            # v38: cross-check the LLM's free-text requirement against what the
            # customer actually said, keyword-matched. Only overrides on disagreement.
            kw = None
            for t in self.transcript:
                if t["role"] == "customer":
                    kw = requirement_from_words(t["text"])
                    if kw:
                        break
            if kw and res.get("requirement") and kw.lower() not in res["requirement"].lower():
                log.info("requirement keyword override: llm=%r kw=%r", res.get("requirement"), kw)
                res["requirement"] = kw
            self.save(res)
            return
        except Exception as e:
            log.error("verdict failed (%s) - saving the call anyway", e)
        # v18: an answered call must NEVER vanish because the model would not
        # produce JSON. Keep the transcript; a human can read it and judge.
        self.save({"qualified": False, "disqualify_reason": "verdict_failed",
                   "ai_summary": " | ".join(t["text"] for t in self.transcript
                                            if t["role"] == "customer")[:500]})

    def save(self, d):
        d["lead_id"]      = self.lead_id
        d["transcript"]   = self.transcript
        d["turns"]        = self.turns
        d["duration_sec"] = int(self.elapsed())
        d.setdefault("language", self.lang)
        try:
            _body = json.dumps({
                "uuid": self.lead_id,
                "outcome": "CONNECTED" if self.turns else "NO_ANSWER",
                "duration": int(self.elapsed()),
                "notes": d.get("ai_summary") or "",
                "transcript": json.dumps(self.transcript, ensure_ascii=False),
            }).encode()
            _r = urllib.request.Request(CRM_URL, data=_body,
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(_r, timeout=5).read()
            log.info("[%s] CRM saved", self.lead_id[:8])
        except Exception as _e:
            log.info("[%s] CRM save skipped: %s", self.lead_id[:8], _e)
        # v46: removed dead Supabase save_qualification call - CRM_URL POST
        # above already persists the call outcome via callUuid lookup.
        log.info("[%s] verdict qualified=%s turns=%d %.0fs",
                  self.lead_id[:8], d.get("qualified"), self.turns, self.elapsed())


def _json_slice(s):
    return s[s.index("{"):s.rindex("}") + 1]


# ------------------------------------------------------------------ server
def handle(conn, addr, prompt):
    a = AudioSocket(conn)
    lead_id = None
    try:
        for _ in range(100):
            if a.uuid is not None or a.closed:
                break
            time.sleep(0.02)
        if not a.uuid:
            log.error("no UUID from %s", addr); return
        lead_id = str(uuid.UUID(bytes=a.uuid)) if len(a.uuid) == 16 \
            else a.uuid.decode(errors="ignore")
        log.info("=== CALL START lead=%s from %s", lead_id, addr)
        Call(a, lead_id, prompt).run()
    except Exception as e:
        log.exception("call %s crashed: %s", lead_id, e)
    finally:
        try:
            conn.close()
        except OSError:
            pass
        log.info("=== CALL END lead=%s", lead_id)


def _refresh_fixed_phrases_from_db():
    """v70: pull the fixed voice-prompt phrases (language retry, no-support
    message, turn-2 fallback, goodbye, thinking filler words) from the CRM
    website's Setting table, at BOOT time only. These get baked into
    pre-rendered audio by prerender() right after this call, so a website
    edit to any of these still needs a service restart to take effect --
    same restart-required category as the TTS voice/speaker settings.
    Wrapped so any failure leaves the existing hardcoded phrases in place;
    this can never break boot or a call.
    """
    global LANG_REPEAT, LANG_NO_SUPPORT, TURN2_FALLBACK, GOODBYE, THINKING, LANG_Q
    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
        cur = conn.cursor()
        cur.execute(
            "SELECT tenantId, key, value FROM Setting WHERE tenantId IS NULL OR tenantId=?",
            (_SETTINGS_TENANT_ID,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        try:
            log.warning(f"_refresh_fixed_phrases_from_db: DB read failed, keeping hardcoded phrases: {e}")
        except Exception:
            pass
        return

    vals = {}
    for tid, k, v in rows:
        if tid is None:
            vals.setdefault(k, v)
    for tid, k, v in rows:
        if tid is not None:
            vals[k] = v

    def _parse_lang_dict(text, joiner=None):
        result = {}
        for line in text.splitlines():
            if not line.strip() or ":" not in line:
                continue
            lang, val = line.split(":", 1)
            lang = lang.strip()
            val = val.strip()
            if joiner is not None:
                val = val.split(joiner)
            result[lang] = val
        return result

    try:
        v = vals.get("callLangQ")
        if v and v.strip():
            LANG_Q = v.strip()
        v = vals.get("callLangRepeat")
        if v and v.strip():
            LANG_REPEAT = v.strip()
        v = vals.get("callLangNoSupport")
        if v and v.strip():
            LANG_NO_SUPPORT = v.strip()
        v = vals.get("callTurn2Fallback")
        if v and v.strip():
            parsed = _parse_lang_dict(v)
            if parsed:
                TURN2_FALLBACK = parsed
        v = vals.get("callGoodbye")
        if v and v.strip():
            parsed = _parse_lang_dict(v)
            if parsed:
                GOODBYE = parsed
        v = vals.get("callThinkingWords")
        if v and v.strip():
            parsed = _parse_lang_dict(v, joiner="|")
            if parsed:
                THINKING = parsed
    except Exception as e:
        try:
            log.warning(f"_refresh_fixed_phrases_from_db: apply failed, keeping hardcoded phrases: {e}")
        except Exception:
            pass


LANG_Q_PCM = None
LANG_Q_TEXT = ""


# v75: a small, high-precision reschedule-phrase list, used by run()
# right after the greeting.
DECLINE_MARKERS = (
    "busy", "not free", "not now", "no time", "not a good time",
    "cant talk", "can't talk", "cannot talk",
    "call later", "call me later", "call back later", "call again later",
    "another time", "call tomorrow", "not interested",
    # v82: Tamil decline phrases (greeting-reply step was English-only).
    "நேரம் இல்ல", "பிசியா இருக்கேன்", "இப்போ வேண்டாம்",
    "இன்ட்ரெஸ்ட் இல்ல", "வேண்டாம் சார்", "பின்னாடி கூப்பிடுங்க",
)


def sounds_like_decline(text):
    t = (text or "").lower()
    return any(m in t for m in DECLINE_MARKERS)

CONFIRM_MARKERS = ("yes", "yeah", "yep", "sure", "ok", "okay", "correct",
                    "aama", "sari", "seri", "ஆமா", "சரி")
def sounds_like_confirm(text):
    t = (text or "").lower()
    return any(m in t for m in CONFIRM_MARKERS)


def prerender():
    global LANG_Q_PCM, LANG_Q_TEXT
    t0 = time.time()
    for k, v in OPENINGS.items():
        OPENING_PCM[k] = sarvam_tts(v, OPENING_LANG)      # v10: greet in English
        log.info("pre-rendered %-9s %.1fs audio", k, len(OPENING_PCM[k]) / (RATE * 2.0))
    for code, txt in GOODBYE.items():                     # v10: one per language
        GOODBYE_PCM[code] = sarvam_tts(txt, code)
        # v86: "madam" variant for the languages we have a vetted
        # sir/madam word pair for (same vocabulary already used to
        # steer the live LLM -- see the self.gender == "mam" block
        # below). Every other language keeps the "sir" clip either
        # way -- not guessing a translation without evidence.
    _MADAM_SWAP = {"ta-IN": ('சார்', 'மேடம்'), "en-IN": (" sir", " madam")}  # v86
    for _mcode, (_w_sir, _w_madam) in _MADAM_SWAP.items():
        if _mcode in GOODBYE:
            _madam_txt = GOODBYE[_mcode].replace(_w_sir, _w_madam)
            if _madam_txt != GOODBYE[_mcode]:
                GOODBYE_PCM_MAM[_mcode] = sarvam_tts(_madam_txt, _mcode)
                log.info("pre-rendered goodbye (madam variant) %s", _mcode)

    for txt in (LANG_REPEAT, LANG_NO_SUPPORT):            # v13: language sub-flow
        FIXED_PCM[txt] = sarvam_tts(txt, OPENING_LANG)
    try:                                                  # v79: confirm nod
        FIXED_PCM["__confirm_ack__"] = sarvam_tts("Ok,", OPENING_LANG)
        log.info("pre-rendered confirm ack")
    except Exception as e:
        log.warning("confirm ack pre-render failed: %s", e)
    # v72: LANG_Q (website setting callLangQ) was never spoken.
    try:
        if LANG_Q and LANG_Q.strip():
            LANG_Q_TEXT = LANG_Q.strip()
            LANG_Q_PCM = sarvam_tts(LANG_Q_TEXT, OPENING_LANG)
            log.info("pre-rendered language question (%.1fs audio)",
                     len(LANG_Q_PCM) / (RATE * 2.0))
    except Exception as e:
        LANG_Q_PCM, LANG_Q_TEXT = None, ""
        log.warning("lang question pre-render failed, not asking it: %s", e)
    for code, txt in TURN2_FALLBACK.items():              # v33: Turn 2 safety net
        FIXED_PCM[txt] = sarvam_tts(txt, code)
    for code, words in THINKING.items():                  # v21: backchannel
        THINK_PCM[code] = [sarvam_tts(w, code) for w in words]
    for script_id, greeting in load_active_scripts():     # v36: per-script greeting
        SCRIPT_GREETING_PCM[script_id] = sarvam_tts(greeting, OPENING_LANG)
        log.info("pre-rendered script greeting %s (%.1fs audio)", script_id[:8],
                 len(SCRIPT_GREETING_PCM[script_id]) / (RATE * 2.0))
    log.info("pre-render done in %.1fs - greeting now plays instantly", time.time() - t0)


def main():
    missing = [k for k, v in (("SARVAM_KEY", SARVAM_KEY), ("GROQ_KEY", GROQ_KEY),
                              ("SUPABASE_KEY", SUPA_KEY)) if not v]
    if missing:
        log.error("missing env: %s - check /etc/ailc/env", ", ".join(missing)); sys.exit(1)
    with open(PROMPT_FILE) as f:
        prompt = f.read()
    log.info("Jinglu v29 prompt loaded (%d chars)", len(prompt))
    _refresh_fixed_phrases_from_db()  # v70: hot-apply website phrase edits before baking audio
    prerender()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((LISTEN_HOST, LISTEN_PORT)); s.listen(20)
    log.info("Jinglu v29 listening on %s:%d (tts=%s/%s stt=%s llm=%s dead_air=%ds lang=%s)",
             LISTEN_HOST, LISTEN_PORT, TTS_MODEL, TTS_SPEAKER, STT_MODEL, LLM_MODEL,
             DEAD_AIR_MS // 1000, DEFAULT_LANG)
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle, args=(conn, addr, prompt), daemon=True).start()


if __name__ == "__main__":
    main()
