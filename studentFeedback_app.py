"""
Automated Student Feedback Analysis and Sentiment Evaluation System
===================================================================

A single-file application covering all three phases:

  Phase 1  Data generation & preprocessing
           generate_synthetic_data()  -> 400-row synthetic feedback dataset (English, Hinglish, Hindi)
           load_feedback()            -> read a CSV (file path or browser upload)
           clean_feedback()           -> duplicates, missing/invalid scores, blank remarks
           preprocess_text()          -> lower-casing + stop-word removal
  Phase 2  Grading & sentiment analysis
           compute_overall_grade()    -> weighted average (50/30/20) -> grade A/B/C/D
           detect_language()          -> English / Hinglish / Hindi for each remark
           analyze_sentiment()        -> VADER polarity -> Positive / Neutral / Negative
                                         (Hinglish & Hindi words mapped to English first)
           gemini_label_batch()       -> optional Gemini engine for Hinglish/Hindi or all remarks
           top_terms()                -> recurring themes (scikit-learn bag-of-words)
  Phase 3  Interactive dashboard
           run_dashboard()            -> Streamlit UI: upload, KPIs, charts, negative remarks

How to run
----------
    pip install -r requirements.txt
    python feedback_app.py            # Phase 1 script: writes student_feedback.csv (400 rows)
    streamlit run feedback_app.py     # Dashboard: upload that CSV, or use the built-in sample

Gemini (optional): get a key at https://aistudio.google.com/apikey, then either paste it in the
sidebar or set the GEMINI_API_KEY environment variable before starting Streamlit.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import string
import unicodedata

import altair as alt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from streamlit import runtime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

if not runtime.exists():
    # Plain `python feedback_app.py` run (no Streamlit server): hide harmless cache notices
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

# =============================================================================
# CONFIGURATION — adjust these to your department's evaluation policy
# =============================================================================
SCORE_COLUMNS = ["Teaching_Score", "Course_Content_Score", "Lab_Facility_Score"]
REQUIRED_COLUMNS = ["Student_ID", "Course_Code", *SCORE_COLUMNS, "Remarks"]
LABELS = {"Teaching_Score": "Teaching", "Course_Content_Score": "Content", "Lab_Facility_Score": "Lab"}

# Weights for the Overall_Grade (normalised automatically, so they need not sum to 1)
GRADE_WEIGHTS = {"Teaching_Score": 0.50, "Course_Content_Score": 0.30, "Lab_Facility_Score": 0.20}

# Weighted score (1–5 scale) -> letter grade. The first band whose cut-off is reached wins.
GRADE_BANDS = [(4.0, "A"), (3.0, "B"), (2.0, "C"), (0.0, "D")]
GRADE_ORDER = [grade for _, grade in GRADE_BANDS]

# VADER's recommended cut-offs on its compound polarity score (range -1 ... +1)
POSITIVE_CUTOFF, NEGATIVE_CUTOFF = 0.05, -0.05
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
NO_REMARK = "No Remark"  # blank remarks get this label and are left out of the sentiment breakdown

# Okabe–Ito colour-blind-safe palette
GRADE_COLORS = ["#009E73", "#56B4E9", "#E69F00", "#D55E00"]
SENTIMENT_COLORS = ["#009E73", "#999999", "#D55E00"]

# Words never treated as stop-words: negations flip meaning ("not helpful"), and the
# rest are standard stop-list entries that matter in engineering feedback.
NEGATIONS = {"no", "not", "nor", "never", "cannot"}
KEEP_WORDS = NEGATIONS | {"system", "interest", "detail"}
# Generic feedback words removed so they don't crowd the theme charts
DOMAIN_STOPWORDS = {"course", "subject", "class", "classes", "topic", "topics", "sir", "madam", "mam", "ma",
                    "maam", "semester", "also", "really", "overall", "much", "almost", "mostly", "every",
                    "always", "often", "frequently", "rarely", "made", "making", "felt", "seems", "enough",
                    "everyone", "everything", "lot", "lots", "thing", "things", "way"}
# What students type when they have nothing to say — treated as a blank remark
PLACEHOLDER_REMARKS = {"", "na", "n/a", "nil", "none", "nothing", "no comment", "no comments",
                       "no remark", "no remarks"}

# Engineering-feedback words missing from VADER's general-purpose lexicon (scale -4 ... +4)
DOMAIN_LEXICON = {"outdated": -1.3, "obsolete": -1.3, "slow": -1.0, "rushed": -1.0, "irrelevant": -1.2,
                  "unhelpful": -1.6, "insufficient": -1.3, "excessive": -1.0, "overcrowded": -1.3,
                  "disorganised": -1.5, "disorganized": -1.5, "monotonous": -1.3, "uninterested": -1.2,
                  "knowledgeable": 1.8, "informative": 1.6, "interactive": 1.2}

# Gemini (optional engine). Google renames models regularly — the name can be changed in the sidebar.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_BATCH_SIZE = 50  # remarks per API request
ENGINE_OFFLINE = "Offline: VADER + Hinglish lexicon"
ENGINE_GEMINI_HINDI = "Gemini for Hinglish & Hindi"
ENGINE_GEMINI_ALL = "Gemini for all remarks"


# =============================================================================
# LANGUAGE RESOURCES — Hinglish (Hindi typed in Roman script) and Hindi (Devanagari)
# =============================================================================
def _nfc(text: str) -> str:
    """Unicode-normalise so the same Devanagari word always has the same code points."""
    return unicodedata.normalize("NFC", text)


# English word VADER understands  <-  Hindi/Hinglish spellings that mean it
HINDI_TO_ENGLISH = {
    "good":       "accha acha achha achcha acche ache achhe acchi achi achhi अच्छा अच्छे अच्छी",
    "excellent":  "badhiya badiya badhia badia behtareen behtarin shandar shaandar बढ़िया शानदार बेहतरीन",
    "great":      "mast zabardast jabardast ज़बरदस्त जबरदस्त",
    "fine":       "sahi सही",
    "better":     "behtar बेहतर",
    "like":       "pasand पसंद",
    "fun":        "maza mazaa maja majaa मज़ा मजा",
    "easy":       "aasan asaan aasaan saral आसान सरल",
    "help":       "madad मदद",
    "thanks":     "dhanyavad dhanyawad dhanyavaad shukriya धन्यवाद शुक्रिया",
    "happy":      "khush ख़ुश खुश",
    "bad":        "kharab kharaab bura buri bure ख़राब खराब बुरा बुरी बुरे",
    "useless":    "bekar bekaar faltu faaltu बेकार फ़ालतू फालतू",
    "terrible":   "bakwas bakwaas ghatiya ghatia बकवास घटिया",
    "dirty":      "ganda gandi gande गंदा गंदी गंदे",
    "difficult":  "mushkil kathin मुश्किल कठिन",
    "problem":    "dikkat dikat pareshani pareshaani takleef taklif samasya दिक्कत परेशानी तकलीफ़ तकलीफ समस्या",
    "weak":       "kamzor kamjor कमज़ोर कमजोर",
    "wrong":      "galat ग़लत गलत",
    "outdated":   "purana purane purani पुराना पुराने पुरानी",
    "very":       "bahut bohot bahot bohut bhot bht बहुत",
    "extremely":  "ekdum ekdam एकदम",
    "completely": "bilkul बिल्कुल बिलकुल",
    "quite":      "kaafi kafi काफ़ी काफी",
    "slightly":   "thoda thodi thode थोड़ा थोड़ी थोड़े",
    "but":        "lekin magar kintu par लेकिन मगर किंतु परंतु पर",
    "not":        "nahi nahin nhi nahee नहीं नही",
}
HINDI_WORDS = {_nfc(spelling): english for english, spellings in HINDI_TO_ENGLISH.items()
               for spelling in spellings.split()}
# Also English words ("ache" = pain, "par", "mast"): only mapped when the remark is Hinglish
AMBIGUOUS_HINDI = {"ache", "par", "mast"}

# Idioms that word-by-word mapping would miss, rewritten to one English word VADER scores
NAHI = r"(?:nahi|nahin|nhi)"
PHRASE_RULES = [
    (r"\b(?:do|does|did|is|are|was|were)(?:\s+not|n't)\s+work(?:ing)?\b", "broken"),     # "PCs don't work"
    (r"\bnot\s+working\b", "broken"),
    (rf"\b(?:kaam|work)\s+(?:hi\s+)?{NAHI}\s+(?:kar|kr)\w*", "broken"),                     # "kaam nahi karte"
    (rf"\bchal(?:ta|te|ti|a)?\s+(?:hi\s+)?{NAHI}\b|\b{NAHI}\s+chal(?:ta|te|ti|a)?\b", "broken"),  # "chalta hi nahi"
    (rf"\b(?:samajh|samjh|samaj)\w*\s+(?:(?:me|mein|main|mai)\s+)?(?:hi\s+)?{NAHI}\b"
     rf"|\b{NAHI}\s+(?:samajh|samjh)\w*", "confusing"),                                      # "samajh nahi aata"
    (rf"\bkuch\s+(?:bhi\s+)?{NAHI}\s+(?:sikh|seekh|padh)\w*", "useless"),                   # "kuch nahi seekha"
    (rf"\b(?:dhyan|dhyaan)\s+(?:hi\s+)?{NAHI}\s+de\w*", "ignore"),                          # "dhyan nahi dete"
    (r"\b(?:seekhne|sikhne)\s+ko\s+mil\w*", "useful"),                                      # "seekhne ko mila"
    (r"समझ\S*\s*(?:में\s*)?(?:ही\s*)?नहीं", "confusing"),
    (r"काम\s*(?:ही\s*)?नहीं\s*कर\S*", "broken"),
    (r"चल\S*\s*(?:ही\s*)?नहीं|नहीं\s*चल\S*", "broken"),
]

# Hindi function words: removed during preprocessing so they don't show up as "themes"
HINDI_STOPWORDS = frozenset(_nfc(w) for w in """
    hai hain ho hota hoti hote tha thi raha rahe rahi kar karna karne karte karta karti kiya kiye
    ka ke ki ko se me mein main mai mujhe hum hame humein humko hamara hamare aap unka unke unki
    inka inke iska iske uska uske yeh ye woh wo vo jo jab tab agar toh bhi aur ya lekin magar par
    pe hi koi kuch sab sabhi bahut bohot bahot bhot kaafi kafi thoda thodi zyada jyada ek do wala
    wale wali liye diya diye gaya gaye gayi jata jati jate lagta lagti laga lagi lage aata aati aaya
    aaye hua hui hue apna apne apni kya kyun kyunki kyuki kaise ji haan sirf bas baar
    है हैं हो होता होती होते था थी थे रहा रहे रही कर करना करते करता करती किया की का के को से में
    मैं मुझे हम हमें आप उनका उनके इसका यह ये वह वो जो जब तब अगर तो भी और या लेकिन पर ही कोई कुछ
    सब बहुत काफी थोड़ा एक लिए गया गई जाता जाती लगता लगा आता आती हुआ हुई अपना अपने क्या क्यों कैसे जी
""".split())
# Romanised Hindi words that signal a Hinglish remark (minus short words English also uses)
HINGLISH_MARKERS = frozenset(
    ({w for w in HINDI_STOPWORDS | set(HINDI_WORDS) if w.isascii()}
     | set("padhai padhate padhane padhte padhaya samajh samjhate tarika theek thik thaak baaki baki "
           "jaisa jaise jaisi pichle saal hafte roz dete deti poochne puchne".split()))
    - {"to", "the", "me", "main", "do", "hi", "ya", "bas", "pe", "so", "ji", "ek"} - AMBIGUOUS_HINDI
)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
PUNCTUATION = string.punctuation + "।॥“”‘’…"


# =============================================================================
# PHASE 1A — SYNTHETIC DATA GENERATION
# =============================================================================
# Realistic remarks about engineering courses, grouped by (tone, aspect). The aspect
# tells the generator which score a remark should pull up or down — e.g. a complaint
# about lab PCs lowers Lab_Facility_Score for that response.
REMARK_BANK: dict[tuple[str, str], list[str]] = {
    ("Positive", "teaching"): [
        "The professor explains every concept clearly with helpful real-world examples.",
        "Excellent teaching! All our doubts are cleared patiently.",
        "Lectures are engaging, well structured and easy to follow.",
        "Sir is very supportive and always encourages us to ask questions.",
        "Great teaching style; the live coding demos made the topics easy to understand.",
        "The faculty is brilliant and the classes are very interactive.",
    ],
    ("Positive", "content"): [
        "The syllabus is well designed and relevant to industry needs.",
        "Course content is up to date and really useful for placements.",
        "Assignments were challenging but helped me understand the subject deeply.",
        "Good balance between theory and practical applications.",
        "The notes and study material provided were excellent.",
        "Loved the mini project, it connected all the topics nicely.",
    ],
    ("Positive", "lab"): [
        "Lab sessions were well organised and every experiment worked smoothly.",
        "The lab assistants were helpful and the systems are fast.",
        "Hands-on lab work was great and made the theory much clearer.",
        "Great lab infrastructure with enough computers for everyone.",
        "Practical sessions were fun and the lab setup is excellent.",
        "Good internet speed and updated software in the lab.",
    ],
    ("Neutral", "teaching"): [
        "The lectures covered the syllabus as planned.",
        "Classes were held regularly according to the timetable.",
        "The teaching pace was average.",
        "Lectures were mostly theory based.",
        "The instructor followed the prescribed textbook.",
        "Classes were taken using slides and the blackboard.",
    ],
    ("Neutral", "content"): [
        "The course content is standard for a third-year subject.",
        "The syllabus is similar to other universities.",
        "Some topics overlapped with the previous semester.",
        "The course had five units and two mid-semester tests.",
        "Content was as described in the course outline.",
        "Most topics were taken from the reference book.",
    ],
    ("Neutral", "lab"): [
        "Lab sessions were conducted once a week.",
        "Experiments were done as per the lab manual.",
        "The lab manual listed ten experiments.",
        "Lab classes were scheduled in the afternoon.",
        "Practical sessions were held in two batches.",
        "Lab records had to be submitted every week.",
    ],
    ("Negative", "teaching"): [
        "The lectures are too fast-paced and difficult to follow.",
        "Concepts are not explained properly and doubts are often ignored.",
        "Teaching is boring; the professor just reads from the slides.",
        "Poor explanation of core topics, I felt lost in most classes.",
        "Classes were frequently cancelled, which hurt our exam preparation.",
        "The teacher seems uninterested and rarely answers questions, which is disappointing.",
    ],
    ("Negative", "content"): [
        "The syllabus is outdated and not useful for current industry trends.",
        "Course content is confusing and poorly organised.",
        "Too much theory and almost no practical examples, which is frustrating.",
        "Assignments are excessive and very stressful.",
        "The study material was useless for the exams.",
        "Topics were rushed at the end of the semester, a bad experience.",
    ],
    ("Negative", "lab"): [
        "The lab computers are old, slow and crash frequently.",
        "Lab equipment is broken and never repaired.",
        "Not enough systems in the lab, which is very frustrating.",
        "The lab has poor internet and the software is outdated.",
        "Lab sessions are a waste of time because of bad equipment.",
        "Projectors and lab PCs often fail, making practicals difficult.",
    ],
}

# The same (tone, aspect) pools written the way students type Hinglish …
HINGLISH_REMARK_BANK: dict[tuple[str, str], list[str]] = {
    ("Positive", "teaching"): ["Sir bahut accha padhate hain, sab samajh aata hai.",
                               "Ma'am ka padhane ka tarika bahut badhiya hai.",
                               "Teaching ekdum mast hai, doubts bhi clear karte hain."],
    ("Positive", "content"): ["Syllabus accha hai aur placement ke liye useful hai.",
                              "Notes bahut acche the, exam mein kaafi madad mili.",
                              "Assignments thode mushkil the lekin bahut kuch seekhne ko mila."],
    ("Positive", "lab"): ["Lab ka setup accha hai, sab systems sahi chalte hain.",
                          "Practical karne mein bahut maza aaya.",
                          "Lab assistant bahut helpful hain."],
    ("Neutral", "teaching"): ["Classes time pe hoti hain.",
                              "Sir board pe padhate hain aur slides bhi use karte hain.",
                              "Padhai theek thaak hai."],
    ("Neutral", "content"): ["Syllabus normal hai, baaki subjects jaisa hi.",
                             "Course mein paanch units hain.",
                             "Exam pattern pichle saal jaisa hai."],
    ("Neutral", "lab"): ["Lab hafte mein ek baar hota hai.",
                         "Lab mein Python aur MySQL use karte hain.",
                         "Lab record har hafte submit karna hota hai."],
    ("Negative", "teaching"): ["Sir bahut fast padhate hain, kuch samajh nahi aata.",
                               "Padhane ka tarika accha nahi hai, sirf slides padhte hain.",
                               "Doubts poochne par dhyan nahi dete."],
    ("Negative", "content"): ["Syllabus bahut purana hai, bilkul bekar.",
                              "Assignments bahut zyada hain, bahut pareshani hoti hai.",
                              "Course content bakwas hai."],
    ("Negative", "lab"): ["Lab ke computer bahut kharab hain.",
                          "Lab mein aadhe systems kaam nahi karte.",
                          "Internet nahi chalta aur software purane hain."],
}
# … and in Hindi (Devanagari)
HINDI_REMARK_BANK: dict[tuple[str, str], list[str]] = {
    ("Positive", "teaching"): ["सर बहुत अच्छा पढ़ाते हैं।"],
    ("Positive", "content"): ["सिलेबस अच्छा है और नोट्स बहुत काम के हैं।"],
    ("Positive", "lab"): ["लैब में प्रैक्टिकल करके बहुत मज़ा आया।"],
    ("Neutral", "teaching"): ["क्लास समय पर होती है।"],
    ("Neutral", "content"): ["सिलेबस में पाँच यूनिट हैं।"],
    ("Neutral", "lab"): ["लैब हफ्ते में एक बार होती है।"],
    ("Negative", "teaching"): ["कुछ समझ में नहीं आता।"],
    ("Negative", "content"): ["सिलेबस पुराना और बेकार है।"],
    ("Negative", "lab"): ["लैब के कंप्यूटर बहुत खराब हैं।"],
}

# Five CSE courses; each has a reception bias (-1 poor ... +1 excellent) so courses differ
COURSE_BIAS = {"CS301": 0.5, "CS302": -0.3, "CS303": 0.1, "CS304": 0.3, "CS305": -0.6}
ASPECT_TO_COLUMN = {"teaching": "Teaching_Score", "content": "Course_Content_Score", "lab": "Lab_Facility_Score"}
BASE_SCORE = {"Positive": 4.2, "Neutral": 3.1, "Negative": 2.0}    # mean Likert score per tone
ASPECT_PULL = {"Positive": 0.5, "Neutral": 0.0, "Negative": -0.6}  # extra shift for the aspect discussed
ASPECT_OFFSET = {"teaching": 0.1, "content": 0.0, "lab": -0.3}     # labs rated a little lower overall


def generate_synthetic_data(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic feedback table with one row per (student, course) response.

    The tone of each remark drives its three Likert scores, so text and numbers agree.
    About 15% of remarks are in Hinglish and 3% in Hindi. About 3% of scores and 5% of
    remarks are blanked out (or written as "NA"/"Nil") so the cleaning step has gaps to fix.
    """
    rng = np.random.default_rng(seed)
    courses = list(COURSE_BIAS)
    n_students = -(-n_rows // len(courses))  # ceiling division: 400 rows -> 80 students x 5 courses
    rows = []
    for student in range(1, n_students + 1):
        for course in courses:
            bias = COURSE_BIAS[course]
            p_pos, p_neg = 0.45 + 0.25 * bias, 0.30 - 0.25 * bias
            tone = str(rng.choice(SENTIMENT_ORDER, p=[p_pos, 1 - p_pos - p_neg, p_neg]))
            aspect = str(rng.choice(list(ASPECT_TO_COLUMN)))
            draw = rng.random()
            bank = HINDI_REMARK_BANK if draw < 0.03 else HINGLISH_REMARK_BANK if draw < 0.18 else REMARK_BANK
            remark = str(rng.choice(bank[(tone, aspect)]))
            pulls = {aspect: ASPECT_PULL[tone]}

            # ~25% of students comment on a second aspect -> mixed, more realistic remarks
            if rng.random() < 0.25:
                aspect2 = str(rng.choice([a for a in ASPECT_TO_COLUMN if a != aspect]))
                tone2 = str(rng.choice(SENTIMENT_ORDER))
                remark = f"{remark} {rng.choice(bank[(tone2, aspect2)])}"
                pulls[aspect2] = ASPECT_PULL[tone2]

            row = {"Student_ID": f"23CSE{student:03d}", "Course_Code": course}
            for asp, col in ASPECT_TO_COLUMN.items():
                mean = BASE_SCORE[tone] + ASPECT_OFFSET[asp] + pulls.get(asp, 0.0)
                row[col] = int(np.clip(round(rng.normal(mean, 0.7)), 1, 5))
            row["Remarks"] = remark
            rows.append(row)

    df = pd.DataFrame(rows).head(n_rows)

    # Inject realistic gaps for the cleaning step to handle
    for col in SCORE_COLUMNS:
        df[col] = df[col].astype("Int64")  # nullable integers, so blanks stay blank in the CSV
        df.loc[rng.random(len(df)) < 0.03, col] = pd.NA
    gaps = rng.random(len(df)) < 0.05
    fillers = np.array(["", "NA", "Nil", "No comments"], dtype=object)
    df.loc[gaps, "Remarks"] = fillers[rng.integers(0, len(fillers), gaps.sum())]
    return df


# =============================================================================
# PHASE 1B — LOADING, CLEANING & TEXT PREPROCESSING
# =============================================================================
def load_feedback(source) -> pd.DataFrame:
    """Read a feedback CSV from a path or file-like object.

    Every column is read as text, so roll numbers like "0012" keep their leading zeros;
    scores are converted to numbers in clean_feedback(). Handles Excel's UTF-8 BOM and Latin-1.
    """
    try:
        return pd.read_csv(source, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, dtype=str, encoding="latin-1")


@st.cache_resource(show_spinner="Preparing language resources (first run only)…")
def get_stopwords() -> frozenset[str]:
    """English stop-words from NLTK (downloaded once; scikit-learn's list if offline) plus Hindi ones."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        words = set(nltk_stopwords.words("english"))
    except LookupError:  # no internet, or a proxy blocked the download
        words = set(ENGLISH_STOP_WORDS)
    return frozenset((words - KEEP_WORDS) | DOMAIN_STOPWORDS | HINDI_STOPWORDS)


def preprocess_text(text: str, stop_words: frozenset[str]) -> str:
    """Lower-case a remark, expand negative contractions, keep letters only, drop stop-words.

    Devanagari letters are kept, so Hindi remarks contribute to the themes as well.
    Example: "Lab PCs don't work!" -> "lab pcs not work"
    """
    text = _nfc(text.lower().replace("’", "'"))
    text = re.sub(r"\b(can't|cannot)\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)                            # don't -> do not, isn't -> is not
    text = re.sub(r"[^a-zऀ-ॣ०-ॿ\s]", " ", text)  # strip punctuation, digits, emojis
    return " ".join(word for word in text.split() if word not in stop_words and len(word) > 1)


def clean_feedback(raw: pd.DataFrame, stop_words: frozenset[str]) -> tuple[pd.DataFrame, dict]:
    """Validate columns, fix missing/invalid values and add a preprocessed `Clean_Remarks` column.

    Returns the cleaned DataFrame and a report of what was changed.
    """
    df = raw.rename(columns=lambda c: str(c).strip())
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required column(s): {', '.join(missing)}")
    df = df[REQUIRED_COLUMNS].copy()
    report = {"rows_read": len(df)}

    # 1. Identifiers: tidy text; keep only the latest response per student per course
    for col in ("Student_ID", "Course_Code"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["Course_Code"] = df["Course_Code"].str.upper().fillna("UNKNOWN")
    repeat = df["Student_ID"].notna() & df.duplicated(["Student_ID", "Course_Code"], keep="last")
    df = df[~repeat].copy()
    report["duplicates_removed"] = int(repeat.sum())

    # 2. Likert scores: text or values outside 1–5 are invalid; all gaps get the column median
    report["invalid_scores"] = report["scores_imputed"] = 0
    for col in SCORE_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = (df[col].notna() & values.isna()) | (values.notna() & ~values.between(1, 5))
        values = values.mask(invalid)
        report["invalid_scores"] += int(invalid.sum())
        report["scores_imputed"] += int(values.isna().sum())
        median = values.median() if values.notna().any() else 3.0  # 3 = scale midpoint
        df[col] = values.fillna(median).astype(float)

    # 3. Remarks: blanks and placeholders ("NA", "Nil", "No comments") become empty strings
    remarks = df["Remarks"].fillna("").astype(str).str.strip()
    is_blank = remarks.str.lower().str.strip(" .!-").isin(PLACEHOLDER_REMARKS)
    df["Remarks"] = remarks.mask(is_blank, "")
    report["blank_remarks"] = int(is_blank.sum())

    # 4. Text preprocessing — feeds the keyword/theme extraction
    df["Clean_Remarks"] = df["Remarks"].map(lambda text: preprocess_text(text, stop_words))
    report["rows_analysed"] = len(df)
    return df.reset_index(drop=True), report


# =============================================================================
# PHASE 2 — GRADING & SENTIMENT ANALYSIS
# =============================================================================
def score_to_grade(score: float) -> str:
    """Map a 1–5 weighted score to a letter grade: A >= 4, B >= 3, C >= 2, otherwise D."""
    for cutoff, grade in GRADE_BANDS:
        if score >= cutoff:
            return grade
    return GRADE_BANDS[-1][1]


def compute_overall_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Add `Weighted_Score` (1–5) and `Overall_Grade` (A–D) using GRADE_WEIGHTS."""
    weights = pd.Series(GRADE_WEIGHTS)
    df = df.copy()
    df["Weighted_Score"] = (df[list(GRADE_WEIGHTS)].mul(weights).sum(axis=1) / weights.sum()).round(2)
    df["Overall_Grade"] = df["Weighted_Score"].map(score_to_grade)
    return df


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """VADER (its lexicon ships with the pip package) plus engineering-feedback words it lacks."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(DOMAIN_LEXICON)
    return analyzer


def detect_language(text: str) -> str:
    """'Hindi' if a remark uses Devanagari, 'Hinglish' if it has enough romanised Hindi words, else 'English'."""
    if not text.strip():
        return ""
    if DEVANAGARI.search(text):
        return "Hindi"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(word in HINGLISH_MARKERS or (word not in AMBIGUOUS_HINDI and _hindi_to_english(word) is not None)
               for word in words)  # the second test also catches stretched spellings like "achaaa"
    return "Hinglish" if hits >= 2 or (words and hits / len(words) >= 0.3) else "English"


def _hindi_to_english(word: str) -> str | None:
    """English equivalent of a Hindi/Hinglish word, tolerating stretched spellings ("achaaa", "bohottt")."""
    word = _nfc(word.lower())
    for candidate in (word, re.sub(r"(.)\1{2,}", r"\1\1", word), re.sub(r"(.)\1+", r"\1", word)):
        if candidate in HINDI_WORDS:
            return HINDI_WORDS[candidate]
    return None


def normalize_for_vader(text: str, lexicon: dict, hinglish: bool) -> str:
    """Rewrite Hindi/Hinglish wording into English that VADER can score; English passes through.

    "lab ke computer bahut kharab hain" -> "lab ke computer very bad hain"
    "padhana accha nahi hai"            -> "padhana not good hai"  (Hindi puts 'nahi' after the word)
    """
    text = _nfc(text)
    for pattern, replacement in PHRASE_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    tokens, negations = text.split(), []
    for i, token in enumerate(tokens):
        core = token.strip(PUNCTUATION)
        if not core or (not hinglish and core.lower() in AMBIGUOUS_HINDI):
            continue
        english = _hindi_to_english(core)
        if english:
            keep_caps = core.isupper() and len(core) > 1      # "BEKAR" -> "USELESS" keeps the emphasis
            tokens[i] = token.replace(core, english.upper() if keep_caps else english)
            if english == "not":
                negations.append(i)
    # Hindi negation follows the word it negates ("accha nahi"); VADER expects "not good"
    for i in negations:
        for j in (i - 1, i - 2):
            if j >= 0 and tokens[j].strip(PUNCTUATION).lower() in lexicon:
                tokens.insert(j, tokens.pop(i))
                break
    return " ".join(tokens)


def analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer,
                      language: str | None = None) -> tuple[str, float]:
    """Classify one remark and return (label, polarity score).

    The polarity score is VADER's compound score in [-1, +1]. VADER reads the ORIGINAL
    remark, not the cleaned text, because it uses punctuation, capitals and words such
    as "not" and "very" that stop-word removal would delete ("NOT helpful!!" != "helpful").
    Hinglish and Hindi wording is first mapped to English equivalents (normalize_for_vader).
    """
    if not text.strip():
        return NO_REMARK, float("nan")
    language = language or detect_language(text)
    polarity = analyzer.polarity_scores(normalize_for_vader(text, analyzer.lexicon, language != "English"))["compound"]
    if polarity >= POSITIVE_CUTOFF:
        return "Positive", polarity
    if polarity <= NEGATIVE_CUTOFF:
        return "Negative", polarity
    return "Neutral", polarity


def add_sentiment(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """Add `Language`, `Sentiment`, `Polarity_Score` and `Engine` columns for every remark."""
    df = df.copy()
    df["Language"] = df["Remarks"].map(detect_language)
    results = [analyze_sentiment(text, analyzer, lang) for text, lang in zip(df["Remarks"], df["Language"])]
    df["Sentiment"] = [label for label, _ in results]
    df["Polarity_Score"] = [score for _, score in results]
    df["Engine"] = np.where(df["Sentiment"] == NO_REMARK, "", "VADER")
    return df


# ---- Optional Gemini engine ---------------------------------------------------
GEMINI_PROMPT = (
    "You label student feedback about engineering courses at an Indian college. Remarks may be in "
    "English, Hindi (Devanagari) or Hinglish (Hindi typed in Roman script, often mixed with English).\n"
    "For every remark give its overall sentiment towards the teaching, course or facilities: label is "
    "Positive, Neutral or Negative, and polarity is a number from -1 (very negative) to 1 (very positive). "
    "Treat sarcasm and complaints that something does not work or does not happen as Negative. "
    "Keep each remark's id.\n\nRemarks:\n"
)
GEMINI_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"},
                       "label": {"type": "string", "enum": SENTIMENT_ORDER},
                       "polarity": {"type": "number"}},
        "required": ["id", "label", "polarity"],
    },
}


def get_gemini_key() -> str:
    """Gemini API key from .streamlit/secrets.toml or the GOOGLE_API_KEY / GEMINI_API_KEY variables."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # no secrets.toml
        key = ""
    return key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


@st.cache_data(show_spinner=False)
def gemini_label_batch(remarks: tuple[str, ...], model: str, _api_key: str) -> dict[int, tuple[str, float]]:
    """Label one batch of remarks with Gemini. Returns {position in batch: (label, polarity)}.

    Only the remark text is sent — never student IDs. Results are cached, so redrawing the
    dashboard or changing the course filter does not call the API again.
    """
    from google import genai  # imported here so the offline app runs without the package
    from google.genai import types

    client = genai.Client(api_key=_api_key, http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=4)))  # waits and retries on rate limits (HTTP 429)
    payload = json.dumps([{"id": i, "text": text} for i, text in enumerate(remarks)], ensure_ascii=False)
    response = client.models.generate_content(
        model=model,
        contents=GEMINI_PROMPT + payload,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=GEMINI_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    labels = {}
    for item in json.loads(response.text or "[]"):
        idx, label = item.get("id"), str(item.get("label", "")).capitalize()
        if isinstance(idx, int) and 0 <= idx < len(remarks) and label in SENTIMENT_ORDER:
            try:
                polarity = float(np.clip(float(item.get("polarity", 0.0)), -1, 1))
            except (TypeError, ValueError):
                polarity = 0.0
            labels[idx] = (label, round(polarity, 3))
    return labels


def apply_gemini(df: pd.DataFrame, scope: str, model: str, api_key: str) -> tuple[pd.DataFrame, int, str]:
    """Re-label remarks with Gemini in batches; VADER's label stays wherever Gemini fails.

    Returns (updated DataFrame, number of remarks left on VADER, last error message).
    """
    mask = df["Sentiment"] != NO_REMARK
    if scope == ENGINE_GEMINI_HINDI:
        mask &= df["Language"] != "English"
    targets = df.index[mask]
    if len(targets) == 0:
        return df, 0, ""
    df, done, error = df.copy(), 0, ""
    progress = st.progress(0.0, text=f"Labelling {len(targets)} remarks with Gemini…")
    for start in range(0, len(targets), GEMINI_BATCH_SIZE):
        batch = targets[start:start + GEMINI_BATCH_SIZE]
        try:
            labels = gemini_label_batch(tuple(df.loc[batch, "Remarks"]), model, api_key)
        except Exception as exc:  # invalid key or model name, quota used up, no internet …
            error = str(exc)[:300]
            break  # don't keep hammering the API; remaining remarks keep their VADER labels
        for pos, row in enumerate(batch):
            if pos in labels:
                df.loc[row, ["Sentiment", "Polarity_Score", "Engine"]] = [*labels[pos], "Gemini"]
                done += 1
        progress.progress((start + len(batch)) / len(targets))
    progress.empty()
    return df, len(targets) - done, error


def top_terms(texts: pd.Series, exclude: frozenset[str] = frozenset(), n: int = 10) -> pd.DataFrame:
    """Most frequent words across preprocessed remarks (bag-of-words via scikit-learn).

    Passing opinion words as `exclude` (VADER's lexicon + the Hindi word list) drops words
    like "good", "poor", "bekar", so what remains is WHAT students talk about: lab, syllabus …
    """
    empty = pd.DataFrame(columns=["Term", "Mentions"])
    docs = texts.map(lambda text: " ".join(w for w in text.split() if w not in exclude))
    docs = docs[docs.str.len() > 0]
    if docs.empty:
        return empty
    # Text is already cleaned, so split on spaces (the default regex would break Hindi words apart)
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, binary=True)
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:  # nothing left after stop-word removal
        return empty
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = pd.DataFrame({"Term": vectorizer.get_feature_names_out(), "Mentions": counts})
    return terms.nlargest(n, "Mentions")


# =============================================================================
# PHASE 3 — STREAMLIT DASHBOARD
# =============================================================================
@st.cache_data(show_spinner="Analysing feedback…")
def run_pipeline(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Phase 1 cleaning -> Phase 2 grading and offline sentiment. Cached, so filters redraw instantly."""
    df, report = clean_feedback(raw, get_stopwords())
    df = compute_overall_grade(df)
    df = add_sentiment(df, get_sentiment_analyzer())
    return df, report


@st.cache_data(show_spinner=False)
def sample_data() -> pd.DataFrame:
    return generate_synthetic_data()  # same seed as the CLI, so identical to student_feedback.csv


@st.cache_data(show_spinner=False)
def read_upload(content: bytes) -> pd.DataFrame:
    return load_feedback(io.BytesIO(content))


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV with a UTF-8 BOM, so Excel shows Hindi text correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


def grade_chart(df: pd.DataFrame) -> alt.LayerChart:
    """Bar chart of how many responses received each grade."""
    counts = (df["Overall_Grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
              .rename_axis("Grade").reset_index(name="Responses"))
    base = alt.Chart(counts).encode(
        x=alt.X("Grade:N", sort=GRADE_ORDER, axis=alt.Axis(labelAngle=0, title="Overall grade")),
        y=alt.Y("Responses:Q", title="Responses"),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color("Grade:N", scale=alt.Scale(domain=GRADE_ORDER, range=GRADE_COLORS), legend=None),
        tooltip=["Grade", "Responses"],
    )
    # Counts drawn inside the bars in dark text: readable in both light and dark themes
    labels = base.transform_filter("datum.Responses > 0").mark_text(
        baseline="top", dy=6, fontSize=13, fontWeight="bold", color="#1F2328").encode(text="Responses:Q")
    return (bars + labels).properties(height=300)


def sentiment_chart(df: pd.DataFrame) -> alt.LayerChart:
    """Pie chart of Positive / Neutral / Negative remarks with percentage labels."""
    counts = (df["Sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
              .rename_axis("Sentiment").reset_index(name="Remarks"))
    counts = counts[counts["Remarks"] > 0].assign(
        Share=lambda d: d["Remarks"] / d["Remarks"].sum(),
        Rank=lambda d: d["Sentiment"].map(SENTIMENT_ORDER.index),  # fixes slice order
    )
    base = alt.Chart(counts).encode(
        theta=alt.Theta("Remarks:Q", stack=True),
        order=alt.Order("Rank:Q"),
        color=alt.Color("Sentiment:N", scale=alt.Scale(domain=SENTIMENT_ORDER, range=SENTIMENT_COLORS),
                        legend=alt.Legend(title=None, orient="right")),
        tooltip=["Sentiment", "Remarks", alt.Tooltip("Share:Q", format=".1%")],
    )
    pie = base.mark_arc(outerRadius=120)
    labels = base.transform_filter("datum.Share >= 0.04").mark_text(radius=78, fontSize=14, fontWeight="bold").encode(
        text=alt.Text("Share:Q", format=".0%"), color=alt.value("#1F2328"))  # dark text on each slice
    return (pie + labels).properties(height=300)


def terms_chart(terms: pd.DataFrame, color: str) -> alt.Chart:
    """Horizontal bar chart of the most frequent topic words."""
    return alt.Chart(terms).mark_bar(color=color, cornerRadiusEnd=3).encode(
        x=alt.X("Mentions:Q", title="Remarks mentioning the term"),
        y=alt.Y("Term:N", sort="-x", title=None),
        tooltip=["Term", "Mentions"],
    ).properties(height=26 * len(terms) + 40)


def run_dashboard() -> None:
    st.set_page_config(page_title="Student Feedback Analysis", page_icon="🎓", layout="wide")
    st.title("🎓 Student Feedback & Sentiment Analysis")
    weights = " · ".join(f"{LABELS[col]} {w:.0%}" for col, w in GRADE_WEIGHTS.items())
    st.caption(f"Weighted grading ({weights}) and sentiment analysis of English, Hinglish and Hindi remarks.")

    # ---- Sidebar: data and sentiment engine ----------------------------------
    with st.sidebar:
        st.header("📂 Data")
        source = st.radio("Data source", ["Upload CSV", "Sample data (400 rows)"],
                          label_visibility="collapsed")
        raw = None
        if source == "Upload CSV":
            upload = st.file_uploader("Feedback CSV", type="csv")
            if upload is not None:
                try:
                    raw = read_upload(upload.getvalue())
                except Exception as exc:  # malformed or empty file
                    st.error(f"Could not read this CSV: {exc}")
        else:
            raw = sample_data()
        with st.expander("Expected CSV format"):
            st.markdown("One row per response:\n\n`Student_ID`, `Course_Code`, `Teaching_Score`, "
                        "`Course_Content_Score`, `Lab_Facility_Score` (1–5), `Remarks`")
            st.download_button("Download sample CSV", to_csv_bytes(sample_data()),
                               file_name="student_feedback.csv", mime="text/csv")

        st.header("🧠 Sentiment engine")
        engine = st.radio("Sentiment engine", [ENGINE_OFFLINE, ENGINE_GEMINI_HINDI, ENGINE_GEMINI_ALL],
                          label_visibility="collapsed",
                          captions=["Free, works without internet",
                                    "Needs a Gemini API key; English stays on VADER",
                                    "Needs a key; also catches subtle English complaints"])
        api_key, model = "", GEMINI_MODEL
        if engine != ENGINE_OFFLINE:
            api_key = get_gemini_key() or st.text_input(
                "Gemini API key", type="password", help="Create one free at https://aistudio.google.com/apikey")
            model = st.text_input("Gemini model", GEMINI_MODEL).strip() or GEMINI_MODEL
            st.caption("Only remark text is sent to Google, never student IDs. With a free-tier key Google "
                       "may use the text to improve its products, so use a paid key or the offline engine "
                       "for confidential feedback.")

    if raw is None:
        st.info("⬅️ Upload a feedback CSV in the sidebar, or choose **Sample data** to try the dashboard.")
        st.stop()

    try:
        df, report = run_pipeline(raw)
    except ValueError as exc:  # e.g. required columns missing
        st.error(str(exc))
        st.stop()

    if engine != ENGINE_OFFLINE:
        if not api_key:
            st.info("Add a Gemini API key in the sidebar to use Gemini. Showing offline results for now.")
        else:
            df, failed, error = apply_gemini(df, engine, model, api_key)
            if failed:
                st.warning(f"Gemini could not label {failed} remark(s), so they keep their offline (VADER) "
                           f"labels. {error}")

    with st.sidebar:
        st.header("🔎 Filter")
        course = st.selectbox("Course", ["All courses", *sorted(df["Course_Code"].unique())])
    view = df if course == "All courses" else df[df["Course_Code"] == course]

    st.caption(f"🧹 Cleaning: {report['rows_read']} rows read · {report['duplicates_removed']} duplicate "
               f"submissions removed · {report['scores_imputed']} missing/invalid scores filled with the "
               f"column median · {report['blank_remarks']} blank remarks")
    if report["invalid_scores"]:
        st.warning(f"{report['invalid_scores']} score(s) were not numbers between 1 and 5 and were replaced "
                   "by the column median. Check that the file uses a 1–5 scale.")

    # ---- KPI row -------------------------------------------------------------
    remarks = view[view["Sentiment"] != NO_REMARK]
    counts = remarks["Sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    avg_score = view["Weighted_Score"].mean()

    kpis = st.columns(5)
    kpis[0].metric("Average Grade", f"{score_to_grade(avg_score)} · {avg_score:.2f}/5", border=True,
                   help=f"Mean weighted score ({weights}) mapped to A–D")
    kpis[1].metric("Total Reviews", f"{len(view):,}", border=True,
                   help=f"{len(remarks)} include a written remark")
    for col, label, icon in zip(kpis[2:], SENTIMENT_ORDER, ["😊", "😐", "☹️"]):
        share = counts[label] / len(remarks) if len(remarks) else 0
        col.metric(f"{icon} {label}", f"{share:.0%}", border=True,
                   help=f"{counts[label]} of {len(remarks)} written remarks")
    languages = " · ".join(f"{lang} {n}" for lang, n in remarks["Language"].value_counts().items())
    engines = " · ".join(f"{name} {n}" for name, n in remarks["Engine"].value_counts().items())
    st.caption("Dimension averages (out of 5): " +
               " · ".join(f"{LABELS[c]} **{view[c].mean():.2f}**" for c in SCORE_COLUMNS) +
               f"  \nRemark languages: {languages}  \nLabelled by: {engines}")

    # ---- Charts ---------------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Grade distribution")
        st.altair_chart(grade_chart(view))
    with right:
        st.subheader("Sentiment of remarks")
        if remarks.empty:
            st.info("No written remarks in this selection.")
        else:
            st.altair_chart(sentiment_chart(remarks))

    # ---- Recurring themes (from the preprocessed text) -----------------------
    st.subheader("Recurring themes")
    st.caption("Most mentioned topic words after stop-word removal; opinion words such as "
               "“good”, “poor” or “bekar” are excluded so the charts show *what* students discuss.")
    opinion_words = frozenset(get_sentiment_analyzer().lexicon) | frozenset(HINDI_WORDS) | NEGATIONS
    concern_col, strength_col = st.columns(2)
    for col, label, title, color in [
        (concern_col, "Negative", "Areas of concern", SENTIMENT_COLORS[2]),
        (strength_col, "Positive", "Areas of excellence", SENTIMENT_COLORS[0]),
    ]:
        with col:
            st.markdown(f"**{title}** — {label.lower()} remarks")
            terms = top_terms(view.loc[view["Sentiment"] == label, "Clean_Remarks"], opinion_words)
            if terms.empty:
                st.caption("Not enough remarks.")
            else:
                st.altair_chart(terms_chart(terms, color))

    # ---- Negative remarks for instructors ------------------------------------
    st.subheader("⚠️ Remarks flagged as Negative")
    negative = view[view["Sentiment"] == "Negative"].sort_values("Polarity_Score")
    if negative.empty:
        st.success("No negative remarks in this selection. 🎉")
    else:
        # Student_ID is deliberately not shown, so instructors read the feedback anonymously
        st.caption(f"{len(negative)} remarks, most negative first. Student IDs are hidden.")
        st.dataframe(
            negative[["Course_Code", "Language", "Remarks", "Polarity_Score", *SCORE_COLUMNS, "Overall_Grade"]],
            hide_index=True,
            column_config={
                "Course_Code": st.column_config.TextColumn("Course", width="small"),
                "Language": st.column_config.TextColumn("Language", width="small"),
                "Remarks": st.column_config.TextColumn("Remark", width=480),
                "Polarity_Score": st.column_config.NumberColumn(
                    "Polarity", format="%.2f", width="small",
                    help="Compound score: -1 very negative … +1 very positive"),
                **{c: st.column_config.NumberColumn(LABELS[c], format="%.1f", width="small")
                   for c in SCORE_COLUMNS},
                "Overall_Grade": st.column_config.TextColumn("Grade", width="small"),
            },
        )

    st.download_button("⬇️ Download analysed results (CSV)", to_csv_bytes(view.drop(columns="Clean_Remarks")),
                       file_name="feedback_analysis_results.csv", mime="text/csv")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main_cli() -> None:
    """Phase 1 as a stand-alone script: python feedback_app.py [--rows 400] [--seed 42] [--out FILE]"""
    parser = argparse.ArgumentParser(description="Generate a synthetic student-feedback CSV.")
    parser.add_argument("--rows", type=int, default=400, help="number of responses (default: 400)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, for reproducible data")
    parser.add_argument("--out", default="student_feedback.csv", help="output CSV path")
    args = parser.parse_args()

    df = generate_synthetic_data(args.rows, args.seed)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")  # BOM: Excel then shows Hindi text correctly
    print(f"Saved {len(df)} rows to {args.out}")
    print("Missing scores per column:", df[SCORE_COLUMNS].isna().sum().to_dict())
    print("Next step: streamlit run feedback_app.py")


if __name__ == "__main__":
    if runtime.exists():  # started with `streamlit run feedback_app.py`
        run_dashboard()
    else:                 # started with `python feedback_app.py`
        main_cli()
