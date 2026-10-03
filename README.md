# Student-Feedback-Analysis
This project aims to develop a comprehensive, automated system for evaluating student feedback by combining quantitative grading metrics with qualitative sentiment analysis. 
# Automated Student Feedback Analysis and Sentiment Evaluation System

Turns end-of-semester feedback forms — 1–5 ratings plus free-text remarks in English, Hinglish or Hindi — into weighted grades, sentiment labels, recurring themes and an interactive dashboard.

The whole application is one Python file, `feedback_app.py`. It runs offline by default; Google Gemini is an optional second sentiment engine.

**Built with:** Python · pandas · NumPy · NLTK · vaderSentiment · scikit-learn · Streamlit · Altair · google-genai (optional)

## What it does

| Objective | How the app delivers it |
|---|---|
| Automated grading computation | Weighted score of Teaching (50%), Course Content (30%) and Lab Facilities (20%), mapped to grades A–D |
| Qualitative sentiment analysis | Every written remark labelled Positive, Neutral or Negative with a polarity score from −1 to +1, including Hinglish and Hindi remarks |
| Thematic insight extraction | Most-mentioned topics in negative remarks (areas of concern) and positive remarks (areas of excellence) |
| Interactive reporting | Streamlit dashboard with a course filter, KPI cards, charts, a table of negative remarks and CSV export |

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python feedback_app.py            # writes student_feedback.csv (400 synthetic responses)
streamlit run feedback_app.py     # opens the dashboard at http://localhost:8501
```

In the sidebar, upload `student_feedback.csv` or your own file — or choose **Sample data**, the same 400 rows built in.

- Generator options: `python feedback_app.py --rows 1000 --seed 7 --out demo.csv`
- The first analysis downloads NLTK's English stop-word list; without internet the app falls back to scikit-learn's list.
- Streamlit also prints a *Network URL* that colleagues on the same network can open (allow it through the firewall if prompted).

## Input format

One row per student per course. Headers must match exactly (case-sensitive); extra columns, such as a Google Forms timestamp, are ignored.

| Column | Content |
|---|---|
| `Student_ID` | Roll number, e.g. `23CSE001` (read as text, so leading zeros survive) |
| `Course_Code` | e.g. `CS301` |
| `Teaching_Score` | 1–5 |
| `Course_Content_Score` | 1–5 |
| `Lab_Facility_Score` | 1–5 |
| `Remarks` | Free text in English, Hinglish or Hindi; may be blank |

```csv
Student_ID,Course_Code,Teaching_Score,Course_Content_Score,Lab_Facility_Score,Remarks
23CSE001,CS301,5,4,4,The professor explains every concept clearly with helpful real-world examples.
23CSE002,CS302,3,3,1,Lab ke computer bahut kharab hain.
23CSE003,CS305,2,2,3,सिलेबस पुराना और बेकार है।
```

- **Google Forms:** download the responses as CSV and rename the headers. **Excel:** save as *CSV UTF-8* — plain "CSV" loses Hindi text.
- **Anonymous feedback:** keep the `Student_ID` column but leave it empty; repeat submissions then can't be detected.
- A sample file is also available in the sidebar under **Expected CSV format → Download sample CSV**.

## Dashboard

| Section | Shows |
|---|---|
| Sidebar | Data source (upload or sample), sentiment engine, course filter |
| Cleaning summary | Rows read, duplicates removed, scores filled, blank remarks |
| KPI cards | Average grade, total reviews, share of Positive / Neutral / Negative remarks; below them, each rating's average, remark languages and the engine used |
| Grade distribution | Bar chart of A–D |
| Sentiment of remarks | Pie chart of written remarks |
| Recurring themes | Top topic words in negative remarks (areas of concern) and positive remarks (areas of excellence) |
| Remarks flagged as Negative | Most negative first, with ratings and grade; Student IDs hidden |
| Download | `feedback_analysis_results.csv` for the current selection |

## How it works

`CSV → clean → grade → detect language → sentiment → themes → dashboard`

### 1. Cleaning and preprocessing

| Problem in the data | What the app does |
|---|---|
| Student submitted twice for the same course | Keeps only the last submission in the file |
| Score missing, non-numeric or outside 1–5 | Fills it with that column's median across the file (3 if the column is empty); invalid values also raise a warning |
| Remark blank or a placeholder (`NA`, `Nil`, `None`, `No comments` …) | Labels it **No Remark** and leaves it out of the sentiment percentages |
| Course code in lower case or blank | Upper-cases it; blanks are grouped as `UNKNOWN` |
| File saved by Excel (UTF-8 BOM or Latin-1) | Reads it automatically |

For theme extraction, each remark is also lower-cased, negative contractions are expanded (*don't* → *do not*), punctuation and digits are removed, and stop-words are dropped: NLTK's English list, Hindi function words (*hai, ka, ke* …) and generic feedback words (*course, sir, semester* …).

### 2. Grading

```text
Weighted score = 0.5 × Teaching + 0.3 × Content + 0.2 × Lab     (1–5, rounded to 2 decimals)
```

| Weighted score | Grade |
|---|---|
| 4.00 – 5.00 | A |
| 3.00 – 3.99 | B |
| 2.00 – 2.99 | C |
| 1.00 – 1.99 | D |

Example: Teaching 4, Content 3, Lab 2 → 2.0 + 0.9 + 0.4 = **3.30 → B**. The *Average Grade* card is the mean weighted score of the selected responses, graded the same way.

### 3. Sentiment analysis

**Language detection.** A remark containing Devanagari is *Hindi*; one with at least two romanised Hindi words (or 30% of its words) is *Hinglish*; anything else is *English*.

**Offline engine (default).** VADER is a rule-based analyser that scores words while accounting for intensifiers, negation, capitals and punctuation. Before scoring, the app adds a Hindi/Hinglish layer:

- Hindi and Hinglish words are mapped to English — *kharab* → bad, *bahut* → very, *bekar* → useless — including stretched spellings such as *achaaa*.
- Common idioms are rewritten whole — *samajh nahi aata* → confusing, *kaam nahi karte* → broken — and Hindi's trailing negation is moved: *accha nahi* → *not good*.
- Engineering-feedback words missing from VADER are added (*outdated, rushed, overcrowded, knowledgeable* …).

VADER reads the original remark, not the cleaned text, so *NOT helpful!!* still scores differently from *helpful*. Its compound score (−1 … +1) sets the label: **≥ 0.05 Positive**, **≤ −0.05 Negative**, otherwise **Neutral**.

**Gemini engines (optional).**

| Sidebar choice | Labelled by Gemini |
|---|---|
| Offline: VADER + Hinglish lexicon | Nothing |
| Gemini for Hinglish & Hindi | Hinglish and Hindi remarks; English stays on VADER |
| Gemini for all remarks | Every remark; also catches subtle English complaints |

Remarks go in batches of 50 at temperature 0 and come back as structured JSON. Results are cached, so changing the course filter doesn't call the API again. If a request fails (wrong key, quota used up, no internet), the remaining remarks keep their VADER labels and a warning appears. The `Engine` column records which engine labelled each remark.

### 4. Recurring themes

scikit-learn's `CountVectorizer` counts how many negative (or positive) remarks mention each cleaned word, and the top 10 are charted. Opinion words — VADER's lexicon, the Hindi word list and negations — are excluded, so the charts show *what* students talk about (*lab, syllabus, assignments*) rather than *good* or *bekar*.

## Output

**Download analysed results (CSV)** saves the current selection as `feedback_analysis_results.csv`: the six input columns after cleaning, plus

| Column | Meaning |
|---|---|
| `Weighted_Score` | Weighted average, 1–5 |
| `Overall_Grade` | A–D |
| `Language` | English, Hinglish or Hindi (blank when there is no remark) |
| `Sentiment` | Positive, Neutral, Negative or No Remark |
| `Polarity_Score` | −1 … +1: VADER's compound score, or Gemini's estimate |
| `Engine` | VADER or Gemini |

All CSVs the app writes include a UTF-8 BOM, so Excel shows Hindi text correctly.

## Gemini setup (optional)

1. Create an API key at <https://aistudio.google.com/apikey>.
2. Provide it in one of these ways (checked in this order):
   - `.streamlit/secrets.toml` in the project folder containing `GEMINI_API_KEY = "your-key"` (add the file to `.gitignore`)
   - environment variable `GOOGLE_API_KEY` or `GEMINI_API_KEY`, set in the same terminal before `streamlit run` — PowerShell: `$env:GEMINI_API_KEY = "your-key"`; Linux/macOS: `export GEMINI_API_KEY="your-key"`
   - the password field that appears in the sidebar when no key is found
3. Choose a Gemini engine in the sidebar. The default model is `gemini-3.5-flash-lite`; if Google renames or retires it, type a current model name in the sidebar.

## Privacy

- The negative-remarks table hides Student IDs, so instructors read feedback anonymously.
- Gemini receives only remark text, never Student IDs — but a name typed inside a remark is sent as written.
- With a free-tier Gemini key, Google may use submitted text to improve its products. For confidential feedback, use a paid key or the offline engine.
- The downloaded results CSV **does** contain `Student_ID`; delete that column before circulating it.

## Configuration

Settings are constants in the *CONFIGURATION* and *LANGUAGE RESOURCES* sections at the top of `feedback_app.py`.

| Setting | Default | Controls |
|---|---|---|
| `GRADE_WEIGHTS` | 0.50 / 0.30 / 0.20 | Weight of each rating (normalised, so they need not sum to 1) |
| `GRADE_BANDS` | A ≥ 4, B ≥ 3, C ≥ 2, else D | Grade cut-offs |
| `POSITIVE_CUTOFF`, `NEGATIVE_CUTOFF` | 0.05, −0.05 | Sentiment thresholds |
| `DOMAIN_LEXICON` | *outdated, rushed, knowledgeable* … | Extra VADER words, scored −4 … +4 |
| `DOMAIN_STOPWORDS` | *course, sir, semester* … | Words kept out of the theme charts |
| `HINDI_TO_ENGLISH`, `PHRASE_RULES` | — | Hinglish/Hindi vocabulary and idioms |
| `GEMINI_MODEL`, `GEMINI_BATCH_SIZE` | `gemini-3.5-flash-lite`, 50 | Default Gemini model; remarks per request |

To add a rating such as `Library_Score`, add it to `SCORE_COLUMNS`, `LABELS` and `GRADE_WEIGHTS`. The sample-data generator would also need the new aspect in `ASPECT_TO_COLUMN`, `ASPECT_OFFSET` and the three remark banks.

## Sample dataset

`python feedback_app.py` and the **Sample data** option generate 80 students (`23CSE001`–`23CSE080`) × 5 courses (`CS301`–`CS305`) = 400 responses:

- Courses are rated differently on purpose, from well liked (CS301) to poorly rated (CS305), so comparisons are meaningful.
- Each remark's tone drives its ratings — a complaint about lab PCs lowers that response's lab score — so text and numbers agree.
- About 82% of remarks are English, 15% Hinglish and 3% Hindi; a quarter comment on two aspects.
- Gaps are injected for the cleaning step: about 3% of scores per column are blank, and 5% of remarks are blank or `NA` / `Nil` / `No comments`.

## Project structure

```text
feedback_app.py          the whole application (Phases 1–3)
requirements.txt         dependencies; google-genai is used only by the Gemini engine
student_feedback.csv     created by: python feedback_app.py
.streamlit/secrets.toml  optional: Gemini API key
```

| Phase | Main functions |
|---|---|
| 1 · Data and preprocessing | `generate_synthetic_data`, `load_feedback`, `clean_feedback`, `preprocess_text` |
| 2 · Grading and sentiment | `compute_overall_grade`, `detect_language`, `normalize_for_vader`, `analyze_sentiment`, `apply_gemini`, `top_terms` |
| 3 · Dashboard | `run_dashboard` |

## Troubleshooting

| Problem | Fix |
|---|---|
| *The CSV is missing required column(s)* | Rename the headers to the six names above, with the same capitalisation |
| Warning that scores are not between 1 and 5 | The file uses another scale (e.g. 1–10) or text ratings; convert to 1–5 before uploading |
| Hindi remarks show as `????` | Export again from the original sheet as *CSV UTF-8* |
| `streamlit: command not found` | Activate the virtual environment, or run `python -m streamlit run feedback_app.py` |
| Gemini: model not found | Enter a current model name from [Google's model list](https://ai.google.dev/gemini-api/docs/models) in the sidebar |
| Gemini: quota or rate-limit error | The app retries, then keeps VADER labels; wait and rerun, use a paid key, or switch to the offline engine |

## Limitations

- The offline engine relies on word lists: it covers common Hinglish words and idioms, but unusual spellings, sarcasm and mixed opinions can be misread — the Gemini engines suit these better.
- Each remark gets one overall label, so *"teaching is great but the lab is bad"* is not split by aspect.
- Themes are single words: *fast-paced* appears as *fast* and *paced*.
- Weights, grade bands and thresholds are starting points; check them against your department's real feedback.
