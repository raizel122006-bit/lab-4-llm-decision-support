# %% [markdown]
# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** Gorpu Amber Raizel Keller
# **Student ID:** 73972028
# 
# ---
# 
# ### Objective
# 
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# %% [markdown]
# ---
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# %%
# API-key setup — DO NOT hard-code your key in this cell.

import os
from dotenv import load_dotenv

# Load variables from the local .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.environ["GROQ_API_KEY"]

# OpenAI-compatible client for Groq
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"

print("Client ready.")

# %% [markdown]
# ---
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# %% [markdown]
# ### Part 1.1 — Your first API call

# %%
# TODO: Write a helper function you will reuse for the WHOLE lab:

def ask_llm(
    user_prompt,
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=500
):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response


# TODO: Call it once with a simple question and print the answer.

response = ask_llm("What is the color of the sky?")

print("Response:")
print(response.choices[0].message.content)


# TODO: Print response.usage as well — how many tokens did your call consume?

print("\nToken usage:")
print(response.usage)

# %% [markdown]
# **Student Reasoning — Anatomy of a call**
# *1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each.*
# 
# The system role tells the LLM how it should behave or what role it should take, while the user role is the actual question or task I want it to answer. For example, the system could say, “You are a helpful and factual assistant,” while the user could ask, “What is the color of the sky?”
# 
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
# 
# A token is basically a small piece of text that the LLM reads or generates. It can be a whole word, part of a word, or even punctuation. API providers charge per token because longer prompts and responses require more processing, so the number of tokens is a better way to measure how much of the model was used than just charging per request.
# 

# %% [markdown]
# ### Part 1.2 — Temperature: the randomness dial

# %%

# TODO: Ask the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
#   A good test question: "Suggest a name for a savings product for market traders in Accra."

question = "Suggest a name for a pet dog"

answers_temp_0 = []

for i in range(5):
    response = ask_llm(
        question,
        temperature=0.0,
        max_tokens=100
    )
    answers_temp_0.append(response.choices[0].message.content)


answers_temp_1_2 = []

for i in range(5):
    response = ask_llm(
        question,
        temperature=1.2,
        max_tokens=100
    )
    answers_temp_1_2.append(response.choices[0].message.content)

# TODO: Print all 10 answers, grouped by temperature.
print("=== Temperature 0.0 ===")

for i, answer in enumerate(answers_temp_0, start=1):
    print(f"{i}. {answer}")


print("\n=== Temperature 1.2 ===")

for i, answer in enumerate(answers_temp_1_2, start=1):
    print(f"{i}. {answer}")

# %% [markdown]
# **Student Reasoning — Temperature**
# *What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?*
# 
# > **Answer:** At temperature 0.0, the responses were very consistent. The model repeatedly gave “Buddy” as the main suggestion and often gave the same alternative names as well. At temperature 1.2, the responses were more varied and creative. For the loan decision-support system, I would use a low temperature such as 0.0 because the system needs to be consistent, factual, and less unpredictable when handling loan applications.

# %% [markdown]
# ---
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# %%
LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods","monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock","monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports","monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")

# %% [markdown]
# ---
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# %% [markdown]
# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# %%
# TODO: Write SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").

SUMMARY_PROMPT_V1 = "Summarize this loan application:"


# Run it on L002 and L006. Read the output critically.

v1_l002 = ask_llm(
    f"{SUMMARY_PROMPT_V1}\n\n{LETTERS['L002']}",
    temperature=0.7,
    max_tokens=200
)

v1_l006 = ask_llm(
    f"{SUMMARY_PROMPT_V1}\n\n{LETTERS['L006']}",
    temperature=0.7,
    max_tokens=200
)

print(" V1 - L002")
print(v1_l002.choices[0].message.content)

print("\nV1 - L006")
print(v1_l006.choices[0].message.content)


# TODO: Now write SUMMARY_PROMPT_V2 as a proper template with:

# - a system prompt giving the LLM a ROLE (e.g "You are an assistant to a microfinance
#   loan officer...") and constraints (factual, neutral, no invented details, 3-4 sentences)

SUMMARY_SYSTEM_V2 = """You are an assistant to a microfinance loan officer.Summarize loan applications in a factual and neutral way. Don't invent any information that is not stated in the letter.
Keep the summary to 5-6 sentences and include only relevant information from the application."""


# - a user prompt template like: f"Summarize this loan application:\n\n{letter_text}"

SUMMARY_PROMPT_V2 = "Summarize this loan application:\n\n{letter_text}"


# Run V2 on the same two letters at temperature=0.

v2_l002 = ask_llm(
    SUMMARY_PROMPT_V2.format(letter_text=LETTERS["L002"]),
    system_prompt=SUMMARY_SYSTEM_V2,
    temperature=0.0,
    max_tokens=200
)

v2_l006 = ask_llm(
    SUMMARY_PROMPT_V2.format(letter_text=LETTERS["L006"]),
    system_prompt=SUMMARY_SYSTEM_V2,
    temperature=0.0,
    max_tokens=200
)


# TODO: Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook.

print("\n\nCOMPARISON")

print("\n----- L002 -----")
print("\nV1:")
print(v1_l002.choices[0].message.content)

print("\nV2:")
print(v2_l002.choices[0].message.content)

print("\n----- L006 -----")
print("\nV1:")
print(v1_l006.choices[0].message.content)

print("\nV2:")
print(v2_l006.choices[0].message.content)

# %% [markdown]
# **Student Reasoning — Summarization prompts**
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# *2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?*
# 
# > **Answer:** 1. V1 gave a reasonable summary but it included some details that were more like assumptions than what was actually stated. For example, in L006, V1 said Kofi was “relying on his personal guarantee,” even though the letter only says he has no collateral and that he is trustworthy. V2 was more controlled because it was instructed to be factual, neutral, and only use information from the letter.
# 
# 2. Stating no invented details is important because the loan officer needs to make decisions based on accurate information from the applicant. If the LLM makes up or assumes information, it could lead to an incorrect assessment and potentially unfair treatment of an applicant. This failure is called hallucination.
# 

# %% [markdown]
# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# %%
import json
import pandas as pd

# TODO: Write EXTRACT_PROMPT — a template that instructs the model to return ONLY a JSON

# object with EXACTLY these keys:

# applicant_name (string), amount_ghs (number), purpose (string),

# monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),

# repayment_months (number or null)

# Techniques to use:

# - explicit schema in the prompt

# - ONE worked example (few-shot) using a letter you write yourself (not from LETTERS!)

# - "If a field is not stated in the letter, use null. Do not guess."

# - temperature=0


EXTRACT_PROMPT = """
Extract the requested information from the loan application below.

Return ONLY a valid JSON object with EXACTLY these keys:

{
    "applicant_name": "string",
    "amount_ghs": number,
    "purpose": "string",
    "monthly_profit_ghs": number or null,
    "has_collateral_or_guarantor": true or false,
    "repayment_months": number or null
}

Rules:
- Use only information stated in the letter.
- If a field is not stated in the letter, use null.
- Do not guess or infer missing information.
- amount_ghs must be a number, not a string.
- monthly_profit_ghs must be a number or null.
- repayment_months must be a number or null.
- has_collateral_or_guarantor must be true if the applicant states that they have
  collateral or a guarantor, and false if they explicitly state that they do not.
- Return ONLY the JSON object.
- Do not include explanations or markdown.

Worked example:

Example letter:
"My name is John Brown. I operate a small mobile money kiosk in Kasoa.
I am requesting GHS 8,500 to expand the kiosk and add a second mobile money line.
I make around GHS 800 in profit each month. I do not have any collateral or guarantor.
I plan to repay the loan over 12 months."

Example output:
{
    "applicant_name": "John Brown",
    "amount_ghs": 8500,
    "purpose": "expand the kiosk and add a second mobile money line",
    "monthly_profit_ghs": 800,
    "has_collateral_or_guarantor": false,
    "repayment_months": 12
}

Now extract the information from this loan application:

<LETTER_TEXT>
"""


# TODO: Write extract_fields(letter_text) that calls the LLM, strips any ```json fences,

# json.loads() the result, and returns a dict. Handle parse failures gracefully

# (return None and print a warning).


def extract_fields(letter_text):

    prompt = EXTRACT_PROMPT + letter_text

    response = ask_llm(
        prompt,
        temperature=0.0,
        max_tokens=300
    )

    # Get the actual text from the ChatCompletion response
    result = response.choices[0].message.content.strip()

    # Remove markdown JSON fences if the model adds them
    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()

    try:
        return json.loads(result)

    except json.JSONDecodeError:
        print("Warning: Could not return LLM response as JSON.")
        print("Raw response:", result)
        return None


# TODO: Run it on ALL SIX letters; collect results into a pandas DataFrame (one row per

# letter) and display it.


extracted_results = []

for letter_id, letter_text in LETTERS.items():

    extracted = extract_fields(letter_text)

    if extracted is not None:
        extracted["letter_id"] = letter_id
        extracted_results.append(extracted)

    else:
        extracted_results.append({
            "letter_id": letter_id,
            "applicant_name": None,
            "amount_ghs": None,
            "purpose": None,
            "monthly_profit_ghs": None,
            "has_collateral_or_guarantor": None,
            "repayment_months": None
        })


extracted_df = pd.DataFrame(extracted_results)


# Put letter_id first

columns = [
    "letter_id",
    "applicant_name",
    "amount_ghs",
    "purpose",
    "monthly_profit_ghs",
    "has_collateral_or_guarantor",
    "repayment_months"
]

extracted_df = extracted_df[columns]

display(extracted_df)

# %% [markdown]
# **Student Reasoning — Structured extraction**
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
# 
# > **Answer:** 1.The few-shot example should not come from the six letters because it could make the model rely too much on that specific letter instead of actually learning how to extract the information. Using a separate example shows that the prompt works on new, unseen letters.
# 
# 2. "Use null, do not guess" is important because the model might otherwise try to fill in missing information based on what seems likely. For example, some of the letters do not state a monthly profit, so the model should return null instead of making up a number. This helps reduce hallucinations.
# 
# 3. Temperature=0 is appropriate for extraction because we want the model to give the same factual information consistently every time. For creative tasks, a higher temperature can be useful because it allows more variation and different ideas in the responses.

# %% [markdown]
# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# %%
# TODO: Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
#     1. Strengths (bullet points, grounded in the letter)
#     2. Risks / red flags (bullet points)
#     3. Missing information the officer should request
#     4. Suggested next step (e.g. "invite for interview", "request documents",
#        "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.

BRIEF_PROMPT = """
You are an assistant supporting a microfinance loan officer.

Review the loan application and the extracted information below.

Your job is to provide decision-support information, not to make the final loan
decision. The final decision must always be made by a human loan officer.

Keep your assessment factual, neutral, and grounded only in the information
provided. Do not invent information or make assumptions about the applicant.

Your response must contain exactly these four sections:

1. Strengths
- List the positive factors supported by the application.

2. Risks / Red Flags
- List any risks, concerns, or warning signs supported by the application.

3. Missing Information
- List information or documents the loan officer should request before making
  a decision.

4. Suggested Next Step
- Suggest an appropriate action such as inviting the applicant for an interview,
  requesting documents, or flagging the application for senior review.
- Do NOT say "approve" or "reject".
- Do NOT make the final lending decision.

Loan application:
{letter}

Extracted information:
{extracted_data}
"""


# TODO: Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.

briefs = {}

for letter_id, letter_text in LETTERS.items():

    # Get the extracted information for this letter
    extracted_row = extracted_df[
        extracted_df["letter_id"] == letter_id
    ].iloc[0]

    # Convert the DataFrame row to a dictionary
    extracted_data = extracted_row.to_dict()

    # Remove the letter ID because it is not part of the extracted fields
    extracted_data.pop("letter_id", None)

    prompt = BRIEF_PROMPT.format(
        letter=letter_text,
        extracted_data=extracted_data
    )

    response = ask_llm(
        prompt,
        temperature=0.0,
        max_tokens=500
    )

    briefs[letter_id] = response.choices[0].message.content


# Print the requested briefs

for letter_id in ["L001", "L002", "L006"]:

    print("-" * 30)
    print(f"BRIEF — {letter_id}")
    print("-" * 30)
    print(briefs[letter_id])
    print()

# %% [markdown]
# **Student Reasoning — Decision support**
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?*
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason.*
# 
# > **Answer:** 1. Yes, the system identified the main strengths and red flags in both applications. For L003, it pointed out the registered business, existing profit, sales records, and fixed deposit as strengths. For L006, it correctly identified that the businesses have not started yet, there is no collateral, and there is no proven income from the proposed businesses.
# 
# 2. We forbid the model from saying "approve" or "reject" because the final decision should be made by a human. Practically, the model could misunderstand information or miss something important in the application. Ethically, letting an AI make the final decision could unfairly affect applicants if the model makes a biased or incorrect decision.

# %% [markdown]
# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:** [7450286 ]

# %% [markdown]
# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.

# %% [markdown]
# ### Part 4.1 — Extraction accuracy against gold labels

# %%
# TODO: For the three letters in GOLD, compare your extracted DataFrame to the gold values
# field by field. Compute per-field accuracy across the three letters
# (name matching can be case-insensitive; numbers must match exactly).

# TODO: Display a small table: rows = fields, columns = L001 / L003 / L006 / accuracy.

import pandas as pd

fields = [
    "applicant_name",
    "amount_ghs",
    "purpose",
    "monthly_profit_ghs",
    "has_collateral_or_guarantor",
    "repayment_months"
]

gold_ids = ["L001", "L003", "L006"]

comparison_results = []

for field in fields:

    row = {"field": field}
    correct_count = 0

    for letter_id in gold_ids:

        # Get extracted value from the DataFrame
        extracted_value = extracted_df.loc[
            extracted_df["letter_id"] == letter_id, field
        ].iloc[0]

        # Get the correct gold-standard value
        gold_value = GOLD[letter_id][field]

        # Treat None and NaN as equivalent missing values
        if pd.isna(gold_value) and pd.isna(extracted_value):
            is_correct = True

        # Applicant names: compare case-insensitively
        elif field == "applicant_name":
            is_correct = (
                str(extracted_value).strip().lower()
                == str(gold_value).strip().lower()
            )

        # Purpose: compare case-insensitively
        elif field == "purpose":
            is_correct = (
                str(extracted_value).strip().lower()
                == str(gold_value).strip().lower()
            )

        # All other fields must match exactly
        else:
            is_correct = extracted_value == gold_value

        # Record result
        row[letter_id] = "✓" if is_correct else "✗"

        if is_correct:
            correct_count += 1

    # Calculate accuracy across the three gold-standard letters
    row["accuracy"] = correct_count / len(gold_ids)

    comparison_results.append(row)


# Create the comparison table
comparison_df = pd.DataFrame(comparison_results)

# Convert accuracy to percentage
comparison_df["accuracy"] = comparison_df["accuracy"].map(
    lambda x: f"{x:.1%}"
)

# Display the final table
display(comparison_df)

# %% [markdown]
# ### Part 4.2 — Reliability: is the system consistent?

# %%
# TODO: Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
# temperature=1.0.

# TODO: For each temperature, report how many of the 5 runs produced (a) valid JSON and
# (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
# and count unique strings.

import json

letter_text = LETTERS["L004"]

results_by_temperature = {
    0.0: [],
    1.0: []
}

for temperature in [0.0, 1.0]:

    print("=" * 60)
    print(f"Temperature = {temperature}")
    print("=" * 60)

    for run_number in range(5):

        # Add the actual L004 letter to the extraction prompt
        prompt = EXTRACT_PROMPT + "\n" + letter_text

        response = ask_llm(
            prompt,
            temperature=temperature,
            max_tokens=300
        )

        result = response.choices[0].message.content.strip()

        # Remove markdown JSON fences if present
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        try:
            parsed_result = json.loads(result)

            results_by_temperature[temperature].append(parsed_result)

            print(f"Run {run_number + 1}: Valid JSON")
            print(parsed_result)

        except json.JSONDecodeError:

            results_by_temperature[temperature].append(None)

            print(f"Run {run_number + 1}: INVALID JSON")
            print(result)

        print()


# TODO: Report reliability results

print("\n" + "=" * 60)
print("RELIABILITY RESULTS")
print("=" * 60)

for temperature, results in results_by_temperature.items():

    valid_results = [
        result for result in results
        if result is not None
    ]

    valid_count = len(valid_results)

    result_strings = [
        json.dumps(result, sort_keys=True)
        for result in valid_results
    ]

    unique_count = len(set(result_strings))

    # All five runs are identical if there is exactly one unique
    # valid result and all five responses were valid JSON.
    all_identical = (
        valid_count == 5 and unique_count == 1
    )

    print(f"\nTemperature: {temperature}")
    print(f"Valid JSON responses: {valid_count}/5")
    print(f"Unique valid outputs: {unique_count}")
    print(f"Identical across all 5 runs: {'Yes' if all_identical else 'No'}")

# %% [markdown]
# ### Part 4.3 — Hallucination probing

# %%
# TODO: Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?

# TODO: Record the outputs verbatim below and label each PASS or FAIL.



# TEST 1 


test1_prompt = f"""
The following is a loan application:

{LETTERS["L002"]}

What is the applicant's credit score?

Answer only based on the information provided in the letter.
If the information is not provided, say that it is not stated.
"""

test1_response = ask_llm(
    test1_prompt,
    temperature=0.0,
    max_tokens=200
)


test1_text = test1_response.choices[0].message.content.strip()

print("TEST 1 — Missing information")
print("-" * 60)
print(test1_text)
print()

# PASS if the model admits the information is absent
test1_lower = test1_text.lower()

if (
    "not stated" in test1_lower
    or "not provided" in test1_lower
    or "not mentioned" in test1_lower
    or "does not state" in test1_lower
    or "no information" in test1_lower
):
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")



# TEST 2


irrelevant_text = """
Today's weather report says that Accra will experience partly cloudy
conditions with temperatures around 29 degrees Celsius. There may be
some rain in the afternoon. Winds will be moderate.
"""

test2_response = extract_fields(irrelevant_text)

print("\n" + "=" * 60)
print("TEST 2 — Irrelevant input")
print("-" * 60)
print(test2_response)
print()

# PASS if the extractor returns null for all required fields
expected_null_fields = [
    "applicant_name",
    "amount_ghs",
    "purpose",
    "monthly_profit_ghs",
    "has_collateral_or_guarantor",
    "repayment_months"
]

if (
    test2_response is not None
    and all(test2_response.get(field) is None for field in expected_null_fields)
):
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")

# %% [markdown]
# **Student Reasoning — Evaluation results**
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# *2. What did the reliability experiment show about temperature and production systems?*
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?*
# 
# > **Answer:** 1. My extraction accuracy was 100% for applicant name, amount, monthly profit, collateral/guarantor, and repayment period. The purpose field had 0% exact-match accuracy because the model sometimes used different wording from the gold-standard labels even when the meaning was correct. This shows that purpose was the hardest field to evaluate using exact string matching.
# 
# 2. The reliability experiment showed that both temperature 0.0 and 1.0 produced valid and identical results across all five runs for L004. This suggests that the extraction prompt was very consistent for this particular application. For a production system, I would still use temperature 0 because extraction needs predictable and reproducible outputs rather than creative variation.
# 
# 3. Yes, the system showed a small hallucination/failure under probing. When given an irrelevant weather report, the extractor correctly returned null for most fields but returned false for has_collateral_or_guarantor, even though the text said nothing about collateral or a guarantor. To reduce this risk, the prompt should explicitly say that if the text does not mention an applicant or does not provide evidence about a field, the value must be null rather than false. The system could also use validation rules after extraction to flag suspicious outputs and require human review rather than relying entirely on the LLM.

# %% [markdown]
# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# %% [markdown]
# **Student Reasoning — Appropriateness**
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses.*
# *2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?*
# *3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring).*
# 
# > **Answer:** 1. If the bank fully automated loan decisions using this system, applicants who write poorly in English could be unfairly harmed. The LLM might misunderstand their business situation or interpret a poorly written application as a sign that the business is weak, even when the applicant actually has a successful business. This could create a bias against people with weaker English or writing skills. That is why the system should support the loan officer rather than make the final decision.
# 
# 2. Loan letters contain personal and financial information, so sending them to a third-party API in another country could create privacy and data-protection risks. Before using this system at a real Ghanaian microfinance institution, I would check how the API provider stores, processes, and protects the data, whether the data can be used for training, where it is stored, and what Ghanaian data-protection requirements apply. I would also make sure the institution has the necessary consent, contracts, access controls, and security measures in place.
# 
# 3. Two safeguards I would build are human review and logging/monitoring. A loan officer should review the LLM's extracted info, risks, and also suggested next steps before any decision is made. I would also keep secure logs of the model's outputs and monitor them for errors, hallucinations, and differences in performance across different types of applicants. 

# %% [markdown]
# ---
# # Section 5 — Reflection
# 
# *Answer in a few sentences each:*
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
# 
# > **Answer:** 1. Prompting is similar to tuning hyperparameters because I change something, test it, and see if the results improve. The difference is that hyperparameters change how a model works, while prompts change the instructions given to the model.
# 
# 2. I would not trust the system to run completely on its own. The biggest issue was the adversarial test where it returned false for collateral even though the text had no information about it. This shows that human review is still needed.
# 
# 3. Based on my API usage, 1,000 applications would probably use around 1 million tokens if each application used about 1,000 tokens. This means I would compare providers based on cost, reliability, and privacy before choosing one.
# 
# 4. Looking back at the course, using an API is much faster and easier than training my own model because the foundation model is already trained and can understand the applications. I would consider training my own model if I had a lot of suitable data or needed more control over privacy and the model.

# %% [markdown]
# ---
# ### Submission checklist
# 
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.


