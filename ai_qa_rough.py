import os
import re
import json
import uuid
import textwrap
import datetime as dt
from pathlib import Path
from typing import List, Dict

import openai
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ───────────────────────── CONFIG ─────────────────────────
MODEL        = "o4-mini"
TEMP         = 0.0
TOP_K_KB     = 6
OUT_FILE     = "qa_report.jsonl"
CHROMA_DIR   = "chroma_kb"
GLOB_PATH    = "acc_sec_md/**/*.md"
EMBED_MODEL  = "text-embedding-3-small"

openai.api_key = os.getenv("OPENAI_API_KEY") or "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"

if not openai.api_key:
    raise EnvironmentError("Set OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# ─────────────────────── CHAT TRANSCRIPT ───────────────────────
CHAT_TRANSCRIPT = textwrap.dedent("""
CUST 08:50 Hey, I just saw an alert about unpaid invoices—where do I check them?
AGT  08:51 Sure—head over to Billing → Invoices and you’ll see all your bills.  
CUST 08:53 In “Invoices” I only see paid ones. How do I filter unpaid?  
AGT  08:54 There’s a “Status” dropdown up top—select “Unpaid.”  
CUST 08:56 I don’t see a dropdown, only a date filter.  
AGT  08:57 Hm… on older UI it’s under Billing Summary, not Invoices. Try that.  
CUST 08:59 Right—I found “Billing Summary” and can filter there. Thanks.
CUST 09:02 I want to change the display name my visitors see.  
AGT  09:03 You can edit that in Settings → Profile.  
CUST 09:05 I only have “Account Settings” and “Security”—no “Profile.”  
AGT  09:06 Sorry—go to People → Agents, click your name, then “Display Name.”  
CUST 09:08 Got it, but the new name isn’t showing yet in chat.  
AGT  09:09 It can take up to 10 minutes to propagate—should update soon.
CUST 09:12 I need to disable 2FA just this once—how?  
AGT  09:13 In Security Settings there’s a “Disable 2FA” toggle.  
CUST 09:15 I don’t see that toggle in Security Settings.  
AGT  09:16 You might not have permission—ask your admin to disable it.  
CUST 09:18 Oh—so agents can’t disable 2FA themselves?  
AGT  09:19 Correct, only admins can temporarily lift the 2FA requirement.
CUST 09:22 how do I remove all permissions from an agent using API?  
AGT  09:23 Use DELETE on `/global/agents/{id}/permissions`.  
CUST 09:25 I tried DELETE but got a 405 error.  
AGT  09:26 Ah—you actually need a PUT to that same endpoint with an empty array.  
CUST 09:28 Perfect—that worked.
CUST 09:31 Can I auto-assign chats to me only between 9 am–5 pm?  
AGT  09:32 Yes—go to Routing → Auto-Assignment and set time rules.  
CUST 09:34 I don’t see “Auto-Assignment” under Routing.  
AGT  09:35 Try under Live Chat → Settings → Triggers → Time-Based.  
CUST 09:37 That worked.
CUST 09:40 How can I update my saved credit card?  
AGT  09:41 Billing Profile has an “Edit” link next to Payment Method.  
CUST 09:43 I only see “Remove Card.”  
AGT  09:44 Click “Remove,” then click “Add Payment Method.”  
CUST 09:46 Ok, that worked.
CUST 09:49 Is there a way to retroactively adjust chat timestamps if I flagged the wrong timezone?  
AGT  09:50 No, once a chat is recorded the timestamp is permanent.  
CUST 09:52 Are you sure? I thought there was an audit option.  
AGT  09:53 Audit only tracks edits, not timestamp changes.
CUST 09:56 Last one—where do I find the API docs?  
AGT  09:57 They’re on docs.comm100.com under Developer → API.  
CUST 09:59 I only got a 404 there.  
AGT  10:00 Try https://developer.comm100.com/restful-api-guide instead.  
CUST 10:02 That loads, thanks!
""").strip()

# ╭──────────────── PROMPT BUILDER ────────────────╮
class PromptBuilder:
    system = "You are a Comm100 Customer and Agent Conversational thread analyzer and grouping expert. "
    # ── shared 4.1-optimized SYSTEM message
    SYSTEM_DRAFT = """
# ROLE & OBJECTIVE
You are **Comm100-GPT**, an expert KB assistant.  
Write two answers to the user’s question – one *short* and one *long* –
strictly grounded in the **Passages** provided below.

# INSTRUCTIONS
1. **Passages are the ONLY source of truth.**  
2. Never treat statements from the question as facts unless the same statement also appears in the passages.  
3. Never rely on world knowledge, policy, or guesswork.
4. If the passages do **not** contain the facts needed, output exactly **I cannot answer this question** for both answers.
If you cannot answer, still return this exact object
{
  "short": {"answer":"I cannot answer this question","context":""},
  "long" : {"answer":"I cannot answer this question","context":""}
}
5. Every statement in the answer must be based on evidence from the docs.  
6. Think step by step silently – never reveal chain-of-thought, breaking down each question.
7. If the question starts with “how many” or “number of”, do **all** of the following:
  - list every distinct item you will count, each on its own Evidence line  
  - after the list, write: “TOTAL = <N>” (still inside the Evidence block)  
  - in the Answer, state the integer only once, e.g. “There are **12** … [2]”
8. Never infer a larger or smaller number – always count the quoted lines you
  listed. If no lines can be counted ⇒ **I cannot answer this question**
9. Write valid **JSON only** in the format shown below – nothing else.

# REASONING STEPS  *(follow internally)*
You may think step-by-step *inside* <scratch> … </scratch> tags;
1. Read the question and decide what facts are required to answer the question from the provided passgages as evidence.
3. If no sentence answers the question → output **I cannot answer this question**  
4. Otherwise draft a the two answers one short and the other longer based on the evidence, do NOT show the evidence directly just a concise grounded answer.  
5. Double-check every claim is true to the provides docs/evidences
6. The evidence is the single source of truth to answer the questions do not use any external knowledge.
""".strip()

    # ── 4.1-style SYSTEM for grading
    SYSTEM_GRADE = """
# ROLE & OBJECTIVE
You are Comm100-QA expert, a expert QA **manager**.  
Given a question, the agent’s answer, two AI reference answers, and KB passages,  
decide how well the agent aligns with the KB truth.

# SCORING RUBRIC
Give a score based on 0 to 5, where 0 is the worst and 5 is best, give a score based on single source of knowledge truth which is the passages given to you. 

# GLOBAL RULES
• KB is the single source of truth.  
• Ignore any AI answer content that conflicts with KB.  
• Output valid JSON only (no extra keys).  

# OUTPUT FORMAT
{
  "ai_score"   : 0-5,
  "ai_rational": "...",
  "kb_verify"  : ["KB line 1", "KB line 2", …]
}
""".strip()

    # ── FEW-SHOT bootstrap (three compact examples)
    FEW_SHOT = [
        {
            "role": "user",
            "content": """\
### Passages
[1] Rule-Based Chat Routing is only available with our **Live Chat Enterprise** plan.

### Question
Why don’t I see Rule-Based Chat Routing on the Business plan?

### Output format
{"short":...,"long":...}"""
        },
        {
            "role": "assistant",
            "content": """\
{"short":{"answer":"It’s an Enterprise-only feature.","context":"[1]"},
 "long":{"answer":"Rule-Based Chat Routing appears only on the Live Chat Enterprise plan, so it’s hidden for Business customers.","context":"[1]"}}"""
        },

        # count example
        {
            "role": "user",
            "content": """\
### Passages
[1] Visitor Status codes range 0–11.

### Question
How many visitor status codes exist?

### Output format
{"short":...,"long":...}"""
        },
        {
            "role": "assistant",
            "content": """\
{"short":{"answer":"There are 12 statuses.","context":"[1]"},
 "long":{"answer":"There are **12** visitor status codes (0 through 11).","context":"[1]"}}"""
        },
    ]

    # ── FEW-SHOT EXAMPLES (match, partial, out-of-scope)
    FEW_GRADE = [
        # 5  ───────────────────────────────────────────────────────────
        {
            "role":"user",
            "content":json.dumps({
                "question":"How do I filter unpaid invoices?",
                "agent":"Use the Status dropdown and select Unpaid.",
                "ai_short":"Select Unpaid in Status filter.",
                "ai_long":"On Invoices page click Status → Unpaid.",
                "kb_evidence":[
                    "On the Invoices page, use the Status dropdown to choose Unpaid. (source: invoices.md)"
                ]
            }, indent=2)
        },
        {
            "role":"assistant",
            "content":json.dumps({
                "ai_score":5,
                "ai_rational":"Agent matches the KB exactly—same instruction as AI short answer.",
                "kb_verify":[
                    "On the Invoices page, use the Status dropdown to choose Unpaid. (source: invoices.md)"
                ]
            })
        },
        # 2.5 ──────────────────────────────────────────────────────────
        {
            "role":"user",
            "content":json.dumps({
                "question":"Can all plans use Rule-Based Chat Routing?",
                "agent":"Yes, it’s available on Pro and Enterprise.",
                "ai_short":"Only Enterprise plan includes it.",
                "ai_long":"Rule-Based Chat Routing is exclusive to Live Chat Enterprise.",
                "kb_evidence":[
                    "Rule-Based Chat Routing is only available with the Live Chat Enterprise plan. (source: plans.md)"
                ]
            }, indent=2)
        },
        {
            "role":"assistant",
            "content":json.dumps({
                "ai_score":4,
                "ai_rational":"Agent is slighlty wrong: lists Enterprise but wrongly adds Pro, contradicting KB.",
                "kb_verify":[
                    "Rule-Based Chat Routing is only available with the Live Chat Enterprise plan. (source: plans.md)"
                ]
            })
        },
        # -1 ───────────────────────────────────────────────────────────
        {
            "role":"user",
            "content":json.dumps({
                "question":"How long until a new display name shows in chat?",
                "agent":"It appears instantly on the mobile app.",
                "ai_short":"Propagates within 10 minutes.",
                "ai_long":"Name sync can take up to 10 minutes before visible.",
                "kb_evidence":[
                    "Name changes propagate within 10 minutes. (source: profile.md)"
                ]
            }, indent=2)
        },
        {
            "role":"assistant",
            "content":json.dumps({
                "ai_match":"none",
                "ai_score":-1,
                "ai_rational":
                    "Agent claims 'instantly on mobile'—KB says up to 10 minutes but doesn’t mention mobile. "
                    "Extra info not in KB and not contradicted ⇒ out-of-scope.",
                "kb_verify":[
                    "Name changes propagate within 10 minutes. (source: profile.md)"
                ]
            })
        },
    ]
    
    @staticmethod
    def split_prompt(txt:str)->str:
        return (
            "### Plan\n"
            "- Read transcript.\n"
            "- Group consecutive customer turns pursuing the SAME intent.\n"
            "- Merge them into one `question` block.\n"
            "- Merge corresponding agent replies into one `answer` block.\n"
            "- Size rule: a thread should feel natural—neither single sentences that split context, nor multi-topic blobs.\n\n"
            "### Transcript\n"
            f"{txt}\n\n"
            "### Output\n"
            "Return JSON: {\"threads\":[{\"qid\":\"T1\",\"question\":\"...\",\"answer\":\"...\"}]}"
        )

    # ── STAGE 2  (new signature: q, kb_passages)
    @staticmethod
    def draft_prompt(q: str, passages: List[str]) -> List[Dict]:
        """Return a list of messages: SYSTEM + few-shot + USER."""
        # format passages with indices
        joined = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
        user_msg = (
            "### Plan\n"
            "- Draft short concise and verified answer from the provided passages below.\n"
            "- Draft richer `long` answer covering details and reasoning based on evidence from source passages provided below.\n\n"
            "### Passages\n"
            f"{joined}\n\n"
            "### Question\n"
            f"{q}\n\n"
            "### Output format\n"
            "{\"short\":{...},\"long\":{...}}"
        )
        return (
            [{"role":"system","content":PromptBuilder.SYSTEM_DRAFT}]
            + PromptBuilder.FEW_SHOT
            + [{"role":"user","content":user_msg}]
        )

    # ── STAGE-3 PROMPT CONSTRUCTOR
    @staticmethod
    def grade_prompt(bundle:Dict) -> List[Dict]:
        """Return full message list: SYSTEM + few-shot + current bundle."""
        user_msg = (
            "### Plan\n"
            "- Compare agent with kb_evidence the and use ai short answer and ai long answer as reference.\n"
            "- Decide ai_score per rubric.\n\n"
            "### Input\n" + json.dumps(bundle, ensure_ascii=False, indent=2) +
            """
            # OUTPUT FORMAT
            {
              "ai_score"   : 0-5,
              "ai_rational": "...",
              "kb_verify"  : ["KB line 1", "KB line 2", …]
            }
            
            
            """
            
        )
        return (
            [{"role":"system", "content":PromptBuilder.SYSTEM_GRADE}]
            + PromptBuilder.FEW_GRADE
            + [{"role":"user", "content":user_msg}]
        )

PB = PromptBuilder()

# ───────────────────── BUILD / LOAD KB ─────────────────────
def build_or_load_collection(name: str, path: str, glob_path: str,
                             chunk_tokens: int = 1000, overlap: int = 200):
    os.makedirs(path, exist_ok=True)
    try:
        client = chromadb.PersistentClient(
            path=path, settings=Settings(),
            tenant="default_tenant", database="default_database"
        )
    except:
        client = chromadb.Client(Settings(is_persistent=False))

    embed_fn = OpenAIEmbeddingFunction(
        api_key=openai.api_key, model_name=EMBED_MODEL
    )

    try:
        col = client.get_collection(name)
    except:
        col = client.create_collection(name=name, embedding_function=embed_fn)

    if col.count() == 0:
        passages, metas = [], []
        for md in tqdm(list(Path().glob(glob_path)), desc="Chunking KB"):
            raw = md.read_text(encoding="utf-8")
            for i in range(0, len(raw), chunk_tokens - overlap):
                passages.append(raw[i:i+chunk_tokens])
                metas.append({"source": str(md)})
        ids = [str(i) for i in range(len(passages))]
        col.add(ids=ids, documents=passages, metadatas=metas)

    return col

COL_KB = build_or_load_collection("kb_docs", CHROMA_DIR, GLOB_PATH)
print(f"✅  KB ready ({COL_KB.count()} docs)")

def safe_json(txt: str) -> Dict:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON object found")
    return json.loads(m.group(0))

# ──────────────────── CHROMA RETRIEVAL ─────────────────────
def fetch_chunks(col, query: str, k: int = TOP_K_KB) -> List[str]:
    emb_resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_emb = emb_resp.data[0].embedding
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs, metas = res["documents"][0], res["metadatas"][0]
    return [
        f"{d.strip()}  (source: {m.get('source','n/a')})"
        for d, m in zip(docs, metas) if d
    ]

# ── helper for stages that need just ONE user string (split) ------------
def chat_simple(user_text: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        # temperature=TEMP,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PB.system},
            {"role": "user",   "content": user_text}
        ]
    )
    return resp.choices[0].message.content


# ── Stage-2 : draft two AI answers --------------------------------------
def gen_ai_answers(question: str, passages: List[str]) -> Dict:
    messages = PB.draft_prompt(question, passages)   # ← full list
    raw = openai.chat.completions.create(
        model=MODEL,
        # temperature=TEMP,
        messages=messages,
        response_format={"type": "json_object"},
    ).choices[0].message.content
    return safe_json(raw)


# ── Stage-3 : grade the agent answer ------------------------------------
def score_pair(question: str, agent_ans: str,
               drafts: Dict, kb: List[str]) -> Dict:
    bundle = {
        "question": question,
        "agent":    agent_ans,
        "ai_short": drafts["short"]["answer"],
        "ai_long":  drafts["long"]["answer"],
        "kb_evidence": kb
    }
    messages = PB.grade_prompt(bundle)               # ← full list
    raw = openai.chat.completions.create(
        model=MODEL,
        # temperature=TEMP,
        response_format={"type": "json_object"},
        messages=messages
    ).choices[0].message.content
    return safe_json(raw)


# ── Stage-1 : thread splitter (string prompt) ---------------------------
def split_pairs(transcript: str) -> List[Dict]:
    raw = chat_simple(PB.split_prompt(transcript))
    threads = safe_json(raw)["threads"]
    for t in threads:
        t["qid"] = t.get("qid") or str(uuid.uuid4())
    return threads

# ─────────────────────── MAIN ──────────────────────────
def main():
    # STEP 1: Split chat into threads
    pairs = split_pairs(CHAT_TRANSCRIPT)
    print(f"🧵 Found {len(pairs)} Q/A pairs\n")

    qa_records = []
    
    # Process each pair: retrieve KB → generate → grade
    for p in tqdm(pairs, desc="Auditing"):
        qid, q, a = p["qid"], p["question"], p["answer"]
        kb_chunks = fetch_chunks(COL_KB, q)
        drafts   = gen_ai_answers(q, kb_chunks)
        grade    = score_pair(q, a, drafts, kb_chunks)

        qa_records.append({
            "id":       qid,
            "site":     "Comm100",
            "origin":   "Chat",
            "reviewed": dt.datetime.utcnow().isoformat(),
            "accuracy": grade["ai_score"],            # 0–5 scale
            "detail": [{
                "rewritten_q": q,
                "agent_ans":   a,
                "short_ans":   drafts["short"]["answer"],
                "long_ans":    drafts["long"]["answer"],
                "ai_score":    grade["ai_score"],
                "ai_rational": grade["ai_rational"],
                "kb_verify":   kb_chunks,
            }]
        })

    df = pd.DataFrame([r["accuracy"] for r in qa_records], columns=["accuracy"])
    avg = df.accuracy.mean() or 0
    print(f"\n===== TECH ACCURACY: {avg:.2f}/5 over {len(df)} pairs =====")

    Path(OUT_FILE).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in qa_records)
    )
    print(f"📄 Report saved to {OUT_FILE}\n")
    print("===== FULL QA REPORT =====")
    print(json.dumps(qa_records, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
