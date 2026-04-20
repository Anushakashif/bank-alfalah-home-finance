# rag.py — Core RAG system (chunking, embedding, retrieval)

import os
import json
import math
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Raw Policy Document ────────────────────────────────────────────────────────
# In a real project this would be loaded from a PDF file.
# For now the policy text is stored here directly.

POLICY_TEXT = """
SECTION: Overview
Bank Alfalah offers flexible and accessible home financing solutions designed for salaried individuals, self-employed professionals, Non-Resident Pakistanis (NRPs), and families looking to buy, build, renovate, or go green with solar energy. Home Finance plans offer competitive pricing, long tenures, and value-added benefits. Whether starting fresh or transferring an existing home loan, Bank Alfalah makes homeownership easier, faster, and more secure.

SECTION: Markup Rates
Bank Alfalah Home Finance is offered on a floating rate basis. The markup rates are as follows:
Local Salaried customers are charged 1 Year KIBOR plus 3 percent per annum.
Self-Employed professionals, Businessmen, and Others are charged 1 Year KIBOR plus 4 percent per annum.
Non-Resident Pakistanis (NRP) are charged 1 Year KIBOR plus 4 percent per annum.
Customers may opt for 6-month KIBOR as a benchmark, however this is not applicable for Roshan Apna Ghar.
Customized products with fixed or hybrid pricing options may also be available subject to terms and conditions.

SECTION: Repayment Tenure
The minimum repayment tenure for Home Finance is 3 years. The maximum repayment tenure is 25 years. For Home Solar Finance, the tenure ranges from 3 to 10 years.

SECTION: Down Payment and Equity
A minimum equity contribution of 20 percent of property value is generally required, subject to terms and conditions. The exact equity required depends on the customer's income segment and financing amount. Home Finance is only available for residential properties, not commercial ones.

SECTION: Home Buyer Product
The Home Buyer product has no minimum or maximum financing limit. It provides up to 80 percent financing of property value. An independent property valuation is conducted to determine fair market value.

SECTION: Plot and Build Product
The Plot and Build product provides up to 80 percent financing of the total home value. 50 percent is allocated for plot purchase and 50 percent for construction. Construction amount is disbursed in 4 equal tranches.

SECTION: Build Your Home Product
The Build Your Home product is for customers who already own a plot. It provides up to 100 percent financing of construction cost. The amount is disbursed in 4 equal tranches.

SECTION: Home Improvement Product
The Home Improvement product provides financing to renovate or expand an existing home. The amount is disbursed in 2 tranches or as per approval.

SECTION: Home BTF Balance Transfer Facility
The Home BTF product allows customers to transfer up to 100 percent of their existing home finance from another bank to Bank Alfalah.

SECTION: Home Secure Product
The Home Secure product allows customers to get 100 percent financing even without upfront equity. Customers can mortgage a family house or plot, or place a lien on a deposit. It can be used to finance a new property or plot plus construction.

SECTION: Mera Ghar Meri Pehchaan Women Financing
Mera Ghar Meri Pehchaan is an exclusive financing product for women. Women customers get up to 100 basis points rate reduction. Salaried women get 50 basis points rate reduction. There is a 25 percent waiver on processing fee. There is also a 25 percent waiver on early settlement and balloon charges.

SECTION: Home Solar Finance
Home Solar Finance is for customers who want to invest in solar energy. It is offered in partnership with reliable solar vendors approved by the bank. Tenure ranges from 3 to 10 years. Customers need to provide a copy of quotation from a Bank Alfalah approved solar vendor.

SECTION: Roshan Apna Ghar for Non-Resident Pakistanis
Roshan Apna Ghar is a home financing solution for Non-Resident Pakistanis. It is available to Bank Alfalah Roshan Digital Account RDA customers. It offers convenient and fast processing for Pakistanis living abroad.

SECTION: Eligibility for Salaried Individuals
Salaried applicants must hold a valid CNIC, SNIC, NICOP, or POC. The minimum net monthly income for local salaried applicants is PKR 75,000. For NRP salaried applicants the minimum is USD 3,000 or equivalent. The minimum age is 25 years and the maximum age is until retirement. Permanent employees require a minimum of 2 years of experience. Contractual employees require 3 to 5 years of experience.

SECTION: Eligibility for Self-Employed and Businessmen
Self-employed applicants and businessmen must hold a valid CNIC, SNIC, NICOP, or POC. The minimum net monthly income for local applicants is PKR 150,000. For NRP self-employed applicants the minimum is USD 4,000 or equivalent. The minimum age is 25 years and the maximum age is 70 years. A minimum of 3 years in business is required.

SECTION: Co-Borrowers
Bank Alfalah allows up to 2 co-borrowers. Co-borrowers must be blood relatives or spouse, such as parents, children, or siblings. Co-borrowers can combine their income to enhance financial eligibility. Co-borrowers can be included for joint property ownership, income clubbing, or both.

SECTION: Documents Required for All Applicants
All applicants must submit a completed Loan Application Form either physically or through the RAPID online portal. A copy of CNIC, SNIC, NICOP, or POC for applicant and co-applicant is required. Color photographs of the applicant and co-applicant are needed. Proof of employment, business, or profession must be provided. Documentary evidence of all income sources is required. Copies of all property title documents must be submitted if the property has already been selected.

SECTION: Documents Required for Salaried Applicants
Salaried applicants must submit their latest salary slips or an employment letter that states date of joining, salary breakup, and job status. Last 6 months bank statement is required.

SECTION: Documents Required for Self-Employed Applicants
Self-employed applicants and businessmen must submit last 1 year bank statement, latest tax returns, and audited financials if available. Income estimation may be required. A professional certificate is needed for self-employed professionals. Sole proprietorship proof is required for sole proprietors. Partnership deed and Form-C if registered is required for partnerships. For limited companies, MOA, Form A, and Form 29 are required.

SECTION: Documents Required for Non-Resident Pakistanis
NRP applicants must submit a copy of valid passport and work or residence permit or Iqama. Salaried NRPs must submit latest salary slips or employment letter. Self-employed NRPs must submit valid business proof and income documents. Latest bank statement is required, 6 months for salaried and 12 months for others. A Credit Bureau Report from the country of residence is needed. Proof of residence such as utility bills or notarized rent agreement is required. Proof of business such as commercial registration certificate or licenses is also needed.

SECTION: Application Process Steps
The application process begins with submitting a home finance application physically at a branch or online through the RAPID portal. The bank then receives and pre-screens submitted documents. A legal opinion Stage 1 and independent property valuation are conducted. The application goes through credit approval by the bank's internal credit department. Once approved, a Facility Offer Letter is issued with financing terms. Life and property insurance are arranged and a bank account is opened. The customer then provides instruction for disbursement.
Bank Alfalah also offers in-principle approval before a property is selected. This allows customers to know the estimated financing amount and required equity in advance. The in-principle approval is valid for 60 days from the date of approval.

SECTION: Repayment Methods
Home Finance repayments can be made through post-dated cheques. Customers can also make monthly cash or cheque deposits into their repayment account. Direct debit from a designated Bank Alfalah account is also available.

SECTION: Partial and Early Repayment
An annual partial payment or balloon payment facility is available to help customers repay faster. A late payment penalty applies and must be settled at end of financing tenure or upon early settlement. For exact charges on early settlement customers should refer to Bank Alfalah's Schedule of Charges.

SECTION: Insurance Requirements
Life insurance is required and is arranged during the formalities stage. Property insurance is also required and is arranged during the formalities stage.

SECTION: Additional Benefits
Bank Alfalah provides legal counseling to verify property title documents. An independent property valuation is conducted for fair market value. Customers may be eligible to claim income tax rebate on markup payments and should consult a tax advisor. Financing is available for both local residents and Non-Resident Pakistanis.

SECTION: Contact and Application
Customers can apply online through the RAPID portal on Bank Alfalah's website. Customers can also visit the nearest Bank Alfalah branch or Consumer Finance Center. The Bank Alfalah helpline number is 111-225-111. The website is bankalfalah.com.
"""


# ── Step 1: Chunking ───────────────────────────────────────────────────────────
def chunk_document(text: str) -> list[dict]:
    """
    Split the policy document into chunks by SECTION headers.
    Each chunk has a title and content.
    """
    chunks = []
    sections = text.strip().split("SECTION:")

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # First line is the section title, rest is content
        lines = section.split("\n", 1)
        title = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""

        if title and content:
            chunks.append({
                "title": title,
                "content": content,
                "full_text": f"{title}: {content}"
            })

    return chunks


# ── Step 2: Embedding ──────────────────────────────────────────────────────────
def get_embedding(text: str, client: Groq) -> list[float]:
    """
    Generate a simple TF-IDF style bag-of-words embedding.
    
    NOTE: Groq does not provide an embedding API.
    In production you would use sentence-transformers or OpenAI embeddings.
    For this portfolio project we use a lightweight keyword-based vector
    that works well for policy Q&A without any extra dependencies.
    """
    # Vocabulary of important home finance keywords
    keywords = [
        "markup", "rate", "kibor", "interest", "profit",
        "salaried", "salary", "income", "employed", "self-employed", "business",
        "nrp", "non-resident", "expatriate", "overseas", "abroad",
        "eligibility", "eligible", "qualify", "requirement",
        "document", "cnic", "passport", "statement", "slip", "letter",
        "tenure", "years", "duration", "repayment", "period",
        "equity", "down payment", "contribution", "percentage",
        "financing", "amount", "limit", "maximum", "minimum",
        "home buyer", "purchase", "buy", "plot", "build", "construction",
        "renovation", "improvement", "solar", "green", "energy",
        "transfer", "btf", "balance", "existing",
        "women", "female", "mera ghar", "pehchaan",
        "roshan", "rda", "digital account",
        "co-borrower", "spouse", "relative", "joint",
        "insurance", "life", "property",
        "apply", "application", "process", "steps", "rapid", "portal",
        "fee", "charges", "penalty", "settlement", "balloon",
        "age", "experience", "permanent", "contractual",
        "legal", "title", "valuation", "opinion",
        "75000", "150000", "3000", "4000", "20", "80", "100",
        "overview", "general", "about", "what"
    ]

    text_lower = text.lower()
    vector = []
    for kw in keywords:
        # Count keyword occurrences, normalize to 0-1
        count = text_lower.count(kw.lower())
        vector.append(min(count, 5) / 5.0)  # cap at 5 occurrences

    return vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Step 3: Build Vector Store ─────────────────────────────────────────────────
def build_vector_store(client: Groq) -> list[dict]:
    """
    Chunk the document and embed each chunk.
    Returns a list of chunks with their embeddings attached.
    """
    chunks = chunk_document(POLICY_TEXT)
    vector_store = []

    for chunk in chunks:
        embedding = get_embedding(chunk["full_text"], client)
        vector_store.append({
            "title": chunk["title"],
            "content": chunk["content"],
            "full_text": chunk["full_text"],
            "embedding": embedding
        })

    return vector_store


# ── Step 4: Retrieve Relevant Chunks ──────────────────────────────────────────
def retrieve(query: str, vector_store: list[dict], top_k: int = 3) -> list[dict]:
    """
    Given a user query, find the top_k most relevant chunks
    using cosine similarity between query embedding and chunk embeddings.
    """
    query_embedding = get_embedding(query, None)

    scored = []
    for chunk in vector_store:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))

    # Sort by similarity score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top_k chunks
    return [chunk for _, chunk in scored[:top_k]]


# ── Step 5: Generate Answer ────────────────────────────────────────────────────
def generate_answer(query: str, retrieved_chunks: list[dict], chat_history: list[dict], client: Groq) -> str:
    """
    Send the retrieved chunks + query to Groq LLM and get an answer.
    """
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[{chunk['title']}]\n{chunk['content']}"
        for chunk in retrieved_chunks
    ])

    system_prompt = f"""You are a professional Home Finance Advisor for Bank Alfalah, one of Pakistan's leading banks.

Answer the customer's question using ONLY the context provided below. 
If the answer is not in the context, say: "I don't have that specific information. Please visit your nearest Bank Alfalah branch or call 111-225-111."
Never make up rates, numbers, or policies.
Be concise and friendly. Use bullet points for lists when helpful.
Respond in the same language the customer uses (Urdu or English).

RELEVANT POLICY CONTEXT:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        *chat_history,
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content


# ── Main RAG Pipeline ──────────────────────────────────────────────────────────
def ask(query: str, vector_store: list[dict], chat_history: list[dict], client: Groq) -> tuple[str, list[str]]:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks for the query
    2. Generate answer from those chunks
    Returns the answer and the list of source section titles used.
    """
    retrieved_chunks = retrieve(query, vector_store, top_k=3)
    answer = generate_answer(query, retrieved_chunks, chat_history, client)
    sources = [chunk["title"] for chunk in retrieved_chunks]
    return answer, sources
