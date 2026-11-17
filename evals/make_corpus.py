"""Build the evaluation corpus — self-authored, fictional sample documents.

Four multi-page PDFs (one topic per page, so every gold question maps to an
unambiguous page). No real company, person, or copyrighted text. Regenerate with:

    .venv/bin/python evals/make_corpus.py
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent / "corpus"

# doc_id -> list of (page_title, page_body), one topic per page.
DOCS: dict[str, list[tuple[str, str]]] = {
    "service_agreement": [
        ("Parties and Effective Date",
         "This Service Agreement is entered into between Northwind Trading Company (the Provider) "
         "and Blue Harbor Logistics (the Client). It takes effect on March 1, 2025 and is governed "
         "by the terms set out in the following sections."),
        ("Term and Renewal",
         "The initial term of this agreement is twelve (12) months. It renews automatically for "
         "successive twelve-month periods unless either party gives written notice of non-renewal "
         "at least sixty (60) days before the end of the current term."),
        ("Termination",
         "Either party may terminate this agreement for convenience with ninety (90) days written "
         "notice. The agreement may be terminated immediately for a material breach that remains "
         "uncured after fifteen (15) days following written notice of the breach."),
        ("Payment and Fees",
         "Invoices are issued monthly and payment is due within thirty (30) days of the invoice "
         "date. Late payments accrue interest at 1.5 percent per month. Any disputed charge must be "
         "raised in writing within ten (10) business days of the invoice."),
        ("Confidentiality",
         "Each party agrees to keep the other party's confidential information secret and to use it "
         "only to perform this agreement. These confidentiality obligations survive for three (3) "
         "years after termination of the agreement."),
        ("Liability and Indemnity",
         "Each party's total liability under this agreement is capped at the fees paid in the twelve "
         "months preceding the claim. Neither party is liable for indirect or consequential damages. "
         "The Provider will indemnify the Client against third-party intellectual property claims."),
        ("Intellectual Property",
         "All pre-existing intellectual property remains owned by the party that created it. "
         "Deliverables created specifically for the Client under this agreement are assigned to the "
         "Client upon full payment of the related fees."),
        ("Governing Law and Disputes",
         "This agreement is governed by the laws of the State of Delaware. Disputes are first "
         "escalated to senior management, and if unresolved within thirty (30) days, submitted to "
         "binding arbitration in Wilmington, Delaware."),
    ],
    "employee_handbook": [
        ("Working Hours and Remote Work",
         "Standard working hours are 9:00 AM to 5:00 PM, Monday through Friday. Employees may work "
         "remotely up to three (3) days per week with manager approval. Core collaboration hours, "
         "when everyone should be available, are 11:00 AM to 3:00 PM."),
        ("Paid Leave and Vacation",
         "Full-time employees receive twenty (20) days of paid vacation per year, accruing monthly. "
         "Up to five (5) unused days may be carried into the next year. Vacation requests should be "
         "submitted to a manager at least two weeks in advance."),
        ("Sick Leave and Medical",
         "Employees are entitled to ten (10) paid sick days per year. A doctor's note is required "
         "for any absence longer than three (3) consecutive days. Unused sick days do not carry over "
         "to the following year."),
        ("Code of Conduct",
         "All employees must act with integrity and treat colleagues with respect. Harassment and "
         "discrimination of any kind are strictly prohibited. Violations may result in disciplinary "
         "action up to and including termination of employment."),
        ("Expense Reimbursement",
         "Business expenses are reimbursed when submitted with receipts within thirty (30) days. "
         "Meals during business travel are reimbursed up to 50 dollars per day. Air travel should be "
         "booked in economy class unless otherwise approved."),
        ("Equipment and Security",
         "Company laptops must have full-disk encryption and an automatic screen lock enabled. "
         "Account passwords must be at least twelve (12) characters. Lost or stolen devices must be "
         "reported to the IT team within twenty-four (24) hours."),
        ("Performance Reviews",
         "Performance reviews are conducted twice per year, in June and December. Reviews assess "
         "goals, competencies, and professional growth. Promotions and salary adjustments are "
         "decided during the December review cycle."),
        ("Resignation and Notice",
         "Employees who resign must provide at least two (2) weeks of written notice. The final "
         "paycheck includes any accrued but unused vacation. All company equipment must be returned "
         "on or before the last working day."),
    ],
    "product_manual": [
        ("Setup and Installation",
         "Unpack the Aurora X1 air purifier and place it on a flat surface at least 30 centimeters "
         "from any wall. Insert the filter, connect the power adapter, and press the power button "
         "for two seconds to turn the unit on."),
        ("Technical Specifications",
         "The Aurora X1 covers rooms up to 40 square meters. It operates between 20 and 50 decibels, "
         "consumes 45 watts at maximum fan speed, and weighs 4.2 kilograms. The HEPA filter captures "
         "99.97 percent of particles down to 0.3 microns."),
        ("Operating Instructions",
         "Use the mode button to cycle between Auto, Sleep, and Turbo modes. In Auto mode the unit "
         "adjusts fan speed based on its air-quality sensor. The filter indicator turns red when the "
         "filter needs to be replaced."),
        ("Troubleshooting",
         "If the unit does not turn on, check that the power adapter is firmly connected. If airflow "
         "is weak, verify that the filter is not clogged. A blinking red light indicates that the "
         "filter must be replaced before continuing use."),
        ("Warranty",
         "The Aurora X1 includes a two (2) year limited warranty covering manufacturing defects. The "
         "warranty does not cover damage caused by misuse or unauthorized repairs. Keep your proof "
         "of purchase to make a warranty claim."),
        ("Safety and Compliance",
         "Do not operate the unit in wet locations or cover the air intake. Always unplug the unit "
         "before cleaning. The Aurora X1 complies with FCC Part 15 and is certified to the IEC 60335 "
         "safety standard."),
    ],
    "privacy_policy": [
        ("Data We Collect",
         "We collect account information such as your name and email address, usage data about how "
         "you interact with the service, and device information including your IP address and "
         "browser type."),
        ("How We Use Data",
         "We use your data to provide and improve the service, personalize your experience, process "
         "payments, and send service-related notifications. We do not sell your personal data to "
         "third parties."),
        ("Data Retention",
         "Account data is retained for as long as your account remains active. After you close your "
         "account, we delete your personal data within ninety (90) days, except where longer "
         "retention is required by law."),
        ("Sharing and Third Parties",
         "We share data only with service providers who help us operate the service, and only under "
         "a written contract. We may disclose data if required by law or to protect our legal "
         "rights."),
        ("Your Rights",
         "You may request access to, correction of, or deletion of your personal data at any time. "
         "You may also object to certain processing and request a copy of your data in a portable, "
         "machine-readable format."),
        ("Security and Contact",
         "We protect data using encryption in transit and at rest, strict access controls, and "
         "regular security reviews. For privacy questions, contact privacy@example.com. We respond "
         "to verified requests within thirty (30) days."),
    ],
    "api_reference": [
        ("Authentication",
         "All requests must include the API key in the X-API-Key header. Keys are issued per project "
         "in the developer console. A request without a valid key returns HTTP 401 Unauthorized."),
        ("Base URL and Versioning",
         "The current API version is v3 and the base URL is https://api.example.com/v3. The previous "
         "version v2 is deprecated and will be removed on December 31, 2025."),
        ("Ingest Endpoint",
         "Documents are uploaded with a POST request to the /v3/ingest endpoint. The maximum file "
         "size is 25 megabytes. Supported formats are PDF, DOCX, and TXT."),
        ("Search Endpoint",
         "Use GET /v3/search with a required query parameter q and an optional top_k parameter. The "
         "default top_k is 5 and the maximum allowed value is 50."),
        ("Rate Limits",
         "The default rate limit is 60 requests per minute per API key. Exceeding it returns HTTP 429 "
         "with a Retry-After header. Enterprise plans allow 600 requests per minute."),
        ("Error Codes",
         "Error E-4001 indicates an invalid request body. Error E-4096 indicates the uploaded file "
         "exceeds the size limit. Error E-5003 indicates an internal processing failure."),
        ("Configuration Keys",
         "Set MAX_BATCH_SIZE to control how many documents are processed together. Set EMBED_MODEL to "
         "choose the embedding model. Set REQUEST_TIMEOUT_SECONDS for the per-request timeout."),
        ("SDK and Webhooks",
         "The official Python SDK is installed with pip install example-sdk. Webhook notifications are "
         "signed with an HMAC-SHA256 signature carried in the X-Signature header."),
    ],
    "security_policy": [
        ("Standards and Compliance",
         "The company maintains ISO 27001 certification and completes an annual SOC 2 Type II audit. "
         "Compliance reports are available to enterprise customers under NDA."),
        ("Encryption",
         "Data is encrypted in transit using TLS 1.3 and at rest using AES-256. Encryption keys are "
         "rotated every ninety (90) days and stored in a hardware security module."),
        ("Access Control",
         "Access follows the principle of least privilege. Multi-factor authentication is required for "
         "all administrative accounts. Access reviews are conducted quarterly."),
        ("Incident Response",
         "Security incidents must be reported to the security team within one (1) hour. Confirmed "
         "breaches affecting customer data are disclosed to affected customers within seventy-two (72) "
         "hours."),
        ("Vulnerability Management",
         "Critical vulnerabilities are patched within seven (7) days of disclosure. The company runs "
         "quarterly penetration tests and a continuous bug-bounty program."),
        ("Password Policy",
         "Administrative passwords must be at least sixteen (16) characters and rotated every sixty "
         "(60) days. A new password is rejected if it matches any of the last ten passwords."),
    ],
    "release_notes": [
        ("Version 3.2.0",
         "Release 3.2.0 adds hybrid retrieval combining dense and keyword search. It also introduces "
         "a rerank parameter on the /v3/search endpoint."),
        ("Version 3.1.0",
         "Release 3.1.0 adds support for DOCX ingestion and improves table extraction accuracy. Bug "
         "BUG-1287, which caused duplicate chunks on re-index, is fixed."),
        ("Version 3.0.0",
         "Release 3.0.0 is a major version that changes the base URL to /v3 and removes the legacy "
         "/v1 endpoints. Migration guidance is in the documentation."),
        ("Known Issues",
         "Issue BUG-1342 causes slow ingestion for files larger than 20 megabytes. The workaround is "
         "to split large files. A fix is planned for release 3.3.0."),
        ("Deprecations",
         "The v2 API is deprecated as of release 3.0.0. The EMBED_MODEL default changes from minilm to "
         "bge-small in release 3.3.0."),
    ],
}


def build(doc_id: str, pages: list[tuple[str, str]]) -> Path:
    doc = pymupdf.open()
    rect = pymupdf.Rect(60, 70, 540, 760)
    for title, body in pages:
        page = doc.new_page()
        page.insert_textbox(rect, f"{title.upper()}\n\n{body}", fontsize=13, fontname="helv")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{doc_id}.pdf"
    doc.save(str(path))
    doc.close()
    return path


def main() -> None:
    for doc_id, pages in DOCS.items():
        path = build(doc_id, pages)
        print(f"wrote {path.relative_to(OUT.parent.parent)}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
