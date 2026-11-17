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
