import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.entity_pools import EntityPoolGenerator

# Public figures for Public Documents (Only PERSON allowed in public docs)
PUBLIC_FIGURES = [
    "Dr. Arshad Khan", "Sarah Jenkins", "Dr. Ayesha Malik", "Robert Miller",
    "Prof. Tariq Siddiqui", "Elizabeth Davis", "Muhammad Usman", "Jessica Taylor",
    "Dr. Bilal Chaudhry", "David Anderson", "Fatima Raza", "Michael Wilson"
]

class DocumentBuilder:
    def __init__(self, doc_id: str, title: str, access_level: str):
        self.doc_id = doc_id
        self.title = title
        self.access_level = access_level
        self.text = ""
        self.spans = []

    def append_text(self, s: str):
        self.text += s

    def append_entity(self, val: str, label: str):
        val_clean = val.strip()
        start = len(self.text)
        self.text += val_clean
        end = len(self.text)
        self.spans.append({
            "start": start,
            "end": end,
            "label": label,
            "text": val_clean
        })

    def build(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": f"{self.doc_id}.md",
            "title": self.title,
            "access_level": self.access_level,
            "text": self.text,
            "spans": self.spans,
            "entities_present": sorted(list({s["label"] for s in self.spans})),
            "sensitive_values": sorted(list({s["text"] for s in self.spans}))
        }

def generate_public_doc(idx: int, pool: EntityPoolGenerator) -> Dict[str, Any]:
    doc_id = f"doc_public_{idx:03d}"
    titles = [
        "Corporate Annual Governance and Ethics Report",
        "Public Cloud Architecture and Technology Whitepaper",
        "Global Corporate Sustainability and Green Initiative",
        "Open Source Software Contribution and License Policy",
        "Executive Leadership Directory and Biographies",
        "Enterprise Product Roadmap and Vision 2027",
        "Public Security, Privacy, and SOC2 Compliance Summary",
        "Diversity, Equity, and Inclusion Annual Overview",
        "Investor Relations and Financial Highlights Summary",
        "Customer Service Charter and Community Standards"
    ]
    title = f"{titles[(idx - 1) % len(titles)]} (Part {((idx - 1) // len(titles)) + 1})"
    builder = DocumentBuilder(doc_id, title, "public")
    
    pub_figure = PUBLIC_FIGURES[(idx - 1) % len(PUBLIC_FIGURES)]
    
    builder.append_text(f"# {title}\n\n")
    builder.append_text(f"**Classification:** PUBLIC | **Document ID:** {doc_id}\n\n")
    builder.append_text("## 1. Executive Summary\n")
    builder.append_text("Acme Global Corporation remains committed to transparency, open communication, and robust technology standards. ")
    builder.append_text("Under the continuous guidance of executive leadership, including ")
    builder.append_entity(pub_figure, "PERSON")
    builder.append_text(", our global teams continue delivering reliable enterprise services worldwide.\n\n")
    
    builder.append_text("## 2. Strategic Objectives\n")
    builder.append_text("Our primary mission encompasses three core principles: innovation, high operational availability, and community trust. ")
    builder.append_text("We actively promote public disclosures of our compliance frameworks, including SOC2 Type II and ISO 27001 certifications. ")
    builder.append_text("All public documentation is reviewed quarterly by corporate governance committees.\n\n")
    
    builder.append_text("## 3. Public Governance and Leadership\n")
    builder.append_text(f"For general press inquiries and stakeholder communications, the corporate office maintains dedicated public channels. ")
    builder.append_text("Key policy statements endorsed by ")
    builder.append_entity(pub_figure, "PERSON")
    builder.append_text(" ensure that our open standards align with international regulatory guidelines.\n")
    
    return builder.build()

def generate_internal_doc(idx: int, pool: EntityPoolGenerator) -> Dict[str, Any]:
    doc_id = f"doc_internal_{idx:03d}"
    titles = [
        "IT Infrastructure Deployment and Kubernetes Cluster Runbook",
        "Engineering Best Practices and Code Review Guidelines",
        "Office IT Equipment Request and Provisioning Protocol",
        "Internal Incident Escalation Matrix and On-Call Rotations",
        "Corporate VPN Setup and Secure Remote Work Policy",
        "Software Architecture Decision Record - Microservices Migration",
        "Departmental Sprint Planning and Agile Delivery Framework",
        "Internal Travel Booking and Expense Reimbursement Guidelines",
        "Facilities Access and Building Security Regulations",
        "Company-Wide Technology Stack and Deprecation Schedule"
    ]
    title = f"{titles[(idx - 1) % len(titles)]} (Rev {((idx - 1) // len(titles)) + 1})"
    builder = DocumentBuilder(doc_id, title, "internal")
    
    lead_name = pool.generate_person()
    emp_id = pool.generate_emp_id()
    email = pool.generate_email(lead_name)
    phone = pool.generate_phone()
    
    builder.append_text(f"# {title}\n\n")
    builder.append_text(f"**Classification:** INTERNAL ONLY | **Document ID:** {doc_id}\n\n")
    builder.append_text("## 1. Scope and Objective\n")
    builder.append_text("This standard operating procedure is restricted to full-time employees, authorized contractors, and internal staff. ")
    builder.append_text("The operational guidelines described herein govern day-to-day engineering, security configurations, and facilities access.\n\n")
    
    builder.append_text("## 2. Operational Procedures and Governance\n")
    builder.append_text("All systems must adhere to strict zero-trust network policies. ")
    builder.append_text("Access keys must be rotated every 90 days, and production deploys require two peer reviews. ")
    builder.append_text("For urgent escalations or system failures, staff must follow the internal on-call protocol.\n\n")
    
    builder.append_text("## 3. Technical Contact and Ownership\n")
    builder.append_text("The designated system maintainer and process lead for this protocol is ")
    builder.append_entity(lead_name, "PERSON")
    builder.append_text(" (Staff ID: ")
    builder.append_entity(emp_id, "EMP_ID")
    builder.append_text("). For direct questions regarding these technical instructions, email ")
    builder.append_entity(email, "EMAIL")
    builder.append_text(" or call office extension ")
    builder.append_entity(phone, "PHONE")
    builder.append_text(".\n")
    
    return builder.build()

def generate_hr_doc(idx: int, pool: EntityPoolGenerator) -> Dict[str, Any]:
    doc_id = f"doc_hr_{idx:03d}"
    titles = [
        "Confidential Compensation Review and Salary Adjustment Memo",
        "Annual Performance Appraisal Dossier and Bonus Determination",
        "Executive Offer Letter and Employment Agreement",
        "Disciplinary Investigation Report and Formal Warning",
        "Employee Domicile and Comprehensive Benefits Enrollment Sheet",
        "Departmental Payroll Batch Reconciliation Audit",
        "Severance Determination and Confidential Exit Agreement",
        "Confidential Medical Leave and Disability Support Record",
        "Talent Acquisition Offer Package and Compensation Breakdown",
        "Human Resources Background Investigation Summary"
    ]
    title = f"{titles[(idx - 1) % len(titles)]} - Case #{1000 + idx}"
    builder = DocumentBuilder(doc_id, title, "hr_restricted")
    
    emp_name = pool.generate_person()
    emp_id = pool.generate_emp_id()
    salary = pool.generate_salary()
    addr = pool.generate_address()
    email = pool.generate_email(emp_name)
    phone = pool.generate_phone()
    
    builder.append_text(f"# {title}\n\n")
    builder.append_text(f"**Classification:** HR RESTRICTED | **Document ID:** {doc_id}\n\n")
    builder.append_text("## 1. Confidential Employee Profile\n")
    builder.append_text("This document contains sensitive Personally Identifiable Information (PII) and payroll records protected under HR data policies. ")
    builder.append_text("Unauthorized dissemination or review will result in immediate disciplinary action.\n\n")
    builder.append_text(f"- **Full Employee Name:** ")
    builder.append_entity(emp_name, "PERSON")
    builder.append_text(f"\n- **Employee Identification Code:** ")
    builder.append_entity(emp_id, "EMP_ID")
    builder.append_text(f"\n- **Residential Domicile:** ")
    builder.append_entity(addr, "ADDRESS")
    builder.append_text(f"\n- **Primary Email:** ")
    builder.append_entity(email, "EMAIL")
    builder.append_text(f"\n- **Telephone Line:** ")
    builder.append_entity(phone, "PHONE")
    builder.append_text("\n\n")
    
    builder.append_text("## 2. Compensation and Remuneration Details\n")
    builder.append_text("Following the annual departmental performance audit, the HR Compensation Committee reviewed all wage bandings. ")
    builder.append_text("The confirmed annual base compensation for ")
    builder.append_entity(emp_name, "PERSON")
    builder.append_text(" is officially approved at ")
    builder.append_entity(salary, "SALARY")
    builder.append_text(" per annum, payable in monthly disbursements.\n\n")
    
    builder.append_text("## 3. Compliance and Sign-Off\n")
    builder.append_text("All tax deduction certificates, health insurance coverage, and official notices will be dispatched to ")
    builder.append_entity(addr, "ADDRESS")
    builder.append_text(". For any payroll disputes, contact Human Resources directly.\n")
    
    return builder.build()

def generate_exec_doc(idx: int, pool: EntityPoolGenerator) -> Dict[str, Any]:
    doc_id = f"doc_exec_{idx:03d}"
    titles = [
        "Board of Directors Compensation Committee Confidential Minutes",
        "C-Suite Executive Employment Agreement and Equity Grant",
        "Confidential Mergers & Acquisitions Executive Retention Brief",
        "Senior Leadership Restructuring and Succession Strategy",
        "Strategic Enterprise Valuation and Capital Allocation Memo",
        "Executive Bonus Pool Determination and Performance Multipliers",
        "Chief Executive Officer Strategic Advisory and Compensation Package",
        "Confidential Intellectual Property Settlement and Officer Indemnity",
        "Executive Domicile Security and Personal Protection Stipend",
        "Senior Officer Severance Contingency and Retirement Agreement"
    ]
    title = f"{titles[(idx - 1) % len(titles)]} - Ref EX-{200 + idx}"
    builder = DocumentBuilder(doc_id, title, "exec_restricted")
    
    exec_name = pool.generate_person()
    emp_id = pool.generate_emp_id()
    salary = pool.generate_salary()
    addr = pool.generate_address()
    email = pool.generate_email(exec_name)
    phone = pool.generate_phone()
    
    builder.append_text(f"# {title}\n\n")
    builder.append_text(f"**Classification:** EXEC RESTRICTED | **Document ID:** {doc_id}\n\n")
    builder.append_text("## 1. Highly Confidential Board Resolution\n")
    builder.append_text("This memorandum contains privileged executive compensation, strategic M&A valuations, and officer disclosures. ")
    builder.append_text("Access is strictly limited to members of the Executive Board and authorized corporate legal counsel.\n\n")
    
    builder.append_text("## 2. Executive Appointment and Financial Terms\n")
    builder.append_text("The Executive Compensation Committee has ratified the contractual package for ")
    builder.append_entity(exec_name, "PERSON")
    builder.append_text(" (Designated Officer ID: ")
    builder.append_entity(emp_id, "EMP_ID")
    builder.append_text("). Under the terms of the executive charter, the agreed baseline executive remuneration is established at ")
    builder.append_entity(salary, "SALARY")
    builder.append_text(" per annum, exclusive of performance stock options and long-term incentives.\n\n")
    
    builder.append_text("## 3. Personal Security and Executive Communications\n")
    builder.append_text("Confidential executive communications and legal notices shall be transmitted to ")
    builder.append_entity(email, "EMAIL")
    builder.append_text(" or verified through secure telephone line ")
    builder.append_entity(phone, "PHONE")
    builder.append_text(". Corporate security has approved the residential perimeter protocols for the executive property situated at ")
    builder.append_entity(addr, "ADDRESS")
    builder.append_text(".\n")
    
    return builder.build()

def main():
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    pool = EntityPoolGenerator(split="doc", seed=42)
    metadata_list = []
    
    # 1. Generate 30 Public Docs
    for idx in range(1, 31):
        doc = generate_public_doc(idx, pool)
        # Write .md file
        with open(docs_dir / doc["filename"], "w", encoding="utf-8") as f:
            f.write(doc["text"])
        # Write sidecar .json
        sidecar = {
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "spans": doc["spans"]
        }
        with open(docs_dir / f"{doc['doc_id']}.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        # Metadata entry
        metadata_list.append({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "entities_present": doc["entities_present"],
            "sensitive_values": doc["sensitive_values"]
        })

    # 2. Generate 30 Internal Docs
    for idx in range(1, 31):
        doc = generate_internal_doc(idx, pool)
        with open(docs_dir / doc["filename"], "w", encoding="utf-8") as f:
            f.write(doc["text"])
        sidecar = {
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "spans": doc["spans"]
        }
        with open(docs_dir / f"{doc['doc_id']}.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        metadata_list.append({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "entities_present": doc["entities_present"],
            "sensitive_values": doc["sensitive_values"]
        })

    # 3. Generate 30 HR Restricted Docs
    for idx in range(1, 31):
        doc = generate_hr_doc(idx, pool)
        with open(docs_dir / doc["filename"], "w", encoding="utf-8") as f:
            f.write(doc["text"])
        sidecar = {
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "spans": doc["spans"]
        }
        with open(docs_dir / f"{doc['doc_id']}.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        metadata_list.append({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "entities_present": doc["entities_present"],
            "sensitive_values": doc["sensitive_values"]
        })

    # 4. Generate 30 Exec Restricted Docs
    for idx in range(1, 31):
        doc = generate_exec_doc(idx, pool)
        with open(docs_dir / doc["filename"], "w", encoding="utf-8") as f:
            f.write(doc["text"])
        sidecar = {
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "spans": doc["spans"]
        }
        with open(docs_dir / f"{doc['doc_id']}.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        metadata_list.append({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "access_level": doc["access_level"],
            "entities_present": doc["entities_present"],
            "sensitive_values": doc["sensitive_values"]
        })

    # Write data/documents/metadata.json
    with open(docs_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(metadata_list)} documents with sidecar JSONs and metadata.json in {docs_dir}")

if __name__ == "__main__":
    main()
