import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.tokenizer import tokenize_with_offsets, align_tokens_and_bio

def create_entry(example_id: str, text: str, raw_entities: list) -> dict:
    entities = []
    for val, label in raw_entities:
        val_clean = val.strip()
        start = text.find(val_clean)
        if start == -1:
            raise ValueError(f"Entity '{val_clean}' not found in text: {text}")
        end = start + len(val_clean)
        entities.append({
            "start": start,
            "end": end,
            "label": label,
            "text": val_clean
        })
    entities.sort(key=lambda x: x["start"])
    tokens_with_offsets = tokenize_with_offsets(text)
    tokens = [t[0] for t in tokens_with_offsets]
    ner_tags = align_tokens_and_bio(text, entities, tokens_with_offsets)
    return {
        "id": example_id,
        "text": text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "entities": entities
    }

def main():
    out_dir = Path("data/ner")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        # --- 10 Zero-Entity Negative Sentences ---
        create_entry("realistic_001", "Hello! The CloudCore v3.2 release notes have been posted to the developer portal as of October 14, 2026.", []),
        create_entry("realistic_002", "Please note that the conference room in our Seattle office has been reserved for the quarterly engineering sync.", []),
        create_entry("realistic_003", "I checked the system logs and ticket TKT-48213 was resolved yesterday after deploying patch v1.8.", []),
        create_entry("realistic_004", "The total Q3 marketing budget of $125,000 was approved by the executive committee during yesterday's meeting.", []),
        create_entry("realistic_005", "Can you confirm if invoice INV-88124 for PKR 450,000 has been reviewed by the accounting department?", []),
        create_entry("realistic_006", "All team members are encouraged to review the updated corporate security and VPN guidelines in the handbook.", []),
        create_entry("realistic_007", "The hardware shipment tracked under ORD-90142 arrived at the Chicago distribution hub earlier this morning.", []),
        create_entry("realistic_008", "We conducted a routine audit of our European database clusters and found zero critical latency anomalies.", []),
        create_entry("realistic_009", "For technical inquiries regarding the new API endpoints, please dial internal extension ext. 4821.", []),
        create_entry("realistic_010", "The executive steering committee meeting scheduled for November 2026 will cover enterprise expansion plans.", []),

        # --- 50 Conversational / Email Entity Sentences ---
        create_entry("realistic_011", "Certainly! According to our directory, Cedric Vance can be contacted at cedric.vance@realistic-test.org or +1-808-555-0142.", [
            ("Cedric Vance", "PERSON"), ("cedric.vance@realistic-test.org", "EMAIL"), ("+1-808-555-0142", "PHONE")
        ]),
        create_entry("realistic_012", "I found the profile for employee EMP-91024: the verified home address on file is 412 Pinecrest Lane, Denver, CO 80201.", [
            ("EMP-91024", "EMP_ID"), ("412 Pinecrest Lane, Denver, CO 80201", "ADDRESS")
        ]),
        create_entry("realistic_013", "Regarding your compensation question, Arham Bokhari currently receives an annual base salary of $135,000.", [
            ("Arham Bokhari", "PERSON"), ("$135,000", "SALARY")
        ]),
        create_entry("realistic_014", "Hello, please forward the signed NDA to Fiona Sinclair via email at fiona.sinclair@realistic-test.org.", [
            ("Fiona Sinclair", "PERSON"), ("fiona.sinclair@realistic-test.org", "EMAIL")
        ]),
        create_entry("realistic_015", "HR records confirm that staff member Rayan Qidwai (badge EMP-92104) is based at House 14, Sector C, Sabzazar, Lahore, 54770.", [
            ("Rayan Qidwai", "PERSON"), ("EMP-92104", "EMP_ID"), ("House 14, Sector C, Sabzazar, Lahore, 54770", "ADDRESS")
        ]),
        create_entry("realistic_016", "You can reach out to our senior systems architect Gareth Mercer directly at +1-808-555-0198.", [
            ("Gareth Mercer", "PERSON"), ("+1-808-555-0198", "PHONE")
        ]),
        create_entry("realistic_017", "The formal offer letter specifies that Clara Hawthorne will draw a starting salary of USD 115,000 per annum.", [
            ("Clara Hawthorne", "PERSON"), ("USD 115,000", "SALARY")
        ]),
        create_entry("realistic_018", "Please update the shipping destination for employee E-93401 to 780 Hawthorn Way, Austin, TX 78701.", [
            ("E-93401", "EMP_ID"), ("780 Hawthorn Way, Austin, TX 78701", "ADDRESS")
        ]),
        create_entry("realistic_019", "In case of an after-hours emergency, dial +92 345 1234567 to reach on-call engineer Zohaib Soofi.", [
            ("+92 345 1234567", "PHONE"), ("Zohaib Soofi", "PERSON")
        ]),
        create_entry("realistic_020", "Our payroll database lists an approved remuneration of PKR 750,000 for associate Insharah Kundi.", [
            ("PKR 750,000", "SALARY"), ("Insharah Kundi", "PERSON")
        ]),
        create_entry("realistic_021", "Hi team, please ensure that Liam Kingsley receives the onboarding welcome pack at liam.kingsley@realistic-test.org.", [
            ("Liam Kingsley", "PERSON"), ("liam.kingsley@realistic-test.org", "EMAIL")
        ]),
        create_entry("realistic_022", "The background check report verifies that Nora Lockwood resides at 1045 Foxglove Drive, Seattle, WA 98101.", [
            ("Nora Lockwood", "PERSON"), ("1045 Foxglove Drive, Seattle, WA 98101", "ADDRESS")
        ]),
        create_entry("realistic_023", "I verified with payroll that staff member EMP-94102 has an authorized annual compensation of $148,000.", [
            ("EMP-94102", "EMP_ID"), ("$148,000", "SALARY")
        ]),
        create_entry("realistic_024", "To confirm the interview slot with candidate Subhan Ghauri, please send an invitation to subhan.ghauri@realistic-test.com.", [
            ("Subhan Ghauri", "PERSON"), ("subhan.ghauri@realistic-test.com", "EMAIL")
        ]),
        create_entry("realistic_025", "Official records state that the primary residence for Ronan Mercer is Flat 3B, Block 2, Jubilee Town, Lahore, 54770.", [
            ("Ronan Mercer", "PERSON"), ("Flat 3B, Block 2, Jubilee Town, Lahore, 54770", "ADDRESS")
        ]),
        create_entry("realistic_026", "The contact phone number registered for employee E-95201 is +1-808-555-0231.", [
            ("E-95201", "EMP_ID"), ("+1-808-555-0231", "PHONE")
        ]),
        create_entry("realistic_027", "Following the performance appraisal, Ehsan Sadozai was awarded a revised base salary of Rs. 1,400,000.", [
            ("Ehsan Sadozai", "PERSON"), ("Rs. 1,400,000", "SALARY")
        ]),
        create_entry("realistic_028", "Could you verify if Chloe Vance has updated her primary email address to chloe.vance@realistic-test.org?", [
            ("Chloe Vance", "PERSON"), ("chloe.vance@realistic-test.org", "EMAIL")
        ]),
        create_entry("realistic_029", "Courier tracking shows the laptop parcel was delivered to 320 Briarwood Boulevard, Atlanta, GA 30301 for Gideon Sinclair.", [
            ("320 Briarwood Boulevard, Atlanta, GA 30301", "ADDRESS"), ("Gideon Sinclair", "PERSON")
        ]),
        create_entry("realistic_030", "For escalation inquiries, associate Dananeer Dasti can be dialed at mobile line +92 345 7654321.", [
            ("Dananeer Dasti", "PERSON"), ("+92 345 7654321", "PHONE")
        ]),
        create_entry("realistic_031", "The executive resolution confirms a baseline compensation of $210,000 for officer Serena Blackwood.", [
            ("$210,000", "SALARY"), ("Serena Blackwood", "PERSON")
        ]),
        create_entry("realistic_032", "The personnel profile for EMP-96301 lists the employee name as Arman Alizai.", [
            ("EMP-96301", "EMP_ID"), ("Arman Alizai", "PERSON")
        ]),
        create_entry("realistic_033", "Please dispatch the benefits confirmation letter to 512 Windsor Avenue, Chicago, IL 60601.", [
            ("512 Windsor Avenue, Chicago, IL 60601", "ADDRESS")
        ]),
        create_entry("realistic_034", "Here are the requested details: Shanzay Bokhari can be emailed at shanzay.bokhari@realistic-test.com or called at +92 345 9876543.", [
            ("Shanzay Bokhari", "PERSON"), ("shanzay.bokhari@realistic-test.com", "EMAIL"), ("+92 345 9876543", "PHONE")
        ]),
        create_entry("realistic_035", "HR confirmed that the annual salary for employee EMP-97401 has been adjusted to PKR 920,000.", [
            ("EMP-97401", "EMP_ID"), ("PKR 920,000", "SALARY")
        ]),
        create_entry("realistic_036", "The residential address provided by Iraj Soofi during onboarding is House 88, Street 4, Gulshan Ravi, Lahore, 54000.", [
            ("Iraj Soofi", "PERSON"), ("House 88, Street 4, Gulshan Ravi, Lahore, 54000", "ADDRESS")
        ]),
        create_entry("realistic_037", "If you need immediate assistance from Cedric Vance, please dial direct line +1-808-555-0312.", [
            ("Cedric Vance", "PERSON"), ("+1-808-555-0312", "PHONE")
        ]),
        create_entry("realistic_038", "A compensation statement indicating an annual rate of $122,000 was generated for Clara Hawthorne.", [
            ("$122,000", "SALARY"), ("Clara Hawthorne", "PERSON")
        ]),
        create_entry("realistic_039", "Please verify that the employee code for Fiona Sinclair is recorded as EMP98102.", [
            ("Fiona Sinclair", "PERSON"), ("EMP98102", "EMP_ID")
        ]),
        create_entry("realistic_040", "The domicile certificate for Rayan Qidwai verifies current occupancy at Plot 12, Model Colony, Lahore, 54700.", [
            ("Rayan Qidwai", "PERSON"), ("Plot 12, Model Colony, Lahore, 54700", "ADDRESS")
        ]),
        create_entry("realistic_041", "You may submit the tax declaration documents directly to arham.bokhari@realistic-test.com.", [
            ("arham.bokhari@realistic-test.com", "EMAIL")
        ]),
        create_entry("realistic_042", "According to the department roster, employee E-98501 receives an annual remuneration of $95k.", [
            ("E-98501", "EMP_ID"), ("$95k", "SALARY")
        ]),
        create_entry("realistic_043", "An urgent SMS dispatch was sent to Gareth Mercer at cell number +1-808-555-0456.", [
            ("Gareth Mercer", "PERSON"), ("+1-808-555-0456", "PHONE")
        ]),
        create_entry("realistic_044", "The relocation reimbursement package was delivered to Nora Lockwood at 650 Crestview Road, Dallas, TX 75201.", [
            ("Nora Lockwood", "PERSON"), ("650 Crestview Road, Dallas, TX 75201", "ADDRESS")
        ]),
        create_entry("realistic_045", "Corporate security issued temporary credentials for Insharah Kundi under badge EMP-99104.", [
            ("Insharah Kundi", "PERSON"), ("EMP-99104", "EMP_ID")
        ]),
        create_entry("realistic_046", "Approved salary ledger records confirm that Liam Kingsley is remunerated at EUR 85,000.", [
            ("Liam Kingsley", "PERSON"), ("EUR 85,000", "SALARY")
        ]),
        create_entry("realistic_047", "For administrative inquiries regarding Subhan Ghauri, write to subhan.ghauri@realistic-test.org.", [
            ("Subhan Ghauri", "PERSON"), ("subhan.ghauri@realistic-test.org", "EMAIL")
        ]),
        create_entry("realistic_048", "The verified mobile contact for employee EMP-99201 is +92 345 3344556.", [
            ("EMP-99201", "EMP_ID"), ("+92 345 3344556", "PHONE")
        ]),
        create_entry("realistic_049", "Residential documents confirm that Zohaib Soofi resides at House 29, Street 1, Samanabad Extension, Lahore, 54500.", [
            ("Zohaib Soofi", "PERSON"), ("House 29, Street 1, Samanabad Extension, Lahore, 54500", "ADDRESS")
        ]),
        create_entry("realistic_050", "The revised appointment letter establishes an annual package of PKR 1,150,000 for Ehsan Sadozai.", [
            ("PKR 1,150,000", "SALARY"), ("Ehsan Sadozai", "PERSON")
        ]),
        create_entry("realistic_051", "Please confirm whether Chloe Vance (EMP-99301) can be reached at chloe.vance@realistic-test.org.", [
            ("Chloe Vance", "PERSON"), ("EMP-99301", "EMP_ID"), ("chloe.vance@realistic-test.org", "EMAIL")
        ]),
        create_entry("realistic_052", "The equipment dispatch was routed to 890 Bramble Lane, Phoenix, AZ 85001 for Gideon Sinclair.", [
            ("890 Bramble Lane, Phoenix, AZ 85001", "ADDRESS"), ("Gideon Sinclair", "PERSON")
        ]),
        create_entry("realistic_053", "Staff member Serena Blackwood holds employee identifier E-99401 and phone +1-808-555-0589.", [
            ("Serena Blackwood", "PERSON"), ("E-99401", "EMP_ID"), ("+1-808-555-0589", "PHONE")
        ]),
        create_entry("realistic_054", "Finance department authorized a salary disbursement of $165,000 for associate Ronan Mercer.", [
            ("$165,000", "SALARY"), ("Ronan Mercer", "PERSON")
        ]),
        create_entry("realistic_055", "Emergency contact records list telephone +92 345 8899001 for Arman Alizai.", [
            ("+92 345 8899001", "PHONE"), ("Arman Alizai", "PERSON")
        ]),
        create_entry("realistic_056", "The primary email registered for Dananeer Dasti is dananeer.dasti@realistic-test.com.", [
            ("Dananeer Dasti", "PERSON"), ("dananeer.dasti@realistic-test.com", "EMAIL")
        ]),
        create_entry("realistic_057", "Employee Shanzay Bokhari residing at House 5, Street 12, Shadbagh, Lahore, 54000 draws a salary of Rs. 1,650,000.", [
            ("Shanzay Bokhari", "PERSON"), ("House 5, Street 12, Shadbagh, Lahore, 54000", "ADDRESS"), ("Rs. 1,650,000", "SALARY")
        ]),
        create_entry("realistic_058", "Please note that Cedric Vance can be reached at office address 120 Thornfield Street, Boston, MA 02108.", [
            ("Cedric Vance", "PERSON"), ("120 Thornfield Street, Boston, MA 02108", "ADDRESS")
        ]),
        create_entry("realistic_059", "The annual compensation review for Iraj Soofi approved an updated salary of PKR 580,000.", [
            ("Iraj Soofi", "PERSON"), ("PKR 580,000", "SALARY")
        ]),
        create_entry("realistic_060", "To finalize badge EMP-99501, submit the signed verification for Clara Hawthorne to clara.hawthorne@realistic-test.org.", [
            ("EMP-99501", "EMP_ID"), ("Clara Hawthorne", "PERSON"), ("clara.hawthorne@realistic-test.org", "EMAIL")
        ])
    ]

    out_file = out_dir / "test_realistic.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generated {len(records)} realistic evaluation examples in {out_file}")

if __name__ == "__main__":
    main()
