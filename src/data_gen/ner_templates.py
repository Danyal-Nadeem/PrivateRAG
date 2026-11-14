from typing import Dict, List, Any

# Slot keys that map to sensitive entity types
SENSITIVE_ENTITY_TYPES = {
    "PERSON": "PERSON",
    "PERSON_2": "PERSON",
    "EMAIL": "EMAIL",
    "EMAIL_2": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_2": "PHONE",
    "EMP_ID": "EMP_ID",
    "EMP_ID_2": "EMP_ID",
    "ADDRESS": "ADDRESS",
    "ADDRESS_2": "ADDRESS",
    "SALARY": "SALARY",
    "SALARY_2": "SALARY"
}

# Negative slots that map to non-entity hard negatives & confusers
HARD_NEGATIVE_SLOTS = {
    "CONFUSER_COMPANY": "company",
    "CONFUSER_LOCATION": "location",
    "CONFUSER_ID": "ticket",
    "CONFUSER_CODE": "code",
    "CONFUSER_AMOUNT": "amount",
    "PRODUCT": "product",
    "DATE": "date",
    "VERSION": "version"
}

# ==============================================================================
# TRAIN TEMPLATES (Total: 105 templates = 30 zero-entity + 75 entity-bearing)
# ==============================================================================
TRAIN_TEMPLATES: List[Dict[str, Any]] = [
    # --- Zero-Entity Hard Negatives & Confusers (30 templates) ---
    {"id": "train_zero_01", "text": "Vendor invoice {CONFUSER_ID} for {CONFUSER_AMOUNT} was submitted to {CONFUSER_COMPANY} for settlement."},
    {"id": "train_zero_02", "text": "Facilities confirmed that {CONFUSER_LOCATION} approved an annual maintenance budget of {CONFUSER_AMOUNT}."},
    {"id": "train_zero_03", "text": "For questions regarding purchase order {CONFUSER_ID}, dial internal {CONFUSER_CODE}."},
    {"id": "train_zero_04", "text": "The quarterly business review at {CONFUSER_LOCATION} finalized procurement spend of {CONFUSER_AMOUNT}."},
    {"id": "train_zero_05", "text": "Shipment tracking code {CONFUSER_ID} was logged by the logistics team at {CONFUSER_LOCATION}."},
    {"id": "train_zero_06", "text": "An organization-wide audit confirmed that {CONFUSER_COMPANY} signed contract {CONFUSER_ID} on {DATE}."},
    {"id": "train_zero_07", "text": "Our team at {CONFUSER_LOCATION} closed maintenance ticket {CONFUSER_ID} after upgrading {PRODUCT} to {VERSION}."},
    {"id": "train_zero_08", "text": "Accounting ledger entries reflect an invoice total of {CONFUSER_AMOUNT} billed under {CONFUSER_ID}."},
    {"id": "train_zero_09", "text": "Notice: supply delivery {CONFUSER_ID} was received at {CONFUSER_LOCATION} on {DATE}."},
    {"id": "train_zero_10", "text": "{CONFUSER_COMPANY} reported total capital expenditure of {CONFUSER_AMOUNT} for cloud migration under {CONFUSER_ID}."},
    {"id": "train_zero_11", "text": "To report equipment failures at {CONFUSER_LOCATION}, contact the dispatch desk at {CONFUSER_CODE}."},
    {"id": "train_zero_12", "text": "The IT steering committee approved a server infrastructure grant of {CONFUSER_AMOUNT} for {CONFUSER_COMPANY}."},
    {"id": "train_zero_13", "text": "System deployment for {PRODUCT} version {VERSION} completed successfully across {CONFUSER_LOCATION}."},
    {"id": "train_zero_14", "text": "Purchase requisition {CONFUSER_ID} totaling {CONFUSER_AMOUNT} was authorized on {DATE}."},
    {"id": "train_zero_15", "text": "Customer service agents at {CONFUSER_LOCATION} should reference work order {CONFUSER_ID}."},
    {"id": "train_zero_16", "text": "An internal advisory memo from {CONFUSER_COMPANY} recommended allocating {CONFUSER_AMOUNT} toward security."},
    {"id": "train_zero_17", "text": "The facility expansion project at {CONFUSER_LOCATION} was cataloged under reference code {CONFUSER_ID}."},
    {"id": "train_zero_18", "text": "Inquiries regarding billing statement {CONFUSER_ID} should be routed to extension {CONFUSER_CODE}."},
    {"id": "train_zero_19", "text": "Release notes for {PRODUCT} {VERSION} were archived on {DATE} by the technical documentation team."},
    {"id": "train_zero_20", "text": "{CONFUSER_COMPANY} reported quarterly gross revenue of {CONFUSER_AMOUNT} during the investor briefing."},
    {"id": "train_zero_21", "text": "Network maintenance scheduled for {CONFUSER_LOCATION} on {DATE} will be tracked via {CONFUSER_ID}."},
    {"id": "train_zero_22", "text": "A departmental software license fee of {CONFUSER_AMOUNT} was charged under account {CONFUSER_ID}."},
    {"id": "train_zero_23", "text": "Conference room booking for the strategy workshop at {CONFUSER_LOCATION} is confirmed under {CONFUSER_ID}."},
    {"id": "train_zero_24", "text": "Audit report {CONFUSER_ID} confirmed that {CONFUSER_COMPANY} completed ISO compliance ahead of {DATE}."},
    {"id": "train_zero_25", "text": "Please direct all inquiries regarding shipping batch {CONFUSER_ID} to telephone {CONFUSER_CODE}."},
    {"id": "train_zero_26", "text": "The hardware refresh budget of {CONFUSER_AMOUNT} was approved for deployment at {CONFUSER_LOCATION}."},
    {"id": "train_zero_27", "text": "{CONFUSER_COMPANY} established a partnership agreement valued at {CONFUSER_AMOUNT} on {DATE}."},
    {"id": "train_zero_28", "text": "Incident report {CONFUSER_ID} concerning database latency at {CONFUSER_LOCATION} was resolved on {DATE}."},
    {"id": "train_zero_29", "text": "Engineering teams updated {PRODUCT} to {VERSION} across all nodes under change ticket {CONFUSER_ID}."},
    {"id": "train_zero_30", "text": "The executive committee allocated an operational reserve of {CONFUSER_AMOUNT} for {CONFUSER_COMPANY}."},

    # --- Entity-Bearing Templates (75 templates) ---
    # Group A: Salary & Compensation (15 templates)
    {"id": "train_ent_01", "text": "The annual base compensation for {PERSON} ({EMP_ID}) is set at {SALARY}."},
    {"id": "train_ent_02", "text": "Employee {PERSON} holds employee badge {EMP_ID} and currently receives {SALARY} per annum."},
    {"id": "train_ent_03", "text": "Approved payroll records indicate that {PERSON} has a base salary of {SALARY}."},
    {"id": "train_ent_04", "text": "The revised compensation package for {EMP_ID} stipulates an annual rate of {SALARY}."},
    {"id": "train_ent_05", "text": "HR recorded that {PERSON} was offered {SALARY} upon joining the department."},
    {"id": "train_ent_06", "text": "According to the payroll schedule, {EMP_ID} is remunerated at {SALARY}."},
    {"id": "train_ent_07", "text": "Total baseline earnings for {PERSON} are listed as {SALARY} in the finance ledger."},
    {"id": "train_ent_08", "text": "The salary adjustment memo for {PERSON} confirms a new rate of {SALARY}."},
    {"id": "train_ent_09", "text": "Staff member {EMP_ID} receives a gross annual compensation of {SALARY}."},
    {"id": "train_ent_10", "text": "Compensation committee records show {PERSON} ({EMP_ID}) earning {SALARY} annually."},
    {"id": "train_ent_11", "text": "Payroll processed a revised figure of {SALARY} for employee {EMP_ID}."},
    {"id": "train_ent_12", "text": "The hiring contract for {PERSON} specifies a starting salary of {SALARY}."},
    {"id": "train_ent_13", "text": "Internal compensation audit logs show {EMP_ID} receiving {SALARY}."},
    {"id": "train_ent_14", "text": "Finance authorized an annual payment of {SALARY} for {PERSON}."},
    {"id": "train_ent_15", "text": "The baseline pay band for {EMP_ID} corresponds to {SALARY}."},

    # Group B: Contacts (Email & Phone) (15 templates)
    {"id": "train_ent_16", "text": "You can reach {PERSON} via email at {EMAIL} or call {PHONE}."},
    {"id": "train_ent_17", "text": "Official correspondence for {PERSON} should be directed to {EMAIL} or phone {PHONE}."},
    {"id": "train_ent_18", "text": "Please contact {EMP_ID} at {EMAIL} or dial {PHONE} regarding this matter."},
    {"id": "train_ent_19", "text": "Direct telephone inquiries for {PERSON} should be routed to {PHONE} with copies to {EMAIL}."},
    {"id": "train_ent_20", "text": "In case of an urgent escalation, call {PHONE} to contact {PERSON} at {EMAIL}."},
    {"id": "train_ent_21", "text": "Employee {EMP_ID} can be reached during office hours at {PHONE} or by emailing {EMAIL}."},
    {"id": "train_ent_22", "text": "Primary contact for {PERSON}: email {EMAIL} and telephone {PHONE}."},
    {"id": "train_ent_23", "text": "Submit the signed verification to {PERSON} at {EMAIL} or call {PHONE}."},
    {"id": "train_ent_24", "text": "For phone verification, dial {PHONE} or email {EMAIL} to reach {EMP_ID}."},
    {"id": "train_ent_25", "text": "Contact details for {PERSON} ({EMP_ID}): email {EMAIL}, phone {PHONE}."},
    {"id": "train_ent_26", "text": "Please forward the project update to {EMAIL} for {PERSON} or notify via {PHONE}."},
    {"id": "train_ent_27", "text": "The emergency contact phone number on file for {PERSON} is {PHONE} with email {EMAIL}."},
    {"id": "train_ent_28", "text": "To schedule a meeting with {EMP_ID}, email {EMAIL} directly or call {PHONE}."},
    {"id": "train_ent_29", "text": "Our records list {EMAIL} and phone {PHONE} as verified contacts for {PERSON}."},
    {"id": "train_ent_30", "text": "Call {PHONE} or send an inquiry to {EMAIL} to reach staff member {PERSON}."},

    # Group C: Residential Address (15 templates)
    {"id": "train_ent_31", "text": "The residential address on file for {PERSON} is {ADDRESS}."},
    {"id": "train_ent_32", "text": "Tax forms for {EMP_ID} should be mailed to {ADDRESS}."},
    {"id": "train_ent_33", "text": "Employee {PERSON} ({EMP_ID}) currently resides at {ADDRESS}."},
    {"id": "train_ent_34", "text": "Physical mailing address recorded for {PERSON} is {ADDRESS}."},
    {"id": "train_ent_35", "text": "HR dispatched the equipment package to {PERSON} at {ADDRESS}."},
    {"id": "train_ent_36", "text": "The verified primary residence for {EMP_ID} is {ADDRESS}."},
    {"id": "train_ent_37", "text": "Please confirm if {PERSON} still lives at {ADDRESS}."},
    {"id": "train_ent_38", "text": "Official relocation documents state that {PERSON} relocated to {ADDRESS}."},
    {"id": "train_ent_39", "text": "The home billing address for employee {EMP_ID} is {ADDRESS}."},
    {"id": "train_ent_40", "text": "Correspondence sent to {PERSON} at {ADDRESS} was acknowledged."},
    {"id": "train_ent_41", "text": "Benefits enrollment documents for {PERSON} list {ADDRESS} as the domicile."},
    {"id": "train_ent_42", "text": "Deliver the notarized agreement for {EMP_ID} directly to {ADDRESS}."},
    {"id": "train_ent_43", "text": "The payroll system registered a change of address for {PERSON} to {ADDRESS}."},
    {"id": "train_ent_44", "text": "Employee residence record: {PERSON} lives at {ADDRESS}."},
    {"id": "train_ent_45", "text": "Courier tracking shows delivery for {EMP_ID} arrived at {ADDRESS}."},

    # Group D: Multi-Entity Combinations (15 templates)
    {"id": "train_ent_46", "text": "Employee record: {PERSON}, ID {EMP_ID}, email {EMAIL}, compensation {SALARY}."},
    {"id": "train_ent_47", "text": "{PERSON} ({EMP_ID}) living at {ADDRESS} receives an annual base of {SALARY}."},
    {"id": "train_ent_48", "text": "Please update records for {PERSON}: phone {PHONE}, address {ADDRESS}, salary {SALARY}."},
    {"id": "train_ent_49", "text": "HR file for {EMP_ID}: contact {EMAIL}, mobile {PHONE}, remuneration {SALARY}."},
    {"id": "train_ent_50", "text": "Send the compensation breakdown of {SALARY} for {PERSON} ({EMP_ID}) to {EMAIL}."},
    {"id": "train_ent_51", "text": "Staff profile: {PERSON}, registered at {ADDRESS}, reachable at {PHONE} and email {EMAIL}."},
    {"id": "train_ent_52", "text": "Direct deposit of {SALARY} was verified for {PERSON} residing at {ADDRESS}."},
    {"id": "train_ent_53", "text": "Verification file: {EMP_ID} belonging to {PERSON}, reachable by phone at {PHONE} or email {EMAIL}."},
    {"id": "train_ent_54", "text": "The offer for {PERSON} includes {SALARY} with notices sent to {ADDRESS}."},
    {"id": "train_ent_55", "text": "Audit summary: employee {EMP_ID} ({PERSON}) confirmed current phone as {PHONE} and email {EMAIL}."},
    {"id": "train_ent_56", "text": "Notify {PERSON} at {EMAIL} that their revised compensation is {SALARY}."},
    {"id": "train_ent_57", "text": "Employee {PERSON} ({EMP_ID}) submitted residence proof for {ADDRESS} and phone {PHONE}."},
    {"id": "train_ent_58", "text": "Contact {PERSON} at {EMAIL} or visit the registered residence at {ADDRESS}."},
    {"id": "train_ent_59", "text": "Payroll confirmed that {EMP_ID} earns {SALARY} and can be emailed at {EMAIL}."},
    {"id": "train_ent_60", "text": "Emergency notification: call {PHONE} or write to {EMAIL} for employee {PERSON}."},

    # Group E: Operational / HR Contexts (15 templates)
    {"id": "train_ent_61", "text": "The annual compensation review for {PERSON} ({EMP_ID}) approved {SALARY} and address {ADDRESS}."},
    {"id": "train_ent_62", "text": "Records indicate {PERSON} lives at {ADDRESS} with secondary contact {PHONE} and email {EMAIL}."},
    {"id": "train_ent_63", "text": "Personnel dossier for {EMP_ID}: name {PERSON}, email {EMAIL}, compensation {SALARY}."},
    {"id": "train_ent_64", "text": "Forward the salary certificate stating {SALARY} to {PERSON} at {EMAIL} or call {PHONE}."},
    {"id": "train_ent_65", "text": "Registered address for {EMP_ID} is {ADDRESS}, with primary phone {PHONE} and email {EMAIL}."},
    {"id": "train_ent_66", "text": "Supervisor notes indicate {PERSON} was assigned to project lead under badge {EMP_ID} with phone {PHONE}."},
    {"id": "train_ent_67", "text": "Disciplinary review committee forwarded documentation to {PERSON} at {EMAIL} and phone {PHONE}."},
    {"id": "train_ent_68", "text": "The background investigation for {PERSON} verified residence at {ADDRESS} and contact {PHONE}."},
    {"id": "train_ent_69", "text": "Performance appraisal for {EMP_ID} recommends an increment reaching {SALARY} and notified at {EMAIL}."},
    {"id": "train_ent_70", "text": "Compliance officer attempted contact with {PERSON} via phone {PHONE} and email {EMAIL}."},
    {"id": "train_ent_71", "text": "Onboarding coordinator emailed welcome instructions to {PERSON} at {EMAIL} and phoned {PHONE}."},
    {"id": "train_ent_72", "text": "Corporate security issued an access badge to {PERSON} under identifier {EMP_ID} and address {ADDRESS}."},
    {"id": "train_ent_73", "text": "Official payroll transfer of {SALARY} was credited to employee {EMP_ID} residing at {ADDRESS}."},
    {"id": "train_ent_74", "text": "Mailing verification completed for {PERSON} at the address {ADDRESS} with phone {PHONE}."},
    {"id": "train_ent_75", "text": "The formal offer letter sent to {EMAIL} guaranteed a salary of {SALARY} for {PERSON}."}
]


# ==============================================================================
# VALIDATION TEMPLATES (Total: 22 templates = 8 zero-entity + 14 entity-bearing)
# ==============================================================================
VAL_TEMPLATES: List[Dict[str, Any]] = [
    # --- Zero-Entity Hard Negatives & Confusers (8 templates) ---
    {"id": "val_zero_01", "text": "Has the procurement office for {CONFUSER_COMPANY} approved purchase invoice {CONFUSER_ID}?"},
    {"id": "val_zero_02", "text": "According to fiscal filings, the annual operating budget for {CONFUSER_LOCATION} was established at {CONFUSER_AMOUNT}."},
    {"id": "val_zero_03", "text": "Can we confirm whether maintenance request {CONFUSER_ID} logged on {DATE} was completed at {CONFUSER_LOCATION}?"},
    {"id": "val_zero_04", "text": "Please direct inquiries regarding supplier agreement {CONFUSER_ID} to internal {CONFUSER_CODE}."},
    {"id": "val_zero_05", "text": "Audit findings indicate that {CONFUSER_COMPANY} completed security validation of {PRODUCT} version {VERSION}."},
    {"id": "val_zero_06", "text": "Did the finance team at {CONFUSER_LOCATION} authorize the equipment expenditure of {CONFUSER_AMOUNT} on {DATE}?"},
    {"id": "val_zero_07", "text": "System alert {CONFUSER_ID} tracking database replication across {CONFUSER_LOCATION} was successfully archived."},
    {"id": "val_zero_08", "text": "The regional conference sponsored by {CONFUSER_COMPANY} reported total registration revenue of {CONFUSER_AMOUNT}."},

    # --- Entity-Bearing Templates (14 templates) ---
    {"id": "val_ent_01", "text": "Could you verify if {PERSON} is currently receiving an annual salary of {SALARY}?"},
    {"id": "val_ent_02", "text": "Has the residential documentation for {EMP_ID} at {ADDRESS} been certified with phone {PHONE}?"},
    {"id": "val_ent_03", "text": "Did the recruiter send the onboarding package to {PERSON} at {EMAIL} or phone {PHONE}?"},
    {"id": "val_ent_04", "text": "Was an escalation voicemail left for {PERSON} at phone number {PHONE} and email {EMAIL}?"},
    {"id": "val_ent_05", "text": "Is staff member {PERSON} still identified under employee code {EMP_ID} and address {ADDRESS}?"},
    {"id": "val_ent_06", "text": "The compliance audit raised questions regarding {SALARY} paid to {EMP_ID} with phone {PHONE}."},
    {"id": "val_ent_07", "text": "Legal notices were forwarded to {PERSON} residing at the address {ADDRESS} with email {EMAIL}."},
    {"id": "val_ent_08", "text": "Please confirm whether {EMAIL} remains the primary contact for {PERSON} ({EMP_ID}) with phone {PHONE}."},
    {"id": "val_ent_09", "text": "A verification SMS was successfully routed to {PERSON} at {PHONE} and email {EMAIL}."},
    {"id": "val_ent_10", "text": "The executive board validated that {PERSON} was assigned a salary of {SALARY} at address {ADDRESS}."},
    {"id": "val_ent_11", "text": "Dispute resolution files document the residence of {EMP_ID} as {ADDRESS} and phone {PHONE}."},
    {"id": "val_ent_12", "text": "Does our directory list {EMAIL} or telephone {PHONE} for reaching employee {PERSON} ({EMP_ID})?"},
    {"id": "val_ent_13", "text": "The contract revision for {PERSON} ({EMP_ID}) stipulates base compensation of {SALARY} and address {ADDRESS}."},
    {"id": "val_ent_14", "text": "Candidate {PERSON} accepted an employment offer of {SALARY} sent to {EMAIL} and phoned at {PHONE}."}
]


# ==============================================================================
# TEST TEMPLATES (Total: 23 templates = 8 zero-entity + 15 entity-bearing)
# ==============================================================================
TEST_TEMPLATES: List[Dict[str, Any]] = [
    # --- Zero-Entity Hard Negatives & Confusers (8 templates) ---
    {"id": "test_zero_01", "text": "Executive audit logs show that {CONFUSER_COMPANY} authorized a facility renovation grant of {CONFUSER_AMOUNT}."},
    {"id": "test_zero_02", "text": "Incident report {CONFUSER_ID} confirms that {PRODUCT} update {VERSION} went live at {CONFUSER_LOCATION} on {DATE}."},
    {"id": "test_zero_03", "text": "Disaster recovery protocols for the branch network at {CONFUSER_LOCATION} were validated under ticket {CONFUSER_ID}."},
    {"id": "test_zero_04", "text": "Billing records confirm that commercial invoice {CONFUSER_ID} totaling {CONFUSER_AMOUNT} was cleared on {DATE}."},
    {"id": "test_zero_05", "text": "For support inquiries regarding operational batch {CONFUSER_ID}, dial internal {CONFUSER_CODE}."},
    {"id": "test_zero_06", "text": "The quarterly business symposium hosted by {CONFUSER_COMPANY} reported total attendee receipts of {CONFUSER_AMOUNT}."},
    {"id": "test_zero_07", "text": "Technical inspection at {CONFUSER_LOCATION} verified that equipment order {CONFUSER_ID} arrived without defect."},
    {"id": "test_zero_08", "text": "Corporate accounting registered an infrastructure research allocation of {CONFUSER_AMOUNT} for {CONFUSER_COMPANY}."},

    # --- Entity-Bearing Templates (15 templates) ---
    {"id": "test_ent_01", "text": "Confidential compensation statement: {PERSON} is remunerated at an agreed rate of {SALARY} and address {ADDRESS}."},
    {"id": "test_ent_02", "text": "Physical inspection records certify that employee {EMP_ID} dwells at {ADDRESS} with phone {PHONE}."},
    {"id": "test_ent_03", "text": "Electronic notification concerning the appraisal was dispatched to {PERSON} at {EMAIL} with telephone {PHONE}."},
    {"id": "test_ent_04", "text": "Direct telephonic communication with {PERSON} should proceed via {PHONE} or email {EMAIL}."},
    {"id": "test_ent_05", "text": "Corporate identity badge {EMP_ID} belongs to senior associate {PERSON} residing at {ADDRESS}."},
    {"id": "test_ent_06", "text": "Remuneration packages totaling {SALARY} were officially assigned to {EMP_ID} with email {EMAIL}."},
    {"id": "test_ent_07", "text": "Notarized real estate filings indicate {PERSON} maintains primary residence at {ADDRESS} with mobile {PHONE}."},
    {"id": "test_ent_08", "text": "Direct inquiries for {PERSON} ({EMP_ID}) can be emailed to {EMAIL} or called at {PHONE}."},
    {"id": "test_ent_09", "text": "Telecom verification confirms the active line for {PERSON} is {PHONE} and email is {EMAIL}."},
    {"id": "test_ent_10", "text": "The compensation panel ratified a baseline salary of {SALARY} for {PERSON} living at {ADDRESS}."},
    {"id": "test_ent_11", "text": "Official correspondence was addressed to {EMP_ID} at the residence {ADDRESS} with phone {PHONE}."},
    {"id": "test_ent_12", "text": "Secure correspondence to {PERSON} should be transmitted to {EMAIL} or dialed at {PHONE}."},
    {"id": "test_ent_13", "text": "The formal payroll agreement for {PERSON} ({EMP_ID}) specifies earnings of {SALARY} and address {ADDRESS}."},
    {"id": "test_ent_14", "text": "Registry records certify that {PERSON} currently inhabits {ADDRESS} and holds phone {PHONE} and email {EMAIL}."},
    {"id": "test_ent_15", "text": "The confidential appointment memo granted {PERSON} ({EMP_ID}) a starting package of {SALARY} sent to {EMAIL}."}
]
