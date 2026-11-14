import random
from typing import Dict, List, Tuple, Optional

# ==========================================
# 1. PERSON NAME POOLS (Partitioned 70/15/15)
# ==========================================

# Western First Names
WESTERN_FIRST_TRAIN = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Emma", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Nicole", "Scott", "Anna", "Brandon", "Samantha",
    "Benjamin", "Katherine", "Samuel", "Christine", "Gregory", "Debra", "Alexander", "Rachel",
    "Patrick", "Carolyn", "Frank", "Janet", "Raymond", "Maria", "Jack", "Heather",
    "Dennis", "Diane", "Jerry", "Julie", "Tyler", "Joyce", "Aaron", "Victoria",
    "Jose", "Kelly", "Adam", "Christina", "Nathan", "Lauren", "Henry", "Joan",
    "Douglas", "Evelyn", "Zachary", "Olivia", "Peter", "Judith", "Kyle", "Megan"
]

WESTERN_FIRST_VAL = [
    "Ethan", "Cheryl", "Walter", "Andrea", "Noah", "Hannah", "Jeremy", "Martha",
    "Christian", "Jacqueline", "Keith", "Frances", "Roger", "Ann", "Terry", "Gloria",
    "Gerald", "Jean", "Harold", "Kathryn", "Sean", "Alice", "Austin", "Teresa",
    "Carl", "Sara", "Arthur", "Janice", "Lawrence", "Doris"
]

WESTERN_FIRST_TEST = [
    "Dylan", "Madison", "Jesse", "Julia", "Jordan", "Grace", "Bryan", "Judy",
    "Billy", "Abigail", "Joe", "Marie", "Bruce", "Denise", "Gabriel", "Beverly",
    "Logan", "Amber", "Albert", "Theresa", "Alan", "Marilyn", "Juan", "Danielle",
    "Wayne", "Diana", "Elijah", "Brittany", "Randy", "Natalie"
]

# Pakistani First Names
PAKISTANI_FIRST_TRAIN = [
    "Muhammad", "Fatima", "Ahmad", "Ayesha", "Ali", "Zainab", "Usman", "Maryam",
    "Bilal", "Sana", "Hamza", "Hira", "Omer", "Noor", "Hassan", "Anam",
    "Hussein", "Rabia", "Tariq", "Amna", "Zubair", "Iqra", "Imran", "Sadia",
    "Farhan", "Mahnoor", "Kamran", "Mehwish", "Saad", "Khadija", "Asad", "Sidra",
    "Waqas", "Nida", "Adnan", "Alina", "Kashif", "Bushra", "Rehan", "Sobia",
    "Shahid", "Kiran", "Faisal", "Faiza", "Babar", "Laiba", "Junaid", "Sundas",
    "Shoaib", "Nazia", "Arslan", "Madiha", "Zeeshan", "Samina", "Rizwan", "Farah",
    "Nasir", "Shazia", "Adeel", "Saima", "Haris", "Tahira", "Noman", "Uzma",
    "Fahad", "Sumera", "Danish", "Rubab", "Shehroz", "Rida", "Taimoor", "Areeba",
    "Umair", "Komal", "Waleed", "Hafsa", "Zahid", "Nimra", "Naveed", "Sahar",
    "Atif", "Mehak", "Rashid", "Fariha", "Jamil", "Afshan", "Sohail", "Fozia",
    "Tahir", "Asma", "Irfan", "Amber", "Qasim", "Maria", "Shakeel", "Beenish",
    "Muneeb", "Zunaira", "Ahsan", "Eman", "Mudassar", "Shehnaz", "Sarmad", "Nayab",
    "Moiz", "Humaira", "Talha", "Bisma", "Shahbaz", "Ghazala", "Zia", "Salma"
]

PAKISTANI_FIRST_VAL = [
    "Sikandar", "Zoya", "Iftikhar", "Hoorain", "Mustafa", "Kinza", "Sheraz", "Manahil",
    "Jahangir", "Shanzay", "Luqman", "Ayla", "Murtaza", "Alishba", "Daniyal", "Hania",
    "Abbas", "Rameen", "Shafqat", "Minahil", "Khawar", "Zimal", "Tanveer", "Aiza",
    "Raza", "Mirha", "Mansoor", "Hareem", "Waqar", "Momina"
]

PAKISTANI_FIRST_TEST = [
    "Azhar", "Aleena", "Ghulam", "Aima", "Arshad", "Zarnish", "Shaukat", "Zoha",
    "Sarwar", "Inaya", "Pervez", "Meerab", "Akram", "Anaya", "Altaf", "Mawra",
    "Anwar", "Aayat", "Mumtaz", "Washma", "Liaquat", "Hoor", "Mushtaq", "Areesha",
    "Saeed", "Zarmgheh", "Ashraf", "Zunera", "Mahmood", "Syeda"
]

# Western Last Names
WESTERN_LAST_TRAIN = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers",
    "Long", "Ross", "Foster", "Jimenez", "Powell", "Jenkins", "Perry", "Russell"
]

WESTERN_LAST_VAL = [
    "Sullivan", "Bell", "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher",
    "Vasquez", "Simmons", "Romero", "Jordan", "Patterson", "Alexander", "Hamilton", "Graham",
    "Reynolds", "Griffin", "Wallace", "Moreno", "West", "Cole", "Hayes", "Bryant", "Herrera"
]

WESTERN_LAST_TEST = [
    "Gibson", "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray", "Ford",
    "Castro", "Marshall", "Owens", "Harrison", "Fernandez", "McDonald", "Woods", "Washington",
    "Kennedy", "Wells", "Vargas", "Henry", "Chen", "Freeman", "Webb", "Tucker", "Guzman"
]

# Pakistani Last Names
PAKISTANI_LAST_TRAIN = [
    "Khan", "Malik", "Chaudhry", "Butt", "Raza", "Bhatti", "Sheikh", "Siddiqui",
    "Qureshi", "Ansari", "Mirza", "Dar", "Abbasi", "Farooqi", "Hashmi", "Gillani",
    "Baig", "Usmani", "Ghafoor", "Laghari", "Khattak", "Mughal", "Niazi", "Jutt",
    "Awan", "Rehman", "Shah", "Gondal", "Warraich", "Khokhar", "Cheema", "Tareen",
    "Durrani", "Tarar", "Wattoo", "Mian", "Pasha", "Gardezi", "Bukhari", "Kazmi",
    "Rasheed", "Aziz", "Iqbal", "Saeed", "Akhtar", "Zaman", "Sharif", "Hameed",
    "Mahmood", "Yousaf", "Majeed", "Yasin", "Shabbir", "Bashir", "Sultan", "Sabir",
    "Munir", "Rafiq", "Nazir", "Latif", "Javed", "Hayat", "Afzal", "Akram",
    "Aslam", "Anwar", "Arif", "Ayub", "Hafeez", "Masood", "Mushtaq", "Nasim",
    "Parvez", "Sadiq", "Sarwar", "Shakoor", "Sohail", "Younas", "Yaqoob", "Zafar",
    "Abid", "Amjad", "Ashraf", "Babar", "Feroz", "Gul", "Habib", "Hamid",
    "Idrees", "Imtiaz", "Ismail", "Jameel", "Kamal", "Khalid", "Liaquat", "Manzoor",
    "Nadeem", "Qasim", "Riaz", "Saleem", "Shahzad", "Tanveer", "Wahid", "Zahid"
]

PAKISTANI_LAST_VAL = [
    "Mazari", "Bizenjo", "Bugti", "Mengal", "Marwat", "Afridi", "Orakzai", "Bangash",
    "Shinwari", "Yusufzai", "Hoti", "Babbar", "Ranjha", "Sial", "Kharal", "Tiwanas",
    "Noon", "Daultana", "Leghari", "Makhdoom", "Pirzada", "Gilani", "Gabol", "Talpur", "Jamali"
]

PAKISTANI_LAST_TEST = [
    "Soomro", "Chandio", "Jatoi", "Khuhro", "Magsi", "Zardari", "Bhutto", "Palijo",
    "Junejo", "Syed", "Bijarani", "Mahar", "Kalhoro", "Alvi", "Memon", "Dawood",
    "Habibullah", "Lakhani", "Tabani", "Rangoonwala", "Fecto", "Monnoo", "Adamjee", "Saigol", "Chinioti"
]


# ==========================================
# 2. ADDRESS POOLS (Partitioned 70/15/15)
# ==========================================

US_STREETS_TRAIN = [
    "Maple", "Oak", "Cedar", "Pine", "Elm", "Washington", "Lake", "Hill",
    "Walnut", "Spring", "North", "Ridge", "Lincoln", "Church", "Willow", "Sunset",
    "Railroad", "Jackson", "Park", "Meadow", "Highland", "Forest", "River", "Valley",
    "Franklin", "Adams", "Madison", "Jefferson", "Monroe", "Chestnut", "Spruce"
]

US_STREETS_VAL = [
    "Lexington", "Broad", "Market", "Front", "Center", "Union", "Court", "High",
    "Commercial", "Pearl", "State", "Water", "Cherry", "Academy", "Fairview"
]

US_STREETS_TEST = [
    "Broadway", "Grove", "Laurel", "Orchard", "Linden", "Summit", "Cypress", "Beech",
    "Poplar", "Magnolia", "Sycamore", "Birch", "Prospect", "College", "Riverside"
]

US_CITIES_TRAIN = [
    ("Austin", "TX", "78701"), ("Chicago", "IL", "60601"), ("Seattle", "WA", "98101"),
    ("Denver", "CO", "80201"), ("Atlanta", "GA", "30301"), ("San Francisco", "CA", "94102"),
    ("Boston", "MA", "02108"), ("Dallas", "TX", "75201"), ("Phoenix", "AZ", "85001"),
    ("San Jose", "CA", "95113"), ("Charlotte", "NC", "28202"), ("Indianapolis", "IN", "46204")
]

US_CITIES_VAL = [
    ("Philadelphia", "PA", "19101"), ("St. Louis", "MO", "63101"), ("Pittsburgh", "PA", "15201"),
    ("New Orleans", "LA", "70112"), ("Columbus", "OH", "43215"), ("Minneapolis", "MN", "55401")
]

US_CITIES_TEST = [
    ("San Antonio", "TX", "78201"), ("Detroit", "MI", "48201"), ("Baltimore", "MD", "21201"),
    ("Albuquerque", "NM", "87101"), ("Nashville", "TN", "37201"), ("Sacramento", "CA", "95814")
]

# Lahore Areas & Sectors
LAHORE_AREAS_TRAIN = [
    ("Gulberg II", "54660"), ("Gulberg III", "54000"), ("Model Town Block A", "54700"),
    ("Model Town Block C", "54700"), ("Garden Town", "54600"), ("Shadman", "54000"),
    ("Faisal Town", "54770"), ("Johar Town Phase 1", "54782"), ("Cavalry Ground", "54810"),
    ("New Garden Town", "54600"), ("Samanabad", "54500"), ("Township Sector B", "54770")
]

LAHORE_AREAS_VAL = [
    ("DHA Phase 1 Sector A", "54792"), ("DHA Phase 2 Sector C", "54792"), ("Muslim Town", "54600"),
    ("Allama Iqbal Town", "54570"), ("Cantt Sector B", "54810"), ("PCSIR Phase 1", "54590")
]

LAHORE_AREAS_TEST = [
    ("DHA Phase 5 Sector G", "54792"), ("DHA Phase 6 Sector J", "54792"), ("Bahria Town Sector C", "53720"),
    ("WAPDA Town Phase 1", "54770"), ("Askari 10 Sector D", "54810"), ("State Life Society Phase 1", "54792")
]


# ==========================================
# 3. EMP_ID POOLS (Range Partitioned)
# ==========================================
# Train: 10000 - 59999
# Val:   60000 - 79999
# Test:  80000 - 99999
EMP_ID_RANGES = {
    "train": (10000, 59999),
    "val": (60000, 79999),
    "test": (80000, 99999),
    "doc": (10000, 59999)
}


# ==========================================
# 4. PHONE POOLS (Area/Prefix Partitioned)
# ==========================================
US_AREA_CODES = {
    "train": [212, 312, 415, 617, 713, 206, 303, 404, 503, 602, 702, 801, 858, 919, 214, 512, 619, 707],
    "val":   [215, 314, 412, 504, 614, 704, 804, 914, 216, 317, 414, 515],
    "test":  [210, 313, 410, 505, 615, 703, 805, 916, 217, 319, 423, 518]
}

PK_PHONE_PREFIXES = {
    "train": ["+92 300", "+92 301", "+92 302", "0300-", "0301-", "0302-", "+92-42-35"],
    "val":   ["+92 310", "+92 311", "+92 312", "0310-", "0311-", "0312-", "+92-42-36"],
    "test":  ["+92 320", "+92 321", "+92 322", "0320-", "0321-", "0322-", "+92-42-37"]
}


# ==========================================
# 5. SALARY POOLS (Amounts & Currencies)
# ==========================================
# Note: SALARY is exempt from zero-overlap check
SALARY_AMOUNTS_USD = [
    "$65,000", "$72,000", "$85,000", "$92,000", "$95,000", "$105,000",
    "$115,000", "$120,000", "$125,000", "$135,000", "$140,000", "$150,000",
    "$165,000", "$175,000", "$185,000", "$195,000", "$210,000", "$225,000",
    "$250,000", "$280,000", "$310,000", "$350,000", "$420,000", "$500,000",
    "USD 75,000", "USD 95,000", "USD 120,000", "USD 145,000", "USD 180,000",
    "USD 220,000", "USD 260,000", "USD 320,000",
    "$85k", "$95k", "$110k", "$125k", "$140k", "$160k", "$180k", "$220k"
]

SALARY_AMOUNTS_PKR = [
    "PKR 180,000", "PKR 250,000", "PKR 320,000", "PKR 350,000", "PKR 450,000",
    "PKR 550,000", "PKR 650,000", "PKR 750,000", "PKR 850,000", "PKR 950,000",
    "PKR 1,100,000", "PKR 1,250,000", "PKR 1,500,000", "PKR 1,800,000", "PKR 2,200,000",
    "PKR 2,800,000", "PKR 3,500,000", "PKR 4,200,000",
    "Rs. 220,000", "Rs. 380,000", "Rs. 600,000", "Rs. 850,000", "Rs. 1,200,000",
    "Rs. 1,600,000", "Rs. 2,400,000", "Rs. 3,200,000"
]

SALARY_AMOUNTS_INTL = [
    "EUR 65,000", "EUR 78,000", "EUR 92,000", "EUR 115,000", "EUR 140,000",
    "GBP 55,000", "GBP 70,000", "GBP 85,000", "GBP 110,000", "GBP 135,000"
]

ALL_SALARIES = SALARY_AMOUNTS_USD + SALARY_AMOUNTS_PKR + SALARY_AMOUNTS_INTL


# ==========================================
# 6. HARD NEGATIVE POOLS (Non-Entity Tokens)
# ==========================================
COMPANIES = [
    "Acme Global Corp", "TechMatrix Industries", "Vertex Solutions", "CloudScale Inc",
    "InnoVate Dynamics", "Nexus AI Systems", "AlphaLogix Corp", "DataVault Corp",
    "OmniTech Global", "Synapse Software", "HyperScale Networks", "BlueWave Systems"
]

PRODUCTS = [
    "CloudCore v3.2", "DataSync Pro", "Sentinel AI", "FlowMatrix 4.0",
    "NeuralEngine v2", "CyberShield Enterprise", "OmniPlatform 5.1", "SecureTunnel 2.4"
]

STANDALONE_CITIES = [
    "Chicago", "Seattle", "San Francisco", "Denver", "Boston", "Atlanta",
    "Lahore", "Karachi", "Islamabad", "Austin", "New York", "Dallas"
]

REVENUE_FIGURES = [
    "$45M in annual revenue", "$12.8M ARR", "$850,000 quarterly marketing budget",
    "$15M Series B financing", "$4.2M operating profit", "$1.8M IT infrastructure spend",
    "$95M annual gross merchandise value", "$28M total turnover"
]

HEADCOUNTS = [
    "team of 45 engineers", "headcount of 1,200 employees", "staff of 85 researchers",
    "department of 32 specialists", "organization of 3,500 workers", "team of 18 analysts"
]

DATES = [
    "October 14, 2024", "November 2026", "Q3 2025", "March 15, 2025",
    "2026-11-14", "June 30, 2024", "January 10, 2026", "Q4 2026"
]

VERSION_NUMBERS = [
    "v2.4.1", "Release 5.0.0", "v1.8-patch3", "v3.10.4", "Build 9421", "v4.2-rc2"
]

TICKET_IDS = [
    "TKT-48213", "BUG-1092", "INC-9941", "TASK-7721", "REQ-3319",
    "PR-4402", "ISSUE-5820", "TKT-19402", "INC-3819", "TASK-8901"
]


# ==========================================
# 7. GENERATOR CLASS
# ==========================================
class EntityPoolGenerator:
    """Generates synthetic entities strictly isolated by split."""

    def __init__(self, split: str = "train", seed: int = 42):
        if split not in ("train", "val", "test", "doc"):
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.rng = random.Random(seed)

    def generate_person(self, is_public_figure: bool = False) -> str:
        """Returns full name (western or pakistani mix). Titles excluded."""
        split_key = "train" if self.split == "doc" else self.split
        is_pakistani = self.rng.random() < 0.5
        
        if split_key == "train":
            first_pool = PAKISTANI_FIRST_TRAIN if is_pakistani else WESTERN_FIRST_TRAIN
            last_pool = PAKISTANI_LAST_TRAIN if is_pakistani else WESTERN_LAST_TRAIN
        elif split_key == "val":
            first_pool = PAKISTANI_FIRST_VAL if is_pakistani else WESTERN_FIRST_VAL
            last_pool = PAKISTANI_LAST_VAL if is_pakistani else WESTERN_LAST_VAL
        else: # test
            first_pool = PAKISTANI_FIRST_TEST if is_pakistani else WESTERN_FIRST_TEST
            last_pool = PAKISTANI_LAST_TEST if is_pakistani else WESTERN_LAST_TEST
            
        first = self.rng.choice(first_pool)
        last = self.rng.choice(last_pool)
        return f"{first} {last}"

    def generate_emp_id(self) -> str:
        """EMP_ID: 80% EMP-12345, 20% EMP12345 or E-12345."""
        split_key = "train" if self.split == "doc" else self.split
        min_id, max_id = EMP_ID_RANGES[split_key]
        num = self.rng.randint(min_id, max_id)
        
        r = self.rng.random()
        if r < 0.80:
            return f"EMP-{num}"
        elif r < 0.90:
            return f"EMP{num}"
        else:
            return f"E-{num}"

    def generate_email(self, person_name: Optional[str] = None) -> str:
        """Email derived from person name or split-specific user name."""
        if person_name:
            parts = person_name.lower().split()
            first, last = parts[0], parts[-1]
        else:
            p = self.generate_person()
            parts = p.lower().split()
            first, last = parts[0], parts[-1]
            
        domains = ["corp.acme.com", "enterprise.org", "techmatrix.net", "global.pk", "privaterag.internal"]
        domain = self.rng.choice(domains)
        
        pattern_type = self.rng.choice([1, 2, 3])
        if pattern_type == 1:
            return f"{first}.{last}@{domain}"
        elif pattern_type == 2:
            return f"{first[0]}{last}@{domain}"
        else:
            split_key = "train" if self.split == "doc" else self.split
            min_id, max_id = EMP_ID_RANGES[split_key]
            num = self.rng.randint(min_id, min_id + 999)
            return f"{first}.{num}@{domain}"

    def generate_phone(self) -> str:
        """US and +92 formats."""
        split_key = "train" if self.split == "doc" else self.split
        is_pakistani = self.rng.random() < 0.5
        
        if is_pakistani:
            prefix = self.rng.choice(PK_PHONE_PREFIXES[split_key])
            if prefix.endswith("-") or prefix.startswith("+92-42"):
                num = self.rng.randint(1000000, 9999999)
                return f"{prefix}{num}"
            else:
                num = self.rng.randint(1000000, 9999999)
                return f"{prefix} {num}"
        else:
            area = self.rng.choice(US_AREA_CODES[split_key])
            mid = self.rng.randint(200, 999)
            end = self.rng.randint(1000, 9999)
            fmt = self.rng.choice([1, 2, 3])
            if fmt == 1:
                return f"({area}) {mid}-{end}"
            elif fmt == 2:
                return f"{area}-{mid}-{end}"
            else:
                return f"+1-{area}-{mid}-{end}"

    def generate_address(self) -> str:
        """One-line full address as ONE entity: 70% US-style, 30% Lahore-style."""
        split_key = "train" if self.split == "doc" else self.split
        is_lahore = self.rng.random() < 0.30
        
        if is_lahore:
            if split_key == "train":
                area_pool = LAHORE_AREAS_TRAIN
            elif split_key == "val":
                area_pool = LAHORE_AREAS_VAL
            else:
                area_pool = LAHORE_AREAS_TEST
            area, zip_code = self.rng.choice(area_pool)
            house_num = self.rng.randint(1, 450)
            street_num = self.rng.randint(1, 35)
            types = [
                f"House {house_num}, Street {street_num}, {area}, Lahore, {zip_code}",
                f"Flat {self.rng.choice(['2A', '3B', '4C', '5D', '101', '204'])}, {area}, Lahore, {zip_code}",
                f"Plot {house_num}, Block {self.rng.choice(['A', 'B', 'C', 'D', 'E', 'F'])}, {area}, Lahore, {zip_code}"
            ]
            return self.rng.choice(types)
        else:
            if split_key == "train":
                st_pool = US_STREETS_TRAIN
                city_pool = US_CITIES_TRAIN
            elif split_key == "val":
                st_pool = US_STREETS_VAL
                city_pool = US_CITIES_VAL
            else:
                st_pool = US_STREETS_TEST
                city_pool = US_CITIES_TEST
            street = self.rng.choice(st_pool)
            city, state, zip_code = self.rng.choice(city_pool)
            num = self.rng.randint(100, 9999)
            suffix = self.rng.choice(["Street", "Avenue", "Boulevard", "Drive", "Lane", "Way", "Road"])
            return f"{num} {street} {suffix}, {city}, {state} {zip_code}"

    def generate_salary(self) -> str:
        """SALARY = currency + amount only."""
        return self.rng.choice(ALL_SALARIES)

    def generate_hard_negative(self, neg_type: str) -> str:
        """Returns hard negative string that looks like an entity but is NOT one."""
        if neg_type == "company":
            return self.rng.choice(COMPANIES)
        elif neg_type == "product":
            return self.rng.choice(PRODUCTS)
        elif neg_type == "city":
            return self.rng.choice(STANDALONE_CITIES)
        elif neg_type == "revenue":
            return self.rng.choice(REVENUE_FIGURES)
        elif neg_type == "headcount":
            return self.rng.choice(HEADCOUNTS)
        elif neg_type == "date":
            return self.rng.choice(DATES)
        elif neg_type == "version":
            return self.rng.choice(VERSION_NUMBERS)
        elif neg_type == "ticket":
            return self.rng.choice(TICKET_IDS)
        else:
            return "General System Notice"
