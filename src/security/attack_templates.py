import random
from typing import Dict

# -------------------------------
# SQL Injection Templates
# -------------------------------
def generate_sql_injection() -> Dict[str, object]:
    payloads = [
        "' OR '1'='1",
        "' UNION SELECT NULL--",
        "' OR sleep(5)--",
        "' AND 1=0 UNION SELECT username, password FROM users--"
    ]

    return {
        "method": random.choice(["GET", "POST"]),
        "path": random.choice(["/login", "/admin/login", "/api/auth"]),
        "request": f"username=admin{random.choice(payloads)}",
        "attack_type": "sql_injection"
    }

# -------------------------------
# Path Traversal Templates
# -------------------------------
def generate_path_traversal() -> Dict[str, object]:
    traversal = [
        "../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "../../var/log/auth.log"
    ]

    return {
        "method": "GET",
        "path": f"/download/{random.choice(traversal)}",
        "request": "",
        "attack_type": "path_traversal"
    }

# -------------------------------
# Credential Stuffing Templates
# -------------------------------
def generate_credential_stuffing() -> Dict[str, object]:
    usernames = ["admin", "root", "test", "user"]

    return {
        "method": "POST",
        "path": "/login",
        "request": f"username={random.choice(usernames)}&password=wrongpassword",
        "attack_type": "credential_stuffing"
    }

# -------------------------------
# Privilege Escalation Templates
# -------------------------------
def generate_privilege_escalation() -> Dict[str, object]:
    return {
        "method": "GET",
        "path": "/admin/config",
        "request": "",
        "attack_type": "privilege_escalation"
    }

# -------------------------------
# Dispatcher
# -------------------------------
ATTACK_GENERATORS = [
    generate_sql_injection,
    generate_path_traversal,
    generate_credential_stuffing,
    generate_privilege_escalation
]

def sample_attack() -> Dict[str, object]:
    generator = random.choice(ATTACK_GENERATORS)
    return generator()
