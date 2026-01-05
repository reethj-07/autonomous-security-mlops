import pandas as pd
import random

def generate_logs(n=500):
    paths = ["/", "/home", "/login", "/admin", "/products"]
    sql_payloads = [
        "SELECT * FROM users",
        "DROP TABLE users",
        "OR 1=1",
        "normal request"
    ]

    data = []
    for _ in range(n):
        path = random.choice(paths)
        payload = random.choice(sql_payloads)
        request = f"{path}?q={payload}"

        data.append({
            "path": path,
            "request": request
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_logs()
    df.to_parquet("data/raw/security_logs.parquet", index=False)
