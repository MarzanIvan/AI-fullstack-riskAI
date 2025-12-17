import os
import pandas as pd
import numpy as np

os.makedirs("datasets", exist_ok=True)

# CREDIT
df_credit = pd.DataFrame({
    "income": np.random.randint(20000, 100000, size=100),
    "debt": np.random.randint(0, 50000, size=100),
    "age": np.random.randint(18, 70, size=100),
    "credit_score": np.random.randint(300, 850, size=100),
    "default": np.random.randint(0, 2, size=100)
})
df_credit.to_csv("datasets/credit.csv", index=False)

# INVESTMENT
df_invest = pd.DataFrame({
    "price": np.random.rand(200) * 100,
    "volume": np.random.rand(200) * 1000
})
df_invest.to_csv("datasets/invest.csv", index=False)

# INSURANCE
df_insurance = pd.DataFrame({
    "driver_age": np.random.randint(18, 70, size=100),
    "power": np.random.randint(50, 400, size=100),
    "region_risk": np.random.rand(100),
    "claims_count": np.random.randint(0, 5, size=100),
    "claim_amount": np.random.rand(100) * 10000
})
df_insurance.to_csv("datasets/claims.csv", index=False)

print("✅ Test datasets created")
