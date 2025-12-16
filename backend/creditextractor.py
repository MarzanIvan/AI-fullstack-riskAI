from datetime import date
import numpy as np

def extract_credit_features(payload: dict) -> dict:
    today = date.today()

    birth = date.fromisoformat(payload["birth_date"])
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

    loans = payload["loans"]

    total_loans = len(loans)
    active_loans = sum(1 for l in loans if l["current_status"] == "active")
    closed_loans = sum(1 for l in loans if l["current_status"] == "closed")

    total_loan_amount = sum(l["loan_amount"] for l in loans)
    total_remaining = sum(l["remaining_balance"] for l in loans)

    delays = [l["max_delay_days"] for l in loans]
    max_delay = max(delays) if delays else 0
    avg_delay = np.mean(delays) if delays else 0

    credit_card_count = sum(1 for l in loans if l["loan_type"] == "credit_card")
    consumer_count = sum(1 for l in loans if l["loan_type"] == "consumer")

    monthly_income = payload["monthly_income"]

    return {
        "age": age,
        "monthly_income": monthly_income,
        "income_sources_count": len(payload["income_sources"]),
        "is_russian": int(payload["citizenship"].lower() == "россия"),
        "is_legally_capable": int(payload["capacity_status"] == "дееспособен"),

        "total_loans": total_loans,
        "active_loans": active_loans,
        "closed_loans": closed_loans,

        "total_loan_amount": total_loan_amount,
        "total_remaining_balance": total_remaining,

        "max_delay_days": max_delay,
        "avg_delay_days": avg_delay,

        "credit_card_share": credit_card_count / (total_loans + 1e-6),
        "consumer_loan_share": consumer_count / (total_loans + 1e-6),

        "debt_to_income": total_remaining / (monthly_income + 1),
        "requested_to_income": payload["requested_loan_amount"] / (monthly_income + 1),
        "requested_to_total_debt": payload["requested_loan_amount"] / (total_remaining + 1),
    }
