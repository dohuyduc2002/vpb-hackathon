transaction_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "Transaction",
    "fields": [
        {"name": "idx", "type": "int"},
        {"name": "user", "type": "string"},
        {"name": "card", "type": "string"},
        {"name": "year", "type": "int"},
        {"name": "month", "type": "int"},
        {"name": "day", "type": "int"},
        {"name": "time", "type": "string"},
        {"name": "amount", "type": "string"},
        {"name": "use_chip", "type": ["null", "string"], "default": None},
        {"name": "merchant_name", "type": ["null", "long"], "default": None},
        {"name": "merchant_city", "type": ["null", "string"], "default": None},
        {"name": "merchant_state", "type": ["null", "string"], "default": None},
        {"name": "zip", "type": ["null", "double"], "default": None},
        {"name": "mcc", "type": "int"},
        {"name": "errors", "type": ["null", "string"], "default": None},
        {"name": "is_fraud", "type": "string"},
    ],
}

user_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "User",
    "fields": [
        {"name": "idx", "type": "int"},
        {"name": "user", "type": "string"},
        {"name": "person", "type": "string"},
        {"name": "current_age", "type": "int"},
        {"name": "retirement_age", "type": "int"},
        {"name": "birth_year", "type": "int"},
        {"name": "birth_month", "type": "int"},
        {"name": "gender", "type": "string"},
        {"name": "address", "type": "string"},
        {"name": "apartment", "type": ["null", "double"], "default": None},
        {"name": "city", "type": "string"},
        {"name": "state", "type": "string"},
        {"name": "zipcode", "type": "int"},
        {"name": "latitude", "type": "double"},
        {"name": "longitude", "type": "double"},
        {"name": "per_capita_income__zipcode", "type": "string"},
        {"name": "yearly_income__person", "type": "string"},
        {"name": "total_debt", "type": "string"},
        {"name": "fico_score", "type": "int"},
        {"name": "num_credit_cards", "type": "int"},
    ],
}


card_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "Card",
    "fields": [
        {"name": "idx", "type": "int"},
        {"name": "user", "type": "string"},
        {"name": "card_index", "type": "long"},
        {"name": "card_brand", "type": "string"},
        {"name": "card_type", "type": "string"},
        {"name": "card_number", "type": "long"},
        {"name": "expires", "type": "string"},
        {"name": "cvv", "type": "int"},
        {"name": "has_chip", "type": "string"},
        {"name": "cards_issued", "type": "long"},
        {"name": "credit_limit", "type": "string"},
        {"name": "acct_open_date", "type": "string"},
        {"name": "year_pin_last_changed", "type": "int"},
        {"name": "card_on_dark_web", "type": "string"},
    ],
}
