import torch
import numpy as np

def preprocess_single(sample, node_mapping, scaler, label_encoders):

    # Clean amount
    amount = float(sample["amount"].replace("$", "").replace(",", ""))

    year = float(sample["year"])
    month = float(sample["month"])
    day = float(sample["day"])

    # Encode Merchant City
    merchant_city = (
        label_encoders["merchant_city"].transform([sample["merchant_city"]])[0]
        if sample["merchant_city"]
        else 0
    )
    use_chip = (
        label_encoders["use_chip"].transform([sample["use_chip"]])[0]
        if sample["use_chip"]
        else 0
    )
    errors = (
        label_encoders["errors"].transform([sample["errors"]])[0]
        if sample["errors"]
        else 0
    )
    mcc = label_encoders["mcc"].transform([sample["mcc"]])[0] if sample["mcc"] else 0
    merchant_name = (
        label_encoders["merchant_name"].transform([sample["merchant_name"]])[0]
        if sample["merchant_name"]
        else 0
    )

    # Scale numeric features
    numeric = np.array([[year, month, day, amount]])
    numeric_scaled = scaler.transform(numeric)

    # Compose edge_attr
    edge_attr = [
        numeric_scaled[0][3],  # amount scaled
        merchant_city,
        use_chip,
        numeric_scaled[0][0],  # year scaled
        numeric_scaled[0][1],  # month scaled
        numeric_scaled[0][2],  # day scaled
        errors,
    ]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(0)

    # Compose src, dst
    card_id = sample["user"] + "_" + sample["card"]
    merchant_node = merchant_name

    src = node_mapping.get(card_id, 0)
    dst = node_mapping.get(merchant_node, 0)

    edge_index = torch.tensor([[src], [dst]], dtype=torch.long)

    return edge_index, edge_attr
