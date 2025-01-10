INDEX = ["date_id", "time_id", "symbol_id"]
FEATURES = [f"feature_{idx:02d}" for idx in range(79)]
RESPONDERS = [f"responder_{idx}" for idx in range(9)]

# Responders are simple moving averages of an underlying signal.
SMA_RESPONDER_MAP = {
    20: ["responder_0", "responder_3", "responder_6"],
    120: ["responder_1", "responder_4", "responder_7"],
    4: ["responder_2", "responder_5", "responder_8"],
}
