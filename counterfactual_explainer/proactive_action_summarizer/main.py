from log_loader import LogLoader
from log_processor import LogProcessor


household_id = 0  # Example household ID, can be set dynamically
for household_id in range(0, 10):  # Adjust range as needed
    try:
        log_loader = LogLoader(household_id)
        log_processor = LogProcessor(log_loader)
        print(f"Processed logs for household {household_id} successfully.")
    except FileNotFoundError as e:
        print(f"Household ID {household_id} not found: {e}")
        continue
