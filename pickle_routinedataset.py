import sys
sys.path.append('helpers')
import os
import pickle
import json
from reader import RoutinesDataset
from encoders import TimeEncodingOptions

# --------- Plug in your values here ---------
data_dir = 'data/HOMER/household0'  # path to dataset folder
batch_size = 32                     # desired batch size
train_days = 30                     # max routines (None if not limited)
time_encoding_type = 'sine_informed'  # or whatever your config uses
# -------------------------------------------

# Load DATA_INFO
with open(os.path.join(data_dir, 'processed', 'common_data.json')) as f:
    data_info = json.load(f)['info']

# Initialize time encoder
weekend_days = data_info.get('weeekend_days', None)
time_options = TimeEncodingOptions(weekend_days)
time_encoding = time_options(time_encoding_type)

# Create the dataset object
data = RoutinesDataset(
    data_path=os.path.join(data_dir, 'processed'),
    time_encoder=time_encoding,
    batch_size=batch_size,
    max_routines=(train_days, None)
)

# Save the object
with open('saved_dataset.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Dataset saved to saved_dataset.pkl")
