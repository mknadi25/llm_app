# --- Combine id + tag columns from two CSV files ---

import pandas as pd

# Read both CSV files
train_df = pd.read_csv("full train dataset.csv")
test_df = pd.read_csv("test dataset.csv")

# Extract only 'id' and 'tag' columns
train_tags = train_df[["id", "tag"]]
test_tags = test_df[["id", "tag"]]

# Combine the two DataFrames
combined_tags = pd.concat([train_tags, test_tags], ignore_index=True)

# Save the combined result to a new CSV file
combined_tags.to_csv("tags.csv", index=False)

print("✅ tag.csv created successfully with combined id and tag columns!")
