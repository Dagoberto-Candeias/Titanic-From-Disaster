import pandas as pd
import numpy as np
from titanic_pipeline.preprocessing import kfold_target_encode
import os

# Create sample data
np.random.seed(42)
data = {
    'Survived': np.random.randint(0, 2, 100),
    'Category': np.random.choice(['A', 'B', 'C'], 100)
}
train = pd.DataFrame(data)

print("Testing kfold_target_encode function...")

try:
    # Test the function
    encoded_series = kfold_target_encode(train, 'Category', 'Survived', n_splits=5, seed=42)

    print("Function executed successfully.")
    print(f"Encoded series name: {encoded_series.name}")
    print(f"Encoded series length: {len(encoded_series)}")

    # Check if map file was created
    map_path = "output/cache/te_maps_Category.json"
    if os.path.exists(map_path):
        print("Map file created successfully.")
    else:
        print("Map file not found.")

    # Basic validation
    assert isinstance(encoded_series, pd.Series), "Result is not a pd.Series"
    assert encoded_series.name == "feat_Category_te", f"Series name is {encoded_series.name}, expected feat_Category_te"
    assert len(encoded_series) == len(train), "Series length mismatch"

    print("All basic checks passed.")

except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
