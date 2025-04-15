from dataset import create_dataset_index, create_data_splits

# Create the main dataset index with all features
df = create_dataset_index()

# Split into train/val/test
create_data_splits('../data/dataset_index.csv')