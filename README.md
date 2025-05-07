# EuroSAT Image Classification (with Non-Image Data)

This repository contains code to work with the EuroSAT_MS dataset (13-band Sentinel-2 satellite images) and associated non-image metadata. It includes scripts for preprocessing, feature engineering, and modeling.

---

## Project Structure

```
eurosat/
├── src/                # All core Python code
│   ├── dataset.py      # Dataset handling and preprocessing
│   ├── train.py        # Training script for deep learning models
│   ├── simple_model.py # Training script for simple models
│   └── load_data.py    # Script to generate the `data/` folder
├── data/               # Generated training data
│   ├── csv_data/       # CSV files for dataset splits
│   └── EuroSAT_MS/     # EuroSAT image files (ignored)
│   └── ne_10m_admin_0_countries/     # Shapefile for country boundaries
├── nonimage_data/      
│   └── ne_10m_admin_0_countries/   # Files for assigning a country to each coordinate pair
├── results/            # Output results
├── data_inspection.ipynb # Various exploratory data cleaning and analysis tasks
├── google_earth_data.ipynb # Download and process Google Earth data (also available in csv_data/)
├── colab_inference.ipynb # Notebook for training models on Google Colab
├── .gitignore
└── README.md
```

---

## Data Requirements

This repository **does not include the full dataset** due to size. To run the pipeline:

1. **Download the EuroSAT dataset**:
   - Link: [EuroSAT Dataset (RGB)](https://madm.dfki.de/files/sentinel/EuroSAT.zip)

2. **Place the data**:
   - Rename the folder `EuroSAT_MS/` (this is a misnomer since it is only the RGB data)
   - Place the folder (directly from the .zip file above) inside the project root.
   - This includes all `.jpg` files, grouped into folders by each classification label.
   - Note: The coordinates are NOT included here, but they can be accessed in `data/csv_data/dataset_index.csv`.

   ```bash
   eurosat/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif
   eurosat/EuroSAT_MS/Forest/Forest_1.tif
   ...
   ```

3. **Generate the processed data**:
   Run the following script to create the cleaned dataset (from `src/`):

   ```bash
   python load_data.py
   ```

   This script will:
   - Read all `.csv` files in `nonimage_data/`
   - Merge country-specific variables on `country`
   - Match metadata to image files in `EuroSAT_MS/`
   - Save the final dataset to `data/`

4. **Optional: Download and process Google Earth data**:
   - Use `google_earth_data.ipynb` to download additional data and remake splits.
   - Pre-generated CSV files are available in the repository.

---

## Training Models

To train models, use the `train.py` script. This script supports various model architectures and input types.

```bash
python train.py --data-dir <data-dir> --image-dir <image-dir> --model <model-type> --input <input-type> --n_epochs <num-epochs>
```

- `<data-dir>`: Directory containing the processed CSV data.
- `<image-dir>`: Directory containing the EuroSAT_MS images.
- `<model-type>`: Model architecture to use (e.g., `simplecnn`, `resnet50`, `biresnet50`).
- `<input-type>`: Input data type (e.g., `image`, `image_country`, `image_country_all`).
- `<num-epochs>`: Number of training epochs.

---

## Notes

- Make sure your `EuroSAT_MS/` folder is correctly structured.
- You can modify `dataset.py` and `train.py` to support different variables or model configurations.
- If you want to keep certain files in ignored folders, use exception rules in `.gitignore`.
