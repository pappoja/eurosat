# EuroSAT Image Classification (with Non-Image Data)

This repository contains code to work with the EuroSAT_MS dataset (13-band Sentinel-2 satellite images) and associated non-image metadata. It includes scripts for preprocessing, feature engineering, and modeling.

---

## Project Structure

```
eurosat/
├── src/                # All core Python code
│   └── load_data.py    # Script to generate the `data/` folder
├── data/               # Generated training data (ignored from Git)
├── nonimage_data/      # Source CSVs used to create data/ (only `nonimage_data.csv` is tracked)
│   └── nonimage_data.csv
├── EuroSAT_MS/         # TIFF image files (ignored)
├── .gitignore
└── README.md
```

---

## Data Requirements

This repository **does not include the full dataset** due to size. To run the pipeline:

1. **Download the EuroSAT_MS dataset**:
   - Link: [https://madm.dfki.de/files/sentinel/EuroSATallBands.zip](https://github.com/phelber/EuroSAT)
   - You need the `.tif` files (2GB+).

2. **Place the data**:
   - Create a folder `EuroSAT_MS/` inside the project root (this is created from .zip file).
   - Place **all .tif files** inside that folder:

   ```bash
   eurosat/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif
   eurosat/EuroSAT_MS/Forest/Forest_1.tif
   ...
   ```

3. **Generate the processed data**:
   Run the following script to create the cleaned dataset:

   ```bash
   python src/load_data.py
   ```

   This script will:
   - Read all `.csv` files in `nonimage_data/`
   - Extract 2017 columns and merge them on `country`
   - Match metadata to image files in `EuroSAT_MS/`
   - Save the final dataset to `data/`

---

## What is Not Tracked by Git

To avoid pushing large or generated files to GitHub, this repo uses a `.gitignore` file that excludes:

```gitignore
EuroSAT_MS/
data/
nonimage_data/*
!nonimage_data/nonimage_data.csv
*.tif
```

Only the source code, `nonimage_data.csv`, and documentation are tracked.

---

## Notes

- Make sure your `EuroSAT_MS/` folder is correctly structured.
- You can modify `load_data.py` to support different variables or CSV schemas.
- If you want to keep certain files in ignored folders, use exception rules in `.gitignore`.

