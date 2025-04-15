# EuroSAT Image Classification (with Non-Image Data)

This repository contains code to work with the EuroSAT_MS dataset (13-band Sentinel-2 satellite images) and associated non-image metadata. It includes scripts for preprocessing, feature engineering, and modeling.

---

## Project Structure

```
eurosat/
├── src/                # All core Python code
│   └── load_data.py    # Script to generate the `data/` folder
├── data/               # Generated training data (ignored from Git)
├── nonimage_data/      
│   └── nonimage_data.csv   # Source CSVs used to create data/
│   └── ne_10m_admin_0_countries/   # Files for assigning a country to each coordinate pair
├── EuroSAT_MS/         # TIFF image files (ignored)
├── .gitignore
└── README.md
```

---

## Data Requirements

This repository **does not include the full dataset** due to size. To run the pipeline:

1. **Download the EuroSAT_MS dataset**:
   - Link: [EuroSAT Dataset (MS)](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip) (3GB)

2. **Place the data**:
   - Place the folder `EuroSAT_MS/` (directly from the .zip file above) inside the project root.
   - This includes the `.tif` files, grouped into folders by each classification label.

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
   - Merge country-specific variables on `country`
   - Match metadata to image files in `EuroSAT_MS/`
   - Save the final dataset to `data/`

---

## Notes

- Make sure your `EuroSAT_MS/` folder is correctly structured.
- You can modify `load_data.py` to support different variables or CSV schemas.
- If you want to keep certain files in ignored folders, use exception rules in `.gitignore`.

