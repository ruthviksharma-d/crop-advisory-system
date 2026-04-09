# CROP.AI — Crop Advisory System Web App

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Crop_recommendation.csv is in this folder

# 3. Run the app
python app.py

# 4. Open browser at
http://localhost:5000
```

## Output
<img src="images/Pic1.png" width="1000"/>
<img src="images/Pic2.png" width="1000"/>
<img src="images/Pic3.png" width="1000"/>
<img src="images/Pic4.png" width="1000"/>
<img src="images/Pic5.png" width="1000"/>
<img src="images/Pic6.png" width="1000"/>
<img src="images/Pic7.png" width="1000"/>
<img src="images/Pic8.png" width="1000"/>
<img src="images/Pic9.png" width="1000"/>
<img src="images/Pic10.png" width="1000"/>
<img src="images/Pic11.png" width="1000"/>
<img src="images/Pic12.png" width="1000"/>
<img src="images/Pic13.png" width="1000"/>
<img src="images/Pic14.png" width="1000"/>

## Features
- **Overview** — dataset stats, class distribution, feature importance charts
- **Dataset** — paginated, searchable table of all 2200 rows
- **Results** — train/test accuracy + full classification report per crop
- **Visualise** — actual vs predicted (line + scatter) + confusion matrix heatmap
- **Predict** — enter N/P/K/temp/humidity/pH/rainfall → get crop recommendation with top-5 probabilities + radar chart

## ⚙️ Algorithms & Tech Stack Used

### 🤖 Algorithms

**Random Forest Classifier**
- An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.
- **Why used:** Provides high accuracy, handles nonlinear relationships well, and reduces overfitting—ideal for agricultural data with multiple environmental factors.

**Decision Trees**
- The base learners used within the Random Forest to make individual predictions based on feature splits.
- **Why used:** Easy to interpret and effective for handling structured/tabular data like soil nutrients and weather conditions.

**Bagging (Bootstrap Aggregation)**
- Technique where multiple subsets of data are sampled to train different trees.
- **Why used:** Improves model stability and generalization by reducing variance and avoiding overfitting.

---

### 💻 Tech Stack

**Python**
- Core programming language used for building the application and ML pipeline.
- **Why used:** Rich ecosystem for machine learning and data processing.

**Flask**
- Lightweight web framework for building the application interface and API.
- **Why used:** Simple, flexible, and perfect for deploying ML models as web apps.

**Scikit-learn**
- Machine learning library used to implement the Random Forest model.
- **Why used:** Provides efficient, ready-to-use implementations of ML algorithms.

**Pandas & NumPy**
- Libraries for data manipulation and numerical computations.
- **Why used:** Essential for preprocessing and handling structured agricultural data.

---

## File Structure
```
crop_app/
├── app.py                    ← Flask backend + ML training
├── Crop_recommendation.csv   ← Dataset
├── requirements.txt
└── templates/
    └── index.html            ← Frontend (brutalist UI)
```
