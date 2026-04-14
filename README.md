# Travel Destination Recommender

A content-based recommendation system that suggests the best cities to visit based on user travel preferences, built on a dataset of 560+ worldwide cities.

---

## What it does

The user rates 8 travel themes from 0 (not important) to 5 (very important):
- Culture, Adventure, Nature, Beaches, Nightlife, Cuisine, Wellness, Urban

The system returns the top 20 most similar cities based on those preferences.

---

## Pipeline

**1. Data Loading and Cleaning**
560+ cities with ratings across 8 travel themes and monthly temperature data. Handled missing values and parsed JSON-formatted temperature fields.

**2. Exploratory Data Analysis**
Visualised distributions, correlation between themes, and budget breakdowns. Key finding: culture and cuisine ratings are highly correlated.

**3. Feature Engineering and Scaling**
Extracted temperature features for the user's selected travel month. Applied StandardScaler to normalise all features — essential so temperature values don't dominate theme ratings.

**4. K-Means Clustering**
Used the elbow method to find the optimal number of clusters — plotting inertia against k values and selecting the point where improvement slows. Grouped cities into clusters of similar travel profiles.

**5. PCA Visualisation**
Compressed 11 features to 2 dimensions to visualise clusters — validating that similar cities genuinely grouped together.

**6. Cosine Similarity Recommendation**
User preferences converted to a scaled vector. Cosine similarity measures alignment between user preferences and each city's profile. Top 20 matches returned ranked by similarity score.

---

## Technologies

Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Plotly

---

## Key Insight

Cities with high culture ratings tend to also score highly on cuisine — suggesting these travel styles are naturally linked. The elbow method identified meaningful natural groupings in the data, validating that cities cluster into distinct travel personas.
