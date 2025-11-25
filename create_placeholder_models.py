"""
Creates placeholder model files for the emotion detection system.
Run this script to generate the required .pkl files before running the Flask app.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("Creating model files...")

# Create a minimal Random Forest with dummy data
X_dummy = np.random.rand(100, 48)  # 100 samples, 48 features
y_dummy = np.random.randint(0, 4, 100)  # 4 emotion classes

# Train a minimal model
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_dummy, y_dummy)

# Save the model
joblib.dump(clf, "emotion_classifier_rf.pkl")
print("✓ Created emotion_classifier_rf.pkl")

# Create and save scaler
scaler = StandardScaler()
scaler.fit(X_dummy)
joblib.dump(scaler, "scaler.pkl")
print("✓ Created scaler.pkl")

print("\n✅ Model files created successfully!")
print("You can now run the Flask application.")
