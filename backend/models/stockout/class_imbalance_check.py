from collections import Counter
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def handle_class_imbalance(X, y, imbalance_threshold=0.4):
    """
    Detects imbalance in binary classification:
    - Drops datetime columns
    - Applies SMOTE if minority < threshold
    - Includes tqdm progress bars
    """

    print("\n🔍 Checking class imbalance...")

    # -----------------------------------------------------------
    # STEP 1: Drop datetime columns (with progress bar)
    # -----------------------------------------------------------
    datetime_cols = [col for col in X.columns if str(X[col].dtype).startswith("datetime")]

    for col in tqdm(datetime_cols, desc="Dropping datetime cols", leave=False):
        print(f"⚠️ Dropping datetime column: {col}")
        X = X.drop(columns=[col])
    
    # -----------------------------------------------------------
    # STEP 2: Check class distribution
    # -----------------------------------------------------------
    class_counts = Counter(y)
    total = len(y)

    class_ratios = {cls: round(cnt / total, 3) for cls, cnt in class_counts.items()}

    print("📊 Class Distribution:", class_counts)
    print("📈 Class Ratios:", class_ratios)

    # -----------------------------------------------------------
    # STEP 3: Detect imbalance
    # -----------------------------------------------------------
    minority_ratio = min(class_ratios.values())

    if minority_ratio >= imbalance_threshold:
        print("✅ No significant class imbalance detected. Skipping SMOTE.")
        return X, y

    # -----------------------------------------------------------
    # STEP 4: Apply SMOTE with progress bar
    # -----------------------------------------------------------
    print("⚠️ Class imbalance detected!")
    print("🔧 Applying SMOTE oversampling...")

    # Artificial progress bar for SMOTE process
    for _ in tqdm(range(1), desc="SMOTE Processing"):
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)

    print("✔️ SMOTE applied successfully.")
    print("📊 New Class Distribution:", Counter(y_resampled))

    return X_resampled, y_resampled
