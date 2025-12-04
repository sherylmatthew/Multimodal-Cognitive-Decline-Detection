"""
Quick Start Script - Cognitive Decline Detection System
Run this file to tracd backendin and test the model
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'sklearn': 'scikit-learn',
        'librosa': 'librosa',
        'cv2': 'opencv-python'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    # Check TensorFlow (optional)
    try:
        import tensorflow as tf
        print(f"✓ tensorflow (version {tf.__version__})")
        use_dl = True
    except ImportError:
        print("✗ tensorflow (will use classical ML instead)")
        use_dl = False
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False, use_dl
    
    return True, use_dl


def create_directory_structure():
    """Create necessary directories"""
    dirs = ['data', 'models', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created directory: {d}/")


def main():
    print("=" * 70)
    print(" " * 15 + "COGNITIVE DECLINE DETECTION SYSTEM")
    print(" " * 20 + "Quick Start Setup")
    print("=" * 70)
    
    print("\n[Step 1] Checking dependencies...")
    print("-" * 70)
    deps_ok, use_dl = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing packages first!")
        return
    
    print("\n[Step 2] Creating directory structure...")
    print("-" * 70)
    create_directory_structure()
    
    print("\n[Step 3] Ready to train model")
    print("-" * 70)
    print("\nYou have two options:")
    print("\nOption A: Copy the complete system")
    print("  1. Copy the 'Complete ML Training & Inference System' code")
    print("  2. Save it as: cognitive_system.py")
    print("  3. Run: python cognitive_system.py")
    
    print("\nOption B: Quick demo (run this script)")
    
    choice = input("\nRun quick demo now? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n[Step 4] Running quick demo...")
        print("-" * 70)
        run_quick_demo(use_dl)
    else:
        print("\n✓ Setup complete! Follow Option A to continue.")


def run_quick_demo(use_tensorflow=False):
    """Run a quick demonstration"""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import pickle
    
    print("\n1. Generating synthetic dataset (500 samples)...")
    
    # Generate simple synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Create features
    data = []
    for i in range(n_samples):
        label = np.random.choice([0, 1], p=[0.6, 0.4])
        
        if label == 1:  # Cognitive decline
            sample = {
                'speech_pause': np.random.normal(1.2, 0.3),
                'speech_rate': np.random.normal(2.5, 0.5),
                'vocab_diversity': np.random.normal(0.65, 0.1),
                'text_complexity': np.random.normal(0.55, 0.1),
                'tremor_score': np.random.normal(0.35, 0.1),
                'handwriting_consistency': np.random.normal(0.65, 0.1),
                'reaction_time': np.random.normal(550, 80),
                'memory_score': np.random.normal(0.60, 0.12),
                'hrv': np.random.normal(35, 10),
                'sleep_quality': np.random.normal(0.72, 0.08),
                'label': label
            }
        else:  # Healthy
            sample = {
                'speech_pause': np.random.normal(0.5, 0.2),
                'speech_rate': np.random.normal(4.0, 0.5),
                'vocab_diversity': np.random.normal(0.85, 0.08),
                'text_complexity': np.random.normal(0.75, 0.08),
                'tremor_score': np.random.normal(0.15, 0.05),
                'handwriting_consistency': np.random.normal(0.88, 0.06),
                'reaction_time': np.random.normal(350, 50),
                'memory_score': np.random.normal(0.90, 0.06),
                'hrv': np.random.normal(55, 10),
                'sleep_quality': np.random.normal(0.88, 0.05),
                'label': label
            }
        
        data.append(sample)
    
    df = pd.DataFrame(data)
    print(f"   ✓ Generated {len(df)} samples")
    print(f"   ✓ Healthy: {sum(df['label']==0)}, Decline: {sum(df['label']==1)}")
    
    # Save dataset
    df.to_csv('data/demo_dataset.csv', index=False)
    print(f"   ✓ Saved to: data/demo_dataset.csv")
    
    print("\n2. Training model...")
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    print("   ✓ Model trained")
    
    print("\n3. Evaluating model...")
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n   Accuracy: {accuracy:.2%}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Healthy', 'Cognitive Decline'],
                                digits=3))
    
    # Feature importance
    importance = model.feature_importances_
    feature_names = X.columns.tolist()
    
    print("\n   Top 5 Important Features:")
    indices = np.argsort(importance)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.3f}")
    
    print("\n4. Saving model...")
    
    with open('models/demo_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/demo_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("   ✓ Model saved to: models/demo_model.pkl")
    
    print("\n5. Testing inference on new patient...")
    
    # Test case: Patient with mild decline
    test_patient = pd.DataFrame([{
        'speech_pause': 0.95,
        'speech_rate': 3.0,
        'vocab_diversity': 0.70,
        'text_complexity': 0.62,
        'tremor_score': 0.28,
        'handwriting_consistency': 0.75,
        'reaction_time': 480,
        'memory_score': 0.68,
        'hrv': 42,
        'sleep_quality': 0.76
    }])
    
    test_scaled = scaler.transform(test_patient)
    prediction = model.predict(test_scaled)[0]
    probability = model.predict_proba(test_scaled)[0]
    
    print("\n   Test Patient Analysis:")
    print(f"   Prediction: {'Cognitive Decline Detected' if prediction == 1 else 'No Significant Decline'}")
    print(f"   Confidence: {max(probability)*100:.1f}%")
    print(f"   Risk Score: {probability[1]*100:.1f}%")
    
    if prediction == 1:
        print("\n   Recommendation: Consult neurologist for comprehensive evaluation")
    else:
        print("\n   Recommendation: Continue regular health monitoring")
    
    print("\n" + "=" * 70)
    print("✓ DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Check generated files in data/ and models/ directories")
    print("2. Use the complete system for production deployment")
    print("3. Integrate with your data pipeline")


if __name__ == "__main__":
    main()