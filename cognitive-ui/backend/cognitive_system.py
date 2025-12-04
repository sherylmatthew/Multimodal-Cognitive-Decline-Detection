"""
AI Cognitive Decline Detection System - Complete Training & Inference Pipeline
Multi-modal deep learning system for early detection of cognitive decline
"""

import numpy as np
import pandas as pd
import librosa
import cv2
import json
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import (Dense, Dropout, Input, LSTM, Conv1D, 
                                         MaxPooling1D, Flatten, Concatenate, 
                                         BatchNormalization, Conv2D, MaxPooling2D)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using classical ML models only.")


class SpeechFeatureExtractor:
    """Extract acoustic features from speech audio"""
    
    @staticmethod
    def extract_features(audio_path, sr=22050):
        """
        Extract comprehensive speech features
        Features include: MFCCs, pitch, energy, pause duration, speech rate
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, duration=30)
            
            features = {}
            
            # MFCCs (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Pitch (F0) features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            features['pitch_mean'] = np.mean(pitch_values) if pitch_values else 0
            features['pitch_std'] = np.std(pitch_values) if pitch_values else 0
            
            # Energy and zero-crossing rate
            energy = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = np.mean(energy)
            features['energy_std'] = np.std(energy)
            
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            
            # Pause detection (silence segments)
            intervals = librosa.effects.split(y, top_db=30)
            pause_durations = []
            for i in range(len(intervals) - 1):
                pause = (intervals[i+1][0] - intervals[i][1]) / sr
                pause_durations.append(pause)
            
            features['pause_mean'] = np.mean(pause_durations) if pause_durations else 0
            features['pause_count'] = len(pause_durations)
            
            # Speech rate (syllables per second approximation)
            features['speech_rate'] = len(intervals) / (len(y) / sr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            return features
            
        except Exception as e:
            print(f"Error extracting speech features: {e}")
            return None


class TextFeatureExtractor:
    """Extract linguistic features from written text"""
    
    @staticmethod
    def extract_features(text_content):
        """
        Extract text complexity and linguistic features
        Features: vocabulary richness, grammar complexity, coherence
        """
        try:
            words = text_content.lower().split()
            sentences = text_content.split('.')
            
            features = {}
            
            # Vocabulary diversity (Type-Token Ratio)
            unique_words = len(set(words))
            total_words = len(words)
            features['vocabulary_diversity'] = unique_words / total_words if total_words > 0 else 0
            
            # Average word length
            features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
            
            # Average sentence length
            features['avg_sentence_length'] = total_words / len(sentences) if sentences else 0
            
            # Word repetition rate
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            features['repetition_rate'] = repeated_words / len(word_counts) if word_counts else 0
            
            # Lexical density (content words vs total words)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            content_words = [w for w in words if w not in common_words]
            features['lexical_density'] = len(content_words) / total_words if total_words > 0 else 0
            
            # Complexity score (combination of factors)
            features['complexity_score'] = (
                features['vocabulary_diversity'] * 0.4 +
                (features['avg_word_length'] / 10) * 0.3 +
                (features['avg_sentence_length'] / 20) * 0.3
            )
            
            return features
            
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None


class HandwritingFeatureExtractor:
    """Extract features from handwriting images"""
    
    @staticmethod
    def extract_features(image_path):
        """
        Extract handwriting characteristics
        Features: tremor, pressure, spacing, consistency
        """
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            features = {}
            
            # Binarize image
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Edge detection for tremor analysis
            edges = cv2.Canny(binary, 50, 150)
            
            # Tremor score (edge irregularity)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                total_irregularity = 0
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    if area > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        total_irregularity += (1 - circularity)
                features['tremor_score'] = total_irregularity / len(contours)
            else:
                features['tremor_score'] = 0
            
            # Pressure estimation (ink density)
            ink_pixels = np.sum(binary > 0)
            total_pixels = binary.shape[0] * binary.shape[1]
            features['pressure_score'] = ink_pixels / total_pixels
            
            # Spacing consistency
            horizontal_projection = np.sum(binary, axis=1)
            line_gaps = []
            in_gap = False
            gap_size = 0
            
            for val in horizontal_projection:
                if val == 0:
                    if in_gap:
                        gap_size += 1
                    else:
                        in_gap = True
                        gap_size = 1
                else:
                    if in_gap and gap_size > 5:
                        line_gaps.append(gap_size)
                    in_gap = False
                    gap_size = 0
            
            features['spacing_consistency'] = np.std(line_gaps) if line_gaps else 0
            features['avg_spacing'] = np.mean(line_gaps) if line_gaps else 0
            
            # Stroke width variation
            distances = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            features['stroke_width_mean'] = np.mean(distances[distances > 0]) if np.any(distances > 0) else 0
            features['stroke_width_std'] = np.std(distances[distances > 0]) if np.any(distances > 0) else 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting handwriting features: {e}")
            return None


class VisualTestFeatureExtractor:
    """Extract features from visual cognition test results"""
    
    @staticmethod
    def extract_features(test_data):
        """
        Extract cognitive test performance features
        Features: reaction time, accuracy, pattern recognition
        """
        try:
            if isinstance(test_data, str):
                with open(test_data, 'r') as f:
                    data = json.load(f)
            else:
                data = test_data
            
            features = {}
            
            # Reaction time features
            if 'reaction_times' in data:
                rt = data['reaction_times']
                features['reaction_time_mean'] = np.mean(rt)
                features['reaction_time_std'] = np.std(rt)
                features['reaction_time_median'] = np.median(rt)
            
            # Accuracy metrics
            if 'correct_responses' in data and 'total_trials' in data:
                features['accuracy'] = data['correct_responses'] / data['total_trials']
            
            # Pattern recognition score
            features['pattern_recognition_score'] = data.get('pattern_score', 0.5)
            
            # Memory recall score
            features['memory_recall_score'] = data.get('memory_score', 0.5)
            
            # Attention span (time on task)
            features['attention_span'] = data.get('attention_duration', 60)
            
            # Error rate
            if 'errors' in data and 'total_trials' in data:
                features['error_rate'] = data['errors'] / data['total_trials']
            
            return features
            
        except Exception as e:
            print(f"Error extracting visual test features: {e}")
            return None


class PhysiologicalFeatureExtractor:
    """Extract features from physiological/health data"""
    
    @staticmethod
    def extract_features(physio_data):
        """
        Extract health metrics features
        Features: HRV, sleep quality, activity levels
        """
        try:
            if isinstance(physio_data, str):
                data = pd.read_csv(physio_data)
            else:
                data = physio_data
            
            features = {}
            
            # Heart rate variability
            if 'heart_rate' in data.columns:
                hr = data['heart_rate'].values
                features['hr_mean'] = np.mean(hr)
                features['hr_std'] = np.std(hr)
                features['hrv'] = np.std(np.diff(hr))  # Simplified HRV
            
            # Sleep quality metrics
            if 'sleep_duration' in data.columns:
                features['sleep_duration_mean'] = data['sleep_duration'].mean()
                features['sleep_efficiency'] = data.get('sleep_efficiency', pd.Series([0.8])).mean()
            
            # Physical activity
            if 'steps' in data.columns:
                features['daily_steps_mean'] = data['steps'].mean()
                features['activity_variance'] = data['steps'].std()
            
            # Vital signs stability
            if 'blood_pressure_systolic' in data.columns:
                features['bp_systolic_mean'] = data['blood_pressure_systolic'].mean()
                features['bp_stability'] = data['blood_pressure_systolic'].std()
            
            # Overall health score (composite)
            health_indicators = []
            if 'hr_mean' in features:
                health_indicators.append(1 - abs(features['hr_mean'] - 70) / 70)
            if 'sleep_efficiency' in features:
                health_indicators.append(features['sleep_efficiency'])
            
            features['overall_health_score'] = np.mean(health_indicators) if health_indicators else 0.5
            
            return features
            
        except Exception as e:
            print(f"Error extracting physiological features: {e}")
            return None


class MultiModalDataGenerator:
    """Generate synthetic multimodal dataset for training"""
    
    @staticmethod
    def generate_dataset(n_samples=1000, save_path='./data/'):
        """Generate synthetic dataset with realistic patterns"""
        
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Generate label (0: healthy, 1: cognitive decline)
            label = np.random.choice([0, 1], p=[0.6, 0.4])
            
            sample = {'label': label}
            
            # Speech features
            if label == 1:  # Cognitive decline
                sample.update({
                    'mfcc_mean_0': np.random.normal(-150, 30),
                    'pitch_mean': np.random.normal(150, 40),
                    'pause_mean': np.random.normal(1.2, 0.3),  # Longer pauses
                    'speech_rate': np.random.normal(2.5, 0.5),  # Slower
                    'energy_mean': np.random.normal(0.03, 0.01),
                })
            else:  # Healthy
                sample.update({
                    'mfcc_mean_0': np.random.normal(-100, 20),
                    'pitch_mean': np.random.normal(180, 30),
                    'pause_mean': np.random.normal(0.5, 0.2),  # Normal pauses
                    'speech_rate': np.random.normal(4.0, 0.5),  # Normal rate
                    'energy_mean': np.random.normal(0.05, 0.01),
                })
            
            # Text features
            if label == 1:
                sample.update({
                    'vocabulary_diversity': np.random.normal(0.65, 0.1),  # Lower
                    'avg_word_length': np.random.normal(4.2, 0.5),
                    'repetition_rate': np.random.normal(0.35, 0.1),  # Higher
                    'complexity_score': np.random.normal(0.55, 0.1),
                })
            else:
                sample.update({
                    'vocabulary_diversity': np.random.normal(0.85, 0.08),
                    'avg_word_length': np.random.normal(5.5, 0.5),
                    'repetition_rate': np.random.normal(0.15, 0.08),
                    'complexity_score': np.random.normal(0.75, 0.08),
                })
            
            # Handwriting features
            if label == 1:
                sample.update({
                    'tremor_score': np.random.normal(0.35, 0.1),  # Higher tremor
                    'pressure_score': np.random.normal(0.25, 0.08),
                    'spacing_consistency': np.random.normal(15, 5),  # Less consistent
                    'stroke_width_std': np.random.normal(2.5, 0.5),
                })
            else:
                sample.update({
                    'tremor_score': np.random.normal(0.15, 0.05),
                    'pressure_score': np.random.normal(0.35, 0.05),
                    'spacing_consistency': np.random.normal(8, 3),
                    'stroke_width_std': np.random.normal(1.5, 0.3),
                })
            
            # Visual test features
            if label == 1:
                sample.update({
                    'reaction_time_mean': np.random.normal(550, 80),  # Slower
                    'accuracy': np.random.normal(0.72, 0.1),  # Lower
                    'pattern_recognition_score': np.random.normal(0.65, 0.1),
                    'memory_recall_score': np.random.normal(0.60, 0.12),
                })
            else:
                sample.update({
                    'reaction_time_mean': np.random.normal(350, 50),
                    'accuracy': np.random.normal(0.92, 0.05),
                    'pattern_recognition_score': np.random.normal(0.88, 0.06),
                    'memory_recall_score': np.random.normal(0.90, 0.06),
                })
            
            # Physiological features
            if label == 1:
                sample.update({
                    'hrv': np.random.normal(35, 10),  # Lower variability
                    'sleep_efficiency': np.random.normal(0.72, 0.08),  # Poorer sleep
                    'daily_steps_mean': np.random.normal(4500, 1000),  # Less active
                    'overall_health_score': np.random.normal(0.65, 0.1),
                })
            else:
                sample.update({
                    'hrv': np.random.normal(55, 10),
                    'sleep_efficiency': np.random.normal(0.88, 0.05),
                    'daily_steps_mean': np.random.normal(8000, 1500),
                    'overall_health_score': np.random.normal(0.85, 0.08),
                })
            
            data.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save dataset
        Path(save_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{save_path}/multimodal_cognitive_dataset.csv', index=False)
        
        print(f"Generated dataset with {n_samples} samples")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")
        
        return df


class MultiModalCognitiveModel:
    """Complete multimodal model for cognitive decline detection"""
    
    def __init__(self, use_deep_learning=False):
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def build_deep_model(self, input_dim):
        """Build deep neural network for multimodal fusion"""
        
        inputs = Input(shape=(input_dim,))
        
        # Feature extraction layers
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=50):
        """Train the model"""
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.use_deep_learning:
            # Deep learning model
            self.model = self.build_deep_model(X_train_scaled.shape[1])
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_auc', mode='max')
            ]
            
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
        else:
            # Classical ML model (Gradient Boosting)
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            val_score = self.model.score(X_val_scaled, y_val)
            
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Validation accuracy: {val_score:.4f}")
            
            return None
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        if self.use_deep_learning:
            predictions = self.model.predict(X_scaled)
            return (predictions > 0.5).astype(int).flatten()
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_scaled = self.scaler.transform(X)
        
        if self.use_deep_learning:
            return self.model.predict(X_scaled).flatten()
        else:
            return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        print("\n=== Model Evaluation ===")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Cognitive Decline']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nAUC-ROC Score: {auc:.4f}")
        
        return {
            'accuracy': np.mean(y_pred == y_test),
            'auc': auc
        }
    
    def get_feature_importance(self, top_n=15):
        """Get feature importance scores"""
        if self.use_deep_learning:
            print("Feature importance not available for deep learning models")
            return None
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        print("\n=== Top Feature Importances ===")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.feature_columns[idx]}: {importance[idx]:.4f}")
        
        return list(zip([self.feature_columns[i] for i in indices], 
                       importance[indices]))
    
    def save_model(self, path='./models/'):
        """Save trained model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if self.use_deep_learning:
            self.model.save(f'{path}/cognitive_model.h5')
        else:
            with open(f'{path}/cognitive_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        
        with open(f'{path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{path}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path='./models/'):
        """Load trained model"""
        if self.use_deep_learning:
            self.model = load_model(f'{path}/cognitive_model.h5')
        else:
            with open(f'{path}/cognitive_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        
        with open(f'{path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{path}/feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        print("Model loaded successfully")


def main_training_pipeline():
    """Complete training pipeline"""
    
    print("=" * 60)
    print("COGNITIVE DECLINE DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate synthetic dataset
    print("\n[1/5] Generating synthetic dataset...")
    df = MultiModalDataGenerator.generate_dataset(n_samples=2000, save_path='./data/')
    
    # Step 2: Prepare data
    print("\n[2/5] Preparing data...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Step 3: Initialize model
    print("\n[3/5] Initializing model...")
    model = MultiModalCognitiveModel(use_deep_learning=TENSORFLOW_AVAILABLE)
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.train(X_train, y_train, validation_split=0.2, epochs=50)
    
    # Step 5: Evaluate model
    print("\n[5/5] Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # Feature importance
    if not model.use_deep_learning:
        model.get_feature_importance()
    
    # Save model
    model.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final AUC-ROC: {metrics['auc']:.4f}")
    print("=" * 60)


def inference_example():
    """Example inference on new data"""
    
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLE")
    print("=" * 60)
    
    # Load model
    model = MultiModalCognitiveModel(use_deep_learning=TENSORFLOW_AVAILABLE)
    model.load_model()
    
    # Create example input (simulating a patient with mild decline)
    example_data = pd.DataFrame([{
        'mfcc_mean_0': -130,
        'pitch_mean': 160,
        'pause_mean': 0.9,
        'speech_rate': 3.2,
        'energy_mean': 0.04,
        'vocabulary_diversity': 0.70,
        'avg_word_length': 4.8,
        'repetition_rate': 0.28,
        'complexity_score': 0.62,
        'tremor_score': 0.28,
        'pressure_score': 0.28,
        'spacing_consistency': 12,
        'stroke_width_std': 2.0,
        'reaction_time_mean': 480,
        'accuracy': 0.78,
        'pattern_recognition_score': 0.72,
        'memory_recall_score': 0.68,
        'hrv': 42,
        'sleep_efficiency': 0.76,
        'daily_steps_mean': 5500,
        'overall_health_score': 0.70
    }])
    
    # Make prediction
    prediction = model.predict(example_data)[0]
    probability = model.predict_proba(example_data)[0]
    
    print(f"\nPrediction: {'Cognitive Decline Detected' if prediction == 1 else 'No Significant Decline'}")
    print(f"Confidence: {probability * 100:.1f}%")
    print(f"Risk Level: {'High' if probability > 0.75 else 'Moderate' if probability > 0.5 else 'Low'}")


class InferenceSystem:
    """Real-time inference system for cognitive decline detection"""
    
    def __init__(self, model_path='./models/'):
        self.model = MultiModalCognitiveModel(use_deep_learning=TENSORFLOW_AVAILABLE)
        try:
            self.model.load_model(model_path)
            print("✓ Model loaded successfully")
        except:
            print("⚠ No trained model found. Please run training first.")
    
    def analyze_patient(self, data_dict):
        """
        Analyze patient data and return comprehensive results
        
        Parameters:
        data_dict: Dictionary with modality data
            - speech_path: path to audio file
            - text_content: text string or path
            - handwriting_path: path to image
            - visual_data: dict or path to JSON
            - physio_data: DataFrame or path to CSV
        """
        features = {}
        
        # Extract features from each modality
        if 'speech_path' in data_dict and data_dict['speech_path']:
            speech_features = SpeechFeatureExtractor.extract_features(data_dict['speech_path'])
            if speech_features:
                features.update(speech_features)
        
        if 'text_content' in data_dict and data_dict['text_content']:
            text_features = TextFeatureExtractor.extract_features(data_dict['text_content'])
            if text_features:
                features.update(text_features)
        
        if 'handwriting_path' in data_dict and data_dict['handwriting_path']:
            hw_features = HandwritingFeatureExtractor.extract_features(data_dict['handwriting_path'])
            if hw_features:
                features.update(hw_features)
        
        if 'visual_data' in data_dict and data_dict['visual_data']:
            visual_features = VisualTestFeatureExtractor.extract_features(data_dict['visual_data'])
            if visual_features:
                features.update(visual_features)
        
        if 'physio_data' in data_dict and data_dict['physio_data']:
            physio_features = PhysiologicalFeatureExtractor.extract_features(data_dict['physio_data'])
            if physio_features:
                features.update(physio_features)
        
        # Create DataFrame with all required features
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.model.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value for missing features
        
        feature_df = feature_df[self.model.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(feature_df)[0]
        probability = self.model.predict_proba(feature_df)[0]
        
        # Generate detailed results
        results = {
            'prediction': 'Cognitive Decline Detected' if prediction == 1 else 'No Significant Decline',
            'confidence': probability * 100,
            'risk_level': 'High' if probability > 0.75 else 'Moderate' if probability > 0.5 else 'Low',
            'features_analyzed': len(features),
            'detailed_scores': features,
            'interpretation': self._generate_interpretation(prediction, probability, features)
        }
        
        return results
    
    def _generate_interpretation(self, prediction, probability, features):
        """Generate clinical interpretation"""
        if prediction == 1:
            interpretation = f"Analysis indicates patterns consistent with cognitive decline (confidence: {probability*100:.1f}%). "
            
            concerns = []
            if 'pause_mean' in features and features['pause_mean'] > 0.8:
                concerns.append("increased speech pauses")
            if 'vocabulary_diversity' in features and features['vocabulary_diversity'] < 0.7:
                concerns.append("reduced vocabulary diversity")
            if 'tremor_score' in features and features['tremor_score'] > 0.25:
                concerns.append("handwriting tremor")
            if 'reaction_time_mean' in features and features['reaction_time_mean'] > 450:
                concerns.append("slower reaction times")
            
            if concerns:
                interpretation += f"Key indicators: {', '.join(concerns)}. "
            
            interpretation += "Recommend comprehensive neurological evaluation and regular monitoring."
        else:
            interpretation = f"Cognitive performance within normal ranges (confidence: {(1-probability)*100:.1f}%). "
            interpretation += "Continue healthy lifestyle practices and regular cognitive engagement."
        
        return interpretation


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print(" " * 15 + "COGNITIVE DECLINE DETECTION SYSTEM")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Run training pipeline
        main_training_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == 'inference':
        # Run inference example
        inference_example()
    else:
        # Interactive mode
        print("\nSelect mode:")
        print("1. Train new model")
        print("2. Run inference example")
        print("3. Both (train + inference)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            main_training_pipeline()
        elif choice == '2':
            inference_example()
        elif choice == '3':
            main_training_pipeline()
            inference_example()
        else:
            print("Invalid choice. Running training by default...")
            main_training_pipeline()
    
    print("\n" + "=" * 70)
    print("✓ System ready for deployment!")
    print("=" * 70)