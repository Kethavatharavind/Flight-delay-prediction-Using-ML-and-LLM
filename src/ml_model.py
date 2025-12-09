"""
Flight Delay Prediction ML Model - ENHANCED VERSION
✅ Uses india_data.db ONLY
✅ More features for higher accuracy
✅ Better hyperparameters
✅ Cross-validation for robust training
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
import logging

# Try importing ML libraries
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not installed. Run: pip install scikit-learn xgboost")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration - using new folder structure
DB_NAME = os.path.join(PROJECT_ROOT, 'data', 'india_data.db')
MODEL_FILE = os.path.join(PROJECT_ROOT, 'models', 'delay_model.pkl')
ENCODER_FILE = os.path.join(PROJECT_ROOT, 'models', 'label_encoders.pkl')


class FlightDelayMLModel:
    """
    Enhanced XGBoost + Random Forest ensemble model
    for flight delay prediction
    """
    
    def __init__(self):
        self.model = None
        self.rf_model = None  # Secondary model for ensemble
        self.encoders = {}
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.is_trained = False
        
        # Enhanced feature set
        self.feature_columns = [
            'origin_encoded', 'destination_encoded', 'airline_encoded',
            'hour_of_day', 'day_of_week', 'month', 'is_weekend',
            'is_morning_rush', 'is_evening_rush', 'is_night_flight',
            'route_delay_history'  # Historical delay rate for this route
        ]
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if exists"""
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
                with open(MODEL_FILE, 'rb') as f:
                    saved_data = pickle.load(f)
                    if isinstance(saved_data, dict):
                        self.model = saved_data.get('xgb_model')
                        self.rf_model = saved_data.get('rf_model')
                        self.scaler = saved_data.get('scaler', self.scaler)
                    else:
                        self.model = saved_data
                with open(ENCODER_FILE, 'rb') as f:
                    self.encoders = pickle.load(f)
                self.is_trained = True
                logger.info("✅ Loaded pre-trained model")
        except Exception as e:
            logger.warning(f"⚠️ Could not load model: {e}")
    
    def _save_model(self):
        """Save trained models"""
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'xgb_model': self.model,
                    'rf_model': self.rf_model,
                    'scaler': self.scaler
                }, f)
            with open(ENCODER_FILE, 'wb') as f:
                pickle.dump(self.encoders, f)
            logger.info(f"✅ Model saved to {MODEL_FILE}")
        except Exception as e:
            logger.error(f"❌ Could not save model: {e}")
    
    def load_data_from_db(self):
        """Load training data from india_data.db"""
        if not os.path.exists(DB_NAME):
            logger.error(f"❌ Database {DB_NAME} not found!")
            return None
        
        conn = sqlite3.connect(DB_NAME)
        
        # Load from flights table
        query = """
            SELECT 
                flight_date,
                flight_number,
                airline_code,
                origin,
                destination,
                scheduled_departure,
                departure_delay,
                arrival_delay,
                status
            FROM flights
            WHERE status IN ('on_time', 'delayed', 'cancelled')
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Also get route-level statistics
        route_stats_query = """
            SELECT 
                origin,
                destination,
                AVG(CASE WHEN status = 'delayed' THEN 1 ELSE 0 END) as route_delay_rate,
                COUNT(*) as route_count
            FROM flights
            WHERE status IN ('on_time', 'delayed', 'cancelled')
            GROUP BY origin, destination
        """
        route_stats = pd.read_sql_query(route_stats_query, conn)
        
        conn.close()
        
        # Merge route stats
        df = df.merge(route_stats, on=['origin', 'destination'], how='left')
        
        logger.info(f"✅ Loaded {len(df)} records from {DB_NAME}")
        return df
    
    def prepare_features(self, df, is_training=True):
        """Extract and encode features for training"""
        data = df.copy()
        
        # Parse date and time
        data['flight_date'] = pd.to_datetime(data['flight_date'], errors='coerce')
        data['day_of_week'] = data['flight_date'].dt.dayofweek
        data['month'] = data['flight_date'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Extract hour from scheduled_departure
        def extract_hour(time_str):
            if pd.isna(time_str) or not time_str:
                return 12
            try:
                return int(str(time_str).split(':')[0])
            except:
                return 12
        
        data['hour_of_day'] = data['scheduled_departure'].apply(extract_hour)
        
        # Time-based features
        data['is_morning_rush'] = ((data['hour_of_day'] >= 6) & (data['hour_of_day'] <= 9)).astype(int)
        data['is_evening_rush'] = ((data['hour_of_day'] >= 17) & (data['hour_of_day'] <= 20)).astype(int)
        data['is_night_flight'] = ((data['hour_of_day'] >= 22) | (data['hour_of_day'] <= 5)).astype(int)
        
        # Route delay history
        data['route_delay_history'] = data['route_delay_rate'].fillna(0.3) * 100
        
        # Fill missing categorical values
        data['airline_code'] = data['airline_code'].fillna('UNKNOWN')
        data['origin'] = data['origin'].fillna('UNKNOWN')
        data['destination'] = data['destination'].fillna('UNKNOWN')
        
        # Encode categorical variables
        for col in ['origin', 'destination', 'airline_code']:
            encoded_col = f'{col}_encoded'
            if is_training:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    # Add 'UNKNOWN' to handle unseen values
                    unique_vals = data[col].astype(str).unique().tolist()
                    if 'UNKNOWN' not in unique_vals:
                        unique_vals.append('UNKNOWN')
                    self.encoders[col].fit(unique_vals)
                data[encoded_col] = self.encoders[col].transform(data[col].astype(str))
            else:
                # Handle unseen labels
                data[col] = data[col].apply(
                    lambda x: str(x) if str(x) in self.encoders[col].classes_ else 'UNKNOWN'
                )
                data[encoded_col] = self.encoders[col].transform(data[col].astype(str))
        
        # Create target: 1 = delayed/cancelled, 0 = on_time
        if 'status' in data.columns:
            data['target'] = (data['status'] != 'on_time').astype(int)
        
        return data
    
    def train(self, test_size=0.2):
        """Train XGBoost + Random Forest ensemble"""
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available!")
            return False
        
        # Load data
        df = self.load_data_from_db()
        if df is None or len(df) < 50:
            logger.error("❌ Not enough data to train! Need at least 50 records.")
            return False
        
        # Reset encoders for fresh training with new data
        self.encoders = {}
        logger.info("🔄 Reset encoders for fresh training")
        
        # Prepare features
        data = self.prepare_features(df, is_training=True)
        
        # Feature columns
        feature_cols = [
            'origin_encoded', 'destination_encoded', 'airline_code_encoded',
            'hour_of_day', 'day_of_week', 'month', 'is_weekend',
            'is_morning_rush', 'is_evening_rush', 'is_night_flight',
            'route_delay_history'
        ]
        
        X = data[feature_cols]
        y = data['target']
        
        # Drop any rows with NaN
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"📊 Training on {len(X)} samples")
        logger.info(f"📊 Delayed/Cancelled: {y.sum()} ({100*y.mean():.1f}%)")
        logger.info(f"📊 On-time: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost with better hyperparameters
        print("\n🔄 Training XGBoost...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle imbalance
        )
        self.model.fit(X_train_scaled, y_train, verbose=False)
        
        # Train Random Forest as secondary model
        print("🔄 Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Ensemble predictions (average of both models)
        xgb_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        rf_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_proba = (xgb_proba * 0.6 + rf_proba * 0.4)  # Weight XGBoost higher
        y_pred = (ensemble_proba > 0.5).astype(int)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Also get individual model scores
        xgb_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
        rf_acc = accuracy_score(y_test, self.rf_model.predict(X_test_scaled))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        print("\n" + "=" * 60)
        print("📈 MODEL PERFORMANCE")
        print("=" * 60)
        print(f"XGBoost Accuracy:     {xgb_acc:.2%}")
        print(f"Random Forest Accuracy: {rf_acc:.2%}")
        print(f"Ensemble Accuracy:    {accuracy:.2%}")
        print(f"Cross-Val Mean:       {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
        print("=" * 60)
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))
        
        # Feature importance
        print("\n📊 Top Feature Importances:")
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importances.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        self.is_trained = True
        self._save_model()
        
        return True
    
    def predict_delay_probability(self, origin, destination, airline_code, 
                                   departure_time, flight_date):
        """
        Predict delay probability using ensemble
        """
        if not self.is_trained or self.model is None:
            return {
                'probability_delay': None,
                'confidence': 'NO_MODEL',
                'message': 'Model not trained. Run train() first.'
            }
        
        try:
            # Parse date
            date_obj = datetime.strptime(flight_date, '%Y-%m-%d')
            day_of_week = date_obj.weekday()
            month = date_obj.month
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Parse time
            try:
                hour_of_day = int(str(departure_time).split(':')[0].split('T')[-1])
            except:
                hour_of_day = 12
            
            is_morning_rush = 1 if 6 <= hour_of_day <= 9 else 0
            is_evening_rush = 1 if 17 <= hour_of_day <= 20 else 0
            is_night_flight = 1 if hour_of_day >= 22 or hour_of_day <= 5 else 0
            
            # Encode categorical features - handle unseen values gracefully
            def safe_encode(value, encoder_name):
                encoder = self.encoders.get(encoder_name)
                if encoder is None:
                    return 0
                value_str = str(value).upper() if value else 'UNKNOWN'
                
                # Try exact match
                if value_str in encoder.classes_:
                    return encoder.transform([value_str])[0]
                
                # Try UNKNOWN
                if 'UNKNOWN' in encoder.classes_:
                    return encoder.transform(['UNKNOWN'])[0]
                
                # Fallback: return 0 (most common class index)
                return 0
            
            origin_enc = safe_encode(origin, 'origin')
            dest_enc = safe_encode(destination, 'destination')
            airline_enc = safe_encode(airline_code, 'airline_code')
            
            # Get route delay history from database if available
            route_delay_history = 30  # Default
            try:
                if os.path.exists(DB_NAME):
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT AVG(CASE WHEN status = 'delayed' THEN 100 ELSE 0 END)
                        FROM flights WHERE origin = ? AND destination = ?
                    """, (origin, destination))
                    result = cursor.fetchone()
                    if result and result[0]:
                        route_delay_history = result[0]
                    conn.close()
            except:
                pass
            
            # Create feature array
            features = np.array([[
                origin_enc, dest_enc, airline_enc,
                hour_of_day, day_of_week, month, is_weekend,
                is_morning_rush, is_evening_rush, is_night_flight,
                route_delay_history
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Ensemble prediction
            xgb_prob = self.model.predict_proba(features_scaled)[0][1]
            rf_prob = self.rf_model.predict_proba(features_scaled)[0][1] if self.rf_model else xgb_prob
            
            delay_prob = (xgb_prob * 0.6 + rf_prob * 0.4) * 100
            
            # Determine confidence based on how extreme the prediction is
            if delay_prob > 75 or delay_prob < 25:
                confidence = 'HIGH'
            elif delay_prob > 60 or delay_prob < 40:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'probability_delay': round(delay_prob, 1),
                'probability_on_time': round(100 - delay_prob, 1),
                'confidence': confidence,
                'model': 'XGBoost+RF Ensemble',
                'xgb_prob': round(xgb_prob * 100, 1),
                'rf_prob': round(rf_prob * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {
                'probability_delay': None,
                'confidence': 'ERROR',
                'message': str(e)
            }
    
    def get_stats(self):
        """Get model statistics"""
        return {
            'is_trained': self.is_trained,
            'model_type': 'XGBoost + Random Forest Ensemble',
            'model_file': MODEL_FILE,
            'features': self.feature_columns,
            'db_source': DB_NAME
        }


# Singleton instance
_ml_model = None


def get_ml_model():
    """Get or create the ML model instance"""
    global _ml_model
    if _ml_model is None:
        _ml_model = FlightDelayMLModel()
    return _ml_model


def predict_with_ml(origin, destination, airline_code, departure_time, flight_date):
    """Convenience function for ML prediction"""
    model = get_ml_model()
    return model.predict_delay_probability(
        origin, destination, airline_code, departure_time, flight_date
    )


def train_model():
    """Train or retrain the model"""
    model = get_ml_model()
    return model.train()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 ENHANCED FLIGHT DELAY ML MODEL TRAINER")
    print("=" * 60)
    print(f"📂 Database: {DB_NAME}")
    print(f"💾 Model Output: {MODEL_FILE}")
    print("🔧 Models: XGBoost + Random Forest Ensemble")
    print("=" * 60 + "\n")
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not installed!")
        print("Run: pip install scikit-learn xgboost")
        exit(1)
    
    model = FlightDelayMLModel()
    
    if model.train():
        print("\n✅ Training complete!")
        
        # Test prediction
        print("\n" + "=" * 60)
        print("🧪 TEST PREDICTION")
        print("=" * 60)
        
        result = model.predict_delay_probability(
            origin='DEL',
            destination='BOM',
            airline_code='6E',
            departure_time='14:30',
            flight_date='2025-12-15'
        )
        
        print(f"Route: DEL → BOM")
        print(f"XGBoost Prob: {result.get('xgb_prob')}%")
        print(f"RF Prob: {result.get('rf_prob')}%")
        print(f"Ensemble: {result.get('probability_delay')}%")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print("\n❌ Training failed! Make sure india_data.db has enough data.")
