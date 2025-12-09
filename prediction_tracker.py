"""
Prediction Tracker - Stores predictions for later verification
✅ Cloud storage (Supabase) with local fallback
✅ Saves predictions when made
✅ Tracks flight + date + predicted probability
"""

import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_FILE = 'pending_predictions.json'

# Try to import Supabase
try:
    from supabase_client import is_cloud_enabled, save_predictions as cloud_save, load_predictions as cloud_load
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False


def load_predictions():
    """Load predictions from cloud or file"""
    # Try cloud first
    if CLOUD_AVAILABLE and is_cloud_enabled():
        try:
            data = cloud_load()
            if data:
                logger.info(f"☁️ Loaded {len(data)} predictions from cloud")
                return data
        except Exception as e:
            logger.warning(f"⚠️ Cloud load failed: {e}")
    
    # Fallback to local file
    if not os.path.exists(PREDICTIONS_FILE):
        return {}
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Could not load predictions: {e}")
        return {}


def save_predictions(predictions):
    """Save predictions to cloud and file"""
    # Save to cloud
    if CLOUD_AVAILABLE and is_cloud_enabled():
        try:
            cloud_save(predictions)
            logger.info(f"☁️ Saved {len(predictions)} predictions to cloud")
        except Exception as e:
            logger.warning(f"⚠️ Cloud save failed: {e}")
    
    # Also save to local file as backup
    try:
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)
    except Exception as e:
        logger.error(f"❌ Could not save predictions: {e}")


def store_prediction(flight_number, origin, destination, flight_date, 
                     predicted_delay_prob, rl_info):
    """
    Store a prediction for later verification
    
    Args:
        flight_number: Flight number (e.g., "6E123")
        origin: Origin airport IATA code
        destination: Destination airport IATA code
        flight_date: Flight date (YYYY-MM-DD)
        predicted_delay_prob: Predicted delay probability (0-100)
        rl_info: RL metadata for learning
    """
    predictions = load_predictions()
    
    # Create unique key
    key = f"{flight_number}_{origin}_{destination}_{flight_date}"
    
    predictions[key] = {
        'flight_number': flight_number,
        'origin': origin,
        'destination': destination,
        'flight_date': flight_date,
        'predicted_delay_prob': predicted_delay_prob,
        'rl_info': rl_info,
        'prediction_timestamp': datetime.now().isoformat(),
        'verified': False
    }
    
    save_predictions(predictions)
    logger.info(f"📝 Stored prediction: {key} → {predicted_delay_prob}% delay")
    
    return key


def get_pending_predictions(flight_date=None):
    """
    Get predictions that haven't been verified yet
    
    Args:
        flight_date: Optional filter by date (YYYY-MM-DD)
        
    Returns:
        dict of pending predictions
    """
    predictions = load_predictions()
    
    pending = {}
    for key, pred in predictions.items():
        if pred.get('verified'):
            continue
        if flight_date and pred.get('flight_date') != flight_date:
            continue
        pending[key] = pred
    
    return pending


def mark_prediction_verified(key, actual_delayed, reward=None):
    """
    Mark a prediction as verified after checking actual outcome
    
    Args:
        key: Prediction key
        actual_delayed: Boolean - was the flight actually delayed?
        reward: Optional RL reward value
    """
    predictions = load_predictions()
    
    if key in predictions:
        predictions[key]['verified'] = True
        predictions[key]['actual_delayed'] = actual_delayed
        predictions[key]['verification_timestamp'] = datetime.now().isoformat()
        if reward is not None:
            predictions[key]['rl_reward'] = reward
        
        save_predictions(predictions)
        logger.info(f"✅ Verified: {key} → Actual: {'DELAYED' if actual_delayed else 'ON-TIME'}")
    
    return predictions.get(key)


def cleanup_old_predictions(days_old=30):
    """Remove predictions older than specified days"""
    predictions = load_predictions()
    cutoff = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    
    cleaned = {}
    removed = 0
    
    for key, pred in predictions.items():
        try:
            pred_time = datetime.fromisoformat(pred['prediction_timestamp']).timestamp()
            if pred_time > cutoff:
                cleaned[key] = pred
            else:
                removed += 1
        except:
            cleaned[key] = pred
    
    save_predictions(cleaned)
    logger.info(f"🧹 Cleaned up {removed} old predictions")
    
    return removed


def get_stats():
    """Get prediction tracking statistics"""
    predictions = load_predictions()
    
    total = len(predictions)
    verified = sum(1 for p in predictions.values() if p.get('verified'))
    pending = total - verified
    
    correct = 0
    for p in predictions.values():
        if p.get('verified'):
            pred_high = p.get('predicted_delay_prob', 50) > 50
            actual = p.get('actual_delayed', False)
            if pred_high == actual:
                correct += 1
    
    accuracy = (correct / verified * 100) if verified > 0 else 0
    
    return {
        'total_predictions': total,
        'verified': verified,
        'pending': pending,
        'accuracy': round(accuracy, 1)
    }
