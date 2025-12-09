"""
Supabase Client Helper - Cloud Database Integration
✅ Replaces SQLite with Supabase PostgreSQL
✅ Replaces JSON files with JSONB storage
"""

import os
import json
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Try to import supabase
supabase_client = None
USE_SUPABASE = False

try:
    from supabase import create_client, Client
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        USE_SUPABASE = True
        logger.info("✅ Supabase connected!")
    else:
        logger.warning("⚠️ Supabase credentials missing - using local storage")
except ImportError:
    logger.warning("⚠️ Supabase not installed - using local storage")
except Exception as e:
    logger.warning(f"⚠️ Supabase connection failed: {e}")


def get_client():
    """Get Supabase client"""
    return supabase_client


def is_cloud_enabled():
    """Check if cloud storage is available"""
    return USE_SUPABASE and supabase_client is not None


# ============================================================
# FLIGHTS TABLE OPERATIONS
# ============================================================

def insert_flight(flight_data: dict):
    """Insert a flight record"""
    if not is_cloud_enabled():
        return None
    
    try:
        result = supabase_client.table('flights').insert(flight_data).execute()
        return result.data
    except Exception as e:
        logger.error(f"❌ Insert flight failed: {e}")
        return None


def get_flights_by_route(origin: str, destination: str, limit: int = 100):
    """Get flights for a specific route"""
    if not is_cloud_enabled():
        return []
    
    try:
        result = supabase_client.table('flights')\
            .select('*')\
            .eq('origin', origin)\
            .eq('destination', destination)\
            .order('flight_date', desc=True)\
            .limit(limit)\
            .execute()
        return result.data
    except Exception as e:
        logger.error(f"❌ Get flights failed: {e}")
        return []


def get_flights_by_date(flight_date: str):
    """Get all flights for a specific date"""
    if not is_cloud_enabled():
        return []
    
    try:
        result = supabase_client.table('flights')\
            .select('*')\
            .eq('flight_date', flight_date)\
            .execute()
        return result.data
    except Exception as e:
        logger.error(f"❌ Get flights by date failed: {e}")
        return []


# ============================================================
# APP_DATA TABLE (JSON STORAGE)
# ============================================================

def save_json_data(key: str, data: dict):
    """Save JSON data to cloud"""
    if not is_cloud_enabled():
        return False
    
    try:
        # Upsert - insert or update
        result = supabase_client.table('app_data').upsert({
            'key': key,
            'data': data,
            'updated_at': 'now()'
        }).execute()
        return True
    except Exception as e:
        logger.error(f"❌ Save JSON failed: {e}")
        return False


def load_json_data(key: str, default=None):
    """Load JSON data from cloud"""
    if not is_cloud_enabled():
        return default
    
    try:
        result = supabase_client.table('app_data')\
            .select('data')\
            .eq('key', key)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['data']
        return default
    except Exception as e:
        logger.error(f"❌ Load JSON failed: {e}")
        return default


# ============================================================
# SPECIFIC DATA HELPERS
# ============================================================

def save_q_table(q_table: dict):
    """Save RL Q-table to cloud"""
    return save_json_data('rl_q_table', q_table)


def load_q_table():
    """Load RL Q-table from cloud"""
    return load_json_data('rl_q_table', {})


def save_rl_metrics(metrics: dict):
    """Save RL metrics to cloud"""
    return save_json_data('rl_metrics', metrics)


def load_rl_metrics():
    """Load RL metrics from cloud"""
    return load_json_data('rl_metrics', {})


def save_predictions(predictions: dict):
    """Save predictions to cloud"""
    return save_json_data('predictions', predictions)


def load_predictions():
    """Load predictions from cloud"""
    return load_json_data('predictions', {})


# ============================================================
# INITIALIZATION
# ============================================================

def create_tables_if_needed():
    """
    Create tables in Supabase.
    NOTE: This should be run once manually via Supabase SQL editor.
    """
    sql = """
    -- Flights table
    CREATE TABLE IF NOT EXISTS flights (
        id SERIAL PRIMARY KEY,
        flight_date TEXT,
        flight_number TEXT,
        airline_code TEXT,
        airline_name TEXT,
        origin TEXT,
        destination TEXT,
        scheduled_departure TEXT,
        actual_departure TEXT,
        scheduled_arrival TEXT,
        actual_arrival TEXT,
        departure_delay INTEGER,
        arrival_delay INTEGER,
        status TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- App data table (for JSON storage)
    CREATE TABLE IF NOT EXISTS app_data (
        key TEXT PRIMARY KEY,
        data JSONB,
        updated_at TIMESTAMP DEFAULT NOW()
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_flights_route ON flights(origin, destination);
    CREATE INDEX IF NOT EXISTS idx_flights_date ON flights(flight_date);
    CREATE INDEX IF NOT EXISTS idx_flights_number ON flights(flight_number);
    """
    print("Run this SQL in Supabase SQL Editor:")
    print(sql)
    return sql
