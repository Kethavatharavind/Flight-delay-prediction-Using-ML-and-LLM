"""
Weather Data Backfill Script
✅ Adds weather data to existing flights in india_data.db
✅ Uses Open-Meteo API (free, no key required)
✅ Stores origin and destination weather for each flight

Run this after update_latest_data.py to enrich flight data with weather.
"""

import sqlite3
import os
import sys
import requests
from datetime import datetime, timedelta
import time
import logging

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

DB_NAME = os.path.join(PROJECT_ROOT, 'data', 'india_data.db')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Airport coordinates map (same as data_fetcher.py)
IATA_LOCATION_MAP = {
    # India
    "DEL": {"city": "Delhi", "lat": 28.5562, "lon": 77.1000},
    "BOM": {"city": "Mumbai", "lat": 19.0896, "lon": 72.8656},
    "BLR": {"city": "Bangalore", "lat": 13.1986, "lon": 77.7066},
    "HYD": {"city": "Hyderabad", "lat": 17.2403, "lon": 78.4294},
    "MAA": {"city": "Chennai", "lat": 12.9941, "lon": 80.1709},
    "CCU": {"city": "Kolkata", "lat": 22.6520, "lon": 88.4463},
    "COK": {"city": "Kochi", "lat": 10.1520, "lon": 76.4019},
    "PNQ": {"city": "Pune", "lat": 18.5793, "lon": 73.9089},
    "GAU": {"city": "Guwahati", "lat": 26.1061, "lon": 91.5859},
    "AMD": {"city": "Ahmedabad", "lat": 23.0772, "lon": 72.6347},
    "GOI": {"city": "Goa", "lat": 15.3808, "lon": 73.8314},
    "JAI": {"city": "Jaipur", "lat": 26.8242, "lon": 75.8122},
    "LKO": {"city": "Lucknow", "lat": 26.7606, "lon": 80.8893},
    "PAT": {"city": "Patna", "lat": 25.5913, "lon": 85.0880},
    "IXC": {"city": "Chandigarh", "lat": 30.6735, "lon": 76.7885},
    "VNS": {"city": "Varanasi", "lat": 25.4524, "lon": 82.8593},
    "TRV": {"city": "Trivandrum", "lat": 8.4821, "lon": 76.9199},
    "IXB": {"city": "Bagdogra", "lat": 26.6812, "lon": 88.3286},
    "SXR": {"city": "Srinagar", "lat": 33.9871, "lon": 74.7742},
    "IXR": {"city": "Ranchi", "lat": 23.3143, "lon": 85.3217},
    "BBI": {"city": "Bhubaneswar", "lat": 20.2444, "lon": 85.8178},
    "IXE": {"city": "Mangalore", "lat": 12.9612, "lon": 74.8900},
    "NAG": {"city": "Nagpur", "lat": 21.0922, "lon": 79.0472},
    "IDR": {"city": "Indore", "lat": 22.7217, "lon": 75.8011},
    "RPR": {"city": "Raipur", "lat": 21.1804, "lon": 81.7387},
    "VTZ": {"city": "Visakhapatnam", "lat": 17.7212, "lon": 83.2245},
    "CJB": {"city": "Coimbatore", "lat": 11.0300, "lon": 77.0434},
    # US airports
    "JFK": {"city": "New York", "lat": 40.6413, "lon": -73.7781},
    "LAX": {"city": "Los Angeles", "lat": 33.9416, "lon": -118.4085},
    "ORD": {"city": "Chicago", "lat": 41.9742, "lon": -87.9073},
    "ATL": {"city": "Atlanta", "lat": 33.6407, "lon": -84.4277},
    "DFW": {"city": "Dallas", "lat": 32.8998, "lon": -97.0403},
    # International
    "DXB": {"city": "Dubai", "lat": 25.2532, "lon": 55.3657},
    "SIN": {"city": "Singapore", "lat": 1.3644, "lon": 103.9915},
    "LHR": {"city": "London", "lat": 51.4700, "lon": -0.4543},
}

# Weather code descriptions
WEATHER_CODES = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Rain showers", 82: "Heavy rain showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm"
}


def init_weather_columns(conn):
    """Add weather columns to flights table if not exist"""
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(flights)")
    columns = [row[1] for row in cursor.fetchall()]
    
    new_columns = [
        ("origin_weather", "TEXT"),
        ("origin_temp_c", "REAL"),
        ("origin_precip_prob", "INTEGER"),
        ("origin_wind_kph", "REAL"),
        ("dest_weather", "TEXT"),
        ("dest_temp_c", "REAL"),
        ("dest_precip_prob", "INTEGER"),
        ("dest_wind_kph", "REAL"),
        ("inferred_delay_reason", "TEXT")  # weather, nas, carrier, late_aircraft
    ]
    
    for col_name, col_type in new_columns:
        if col_name not in columns:
            cursor.execute(f"ALTER TABLE flights ADD COLUMN {col_name} {col_type}")
            logger.info(f"✅ Added column: {col_name}")
    
    conn.commit()


def get_historical_weather(airport_code, date_str, time_str):
    """
    Fetch HISTORICAL weather for a past date using Open-Meteo Archive API.
    For future dates, uses regular forecast API.
    """
    location = IATA_LOCATION_MAP.get(airport_code.upper())
    if not location:
        return None
    
    try:
        # Parse the date
        flight_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now()
        
        # Parse hour from time (e.g., "14:30:00" -> 14)
        hour = 12  # default
        if time_str:
            try:
                hour = int(time_str.split(':')[0])
            except:
                hour = 12
        
        # Use archive API for past dates, forecast API for recent/future
        if flight_date.date() < (today - timedelta(days=5)).date():
            # Historical weather (archive API) - uses precipitation (mm) not probability
            url = "https://archive-api.open-meteo.com/v1/archive"
            hourly_params = 'temperature_2m,precipitation,windspeed_10m,weathercode'
        else:
            # Recent/future weather (forecast API) - has precipitation_probability
            url = "https://api.open-meteo.com/v1/forecast"
            hourly_params = 'temperature_2m,precipitation_probability,precipitation,windspeed_10m,weathercode'
        
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'hourly': hourly_params,
            'start_date': date_str,
            'end_date': date_str,
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        
        if not times:
            return None
        
        # Find closest hour
        temps = hourly.get('temperature_2m', [])
        precip_probs = hourly.get('precipitation_probability', [])  # Only for forecast API
        precip_mm = hourly.get('precipitation', [])  # Actual precipitation in mm
        wind_speeds = hourly.get('windspeed_10m', [])
        weather_codes = hourly.get('weathercode', [])
        
        # Find index closest to our hour
        closest_idx = min(hour, len(times) - 1)
        
        weather_code = weather_codes[closest_idx] if closest_idx < len(weather_codes) else 0
        condition = WEATHER_CODES.get(weather_code, "Unknown")
        
        # Get precipitation - use probability if available, otherwise convert mm to estimated %
        precip_value = None
        if precip_probs and closest_idx < len(precip_probs) and precip_probs[closest_idx] is not None:
            precip_value = precip_probs[closest_idx]
        elif precip_mm and closest_idx < len(precip_mm) and precip_mm[closest_idx] is not None:
            # Convert precipitation mm to rough probability (0mm=0%, 5mm+=100%)
            precip_value = min(100, int(precip_mm[closest_idx] * 20))
        
        return {
            "condition": condition,
            "temp_c": round(temps[closest_idx], 1) if closest_idx < len(temps) and temps[closest_idx] else None,
            "precip_prob": precip_value,
            "wind_kph": round(wind_speeds[closest_idx], 1) if closest_idx < len(wind_speeds) and wind_speeds[closest_idx] else None,
            "weather_code": weather_code
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Weather fetch failed for {airport_code} on {date_str}: {e}")
        return None


def infer_delay_reason(origin_weather, dest_weather, status, delay_minutes):
    """
    Infer the likely delay reason based on weather conditions.
    Returns: 'weather', 'nas', 'carrier', 'late_aircraft', or None
    """
    if status == 'on_time' or (delay_minutes and delay_minutes <= 15):
        return None
    
    if status == 'cancelled':
        # Check for severe weather
        severe_conditions = ['Thunderstorm', 'Heavy rain', 'Heavy snow', 'Fog', 'Severe thunderstorm']
        
        origin_cond = origin_weather.get('condition', '') if origin_weather else ''
        dest_cond = dest_weather.get('condition', '') if dest_weather else ''
        
        if any(s in origin_cond for s in severe_conditions) or any(s in dest_cond for s in severe_conditions):
            return 'weather'
        return 'carrier'
    
    if status == 'delayed':
        # Check weather conditions
        bad_weather = ['Rain', 'Thunderstorm', 'Snow', 'Fog', 'Drizzle']
        
        origin_cond = origin_weather.get('condition', '') if origin_weather else ''
        dest_cond = dest_weather.get('condition', '') if dest_weather else ''
        
        origin_precip = origin_weather.get('precip_prob', 0) if origin_weather else 0
        dest_precip = dest_weather.get('precip_prob', 0) if dest_weather else 0
        
        # High precipitation probability or bad conditions
        if (origin_precip and origin_precip > 50) or (dest_precip and dest_precip > 50):
            return 'weather'
        
        if any(b in origin_cond for b in bad_weather) or any(b in dest_cond for b in bad_weather):
            return 'weather'
        
        # Wind-related delays
        origin_wind = origin_weather.get('wind_kph', 0) if origin_weather else 0
        dest_wind = dest_weather.get('wind_kph', 0) if dest_weather else 0
        
        if (origin_wind and origin_wind > 40) or (dest_wind and dest_wind > 40):
            return 'weather'
        
        # Default to carrier if no weather issues found
        return 'carrier'
    
    return None


def backfill_weather_data(limit=None, days_back=30):
    """
    Main function to backfill weather data for existing flights.
    
    Args:
        limit: Max number of flights to process (None = all)
        days_back: Only process flights from last N days
    """
    print("\n" + "=" * 60)
    print("🌤️ WEATHER DATA BACKFILL")
    print("=" * 60)
    
    if not os.path.exists(DB_NAME):
        print(f"❌ Database not found: {DB_NAME}")
        return
    
    conn = sqlite3.connect(DB_NAME)
    
    # Add weather columns if needed
    init_weather_columns(conn)
    
    cursor = conn.cursor()
    
    # Get flights without weather data
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    query = """
        SELECT id, flight_date, origin, destination, scheduled_departure, scheduled_arrival, status, departure_delay
        FROM flights 
        WHERE flight_date >= ? 
        AND origin_weather IS NULL
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, (cutoff_date,))
    flights = cursor.fetchall()
    
    print(f"📊 Found {len(flights)} flights without weather data")
    
    if not flights:
        print("✅ All flights already have weather data!")
        conn.close()
        return
    
    processed = 0
    weather_found = 0
    reasons_inferred = 0
    
    for flight in flights:
        flight_id, date, origin, dest, dep_time, arr_time, status, delay = flight
        
        # Get origin weather
        origin_weather = get_historical_weather(origin, date, dep_time)
        
        # Get destination weather
        dest_weather = get_historical_weather(dest, date, arr_time or dep_time)
        
        # Infer delay reason
        reason = infer_delay_reason(origin_weather, dest_weather, status, delay)
        
        # Update database
        update_sql = """
            UPDATE flights SET
                origin_weather = ?,
                origin_temp_c = ?,
                origin_precip_prob = ?,
                origin_wind_kph = ?,
                dest_weather = ?,
                dest_temp_c = ?,
                dest_precip_prob = ?,
                dest_wind_kph = ?,
                inferred_delay_reason = ?
            WHERE id = ?
        """
        
        cursor.execute(update_sql, (
            origin_weather.get('condition') if origin_weather else None,
            origin_weather.get('temp_c') if origin_weather else None,
            origin_weather.get('precip_prob') if origin_weather else None,
            origin_weather.get('wind_kph') if origin_weather else None,
            dest_weather.get('condition') if dest_weather else None,
            dest_weather.get('temp_c') if dest_weather else None,
            dest_weather.get('precip_prob') if dest_weather else None,
            dest_weather.get('wind_kph') if dest_weather else None,
            reason,
            flight_id
        ))
        
        processed += 1
        if origin_weather or dest_weather:
            weather_found += 1
        if reason:
            reasons_inferred += 1
        
        # Progress update
        if processed % 10 == 0:
            print(f"   Processed {processed}/{len(flights)}...", end='\r')
        
        # Rate limiting for API
        time.sleep(0.2)  # 5 requests per second max
        
        # Commit every 50 records
        if processed % 50 == 0:
            conn.commit()
    
    conn.commit()
    conn.close()
    
    print(f"\n\n{'=' * 60}")
    print("✅ WEATHER BACKFILL COMPLETE!")
    print(f"📊 Processed: {processed} flights")
    print(f"🌤️ Weather data added: {weather_found} flights")
    print(f"🔍 Delay reasons inferred: {reasons_inferred} flights")
    print("=" * 60)


def show_weather_stats():
    """Show statistics about weather data in database"""
    if not os.path.exists(DB_NAME):
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print("\n📊 WEATHER DATA STATISTICS")
    print("-" * 40)
    
    # Total flights
    cursor.execute("SELECT COUNT(*) FROM flights")
    total = cursor.fetchone()[0]
    
    # Flights with weather
    cursor.execute("SELECT COUNT(*) FROM flights WHERE origin_weather IS NOT NULL")
    with_weather = cursor.fetchone()[0]
    
    # Delay reason breakdown
    cursor.execute("""
        SELECT inferred_delay_reason, COUNT(*) 
        FROM flights 
        WHERE inferred_delay_reason IS NOT NULL
        GROUP BY inferred_delay_reason
    """)
    reasons = cursor.fetchall()
    
    print(f"Total flights: {total}")
    print(f"With weather data: {with_weather} ({with_weather/total*100:.1f}%)" if total > 0 else "No flights")
    
    if reasons:
        print("\nInferred Delay Reasons:")
        for reason, count in reasons:
            print(f"  • {reason}: {count}")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill weather data for flights")
    parser.add_argument('--limit', type=int, help='Max flights to process')
    parser.add_argument('--days', type=int, default=30, help='Days back to process (default: 30)')
    parser.add_argument('--stats', action='store_true', help='Show weather statistics only')
    
    args = parser.parse_args()
    
    if args.stats:
        show_weather_stats()
    else:
        backfill_weather_data(limit=args.limit, days_back=args.days)
        print()
        show_weather_stats()
