import sqlite3
import os

DB_NAME = 'india_data.db'

def create_database():
    """Creates the SQLite database and the recent_flights table."""
    if os.path.exists(DB_NAME):
        print(f"Database '{DB_NAME}' already exists.")
    else:
        print(f"Creating new database: '{DB_NAME}'")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create the table
    # We add a UNIQUE constraint to prevent duplicate entries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recent_flights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flight_number TEXT NOT NULL,
        origin TEXT NOT NULL,
        destination TEXT NOT NULL,
        flight_date TEXT NOT NULL,
        status TEXT NOT NULL,
        delay_minutes INTEGER,
        UNIQUE(flight_number, origin, destination, flight_date)
    )
    """)
    
    print(f"Table 'recent_flights' is ready.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()