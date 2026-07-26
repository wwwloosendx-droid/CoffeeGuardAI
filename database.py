# ==========================================================
# DATABASE CONFIGURATION
# ==========================================================

import sqlite3


DATABASE = "predictions.db"



def create_connection():

    conn = sqlite3.connect(
        DATABASE
    )

    return conn



def create_table():

    conn = create_connection()

    cursor = conn.cursor()


    cursor.execute("""

    CREATE TABLE IF NOT EXISTS analyses (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        image_name TEXT,

        ripe INTEGER,

        unripe INTEGER,

        ripening INTEGER,

        spoilt INTEGER,

        coffee_tree INTEGER,

        readiness REAL,

        spoilage REAL,

        recommendation TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )

    """)


    conn.commit()

    conn.close()



# ==========================================================
# SAVE ANALYSIS
# ==========================================================


def save_analysis(data):


    conn = create_connection()

    cursor = conn.cursor()



    cursor.execute("""

    INSERT INTO analyses (

        image_name,

        ripe,

        unripe,

        ripening,

        spoilt,

        coffee_tree,

        readiness,

        spoilage,

        recommendation

    )

    VALUES (?,?,?,?,?,?,?,?,?)

    """,

    (

        data["image_name"],

        data["ripe"],

        data["unripe"],

        data["ripening"],

        data["spoilt"],

        data["coffee_tree"],

        data["readiness"],

        data["spoilage"],

        data["recommendation"]

    ))



    conn.commit()

    conn.close()




# ==========================================================
# GET ALL ANALYSES
# ==========================================================


def get_all_analysis():


    conn = create_connection()

    cursor = conn.cursor()


    cursor.execute("""

    SELECT *

    FROM analyses

    ORDER BY created_at DESC

    """)


    results = cursor.fetchall()


    conn.close()


    return results