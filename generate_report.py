from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import sqlite3

def generate_pdf():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    data = c.fetchall()
    conn.close()

    pdf = SimpleDocTemplate("CoffeeGuard_Report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("CoffeeGuard AI - Prediction Report", styles["Title"]))

    table_data = [["ID", "Image", "Result", "Time"]]

    for row in data:
        table_data.append(list(row))

    table = Table(table_data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.pink),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))

    content.append(table)

    pdf.build(content)
    print("PDF Generated Successfully!")

generate_pdf() 