import sqlite3
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------
conn = sqlite3.connect("predictions.db")
c = conn.cursor()

c.execute("SELECT result FROM predictions")
data = c.fetchall()
conn.close()

# ---------------- COUNT RESULTS ----------------
labels = ["Healthy", "Rust", "Wilt"]
counts = [0, 0, 0]

for row in data:
    result = row[0]

    if "Healthy" in result:
        counts[0] += 1
    elif "Rust" in result:
        counts[1] += 1
    elif "Wilt" in result:
        counts[2] += 1

# ---------------- PLOT GRAPH ----------------
plt.bar(labels, counts)
plt.title("Coffee Leaf Disease Distribution")
plt.xlabel("Disease Type")
plt.ylabel("Number of Predictions")

plt.savefig("static/graph.png")
plt.show()