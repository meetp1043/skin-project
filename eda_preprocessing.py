import os
import matplotlib.pyplot as plt

data_dir = "data/train"

classes = os.listdir(data_dir)
counts = []

for c in classes:
    counts.append(len(os.listdir(os.path.join(data_dir, c))))

plt.figure(figsize=(10,5))
plt.bar(classes, counts)
plt.xticks(rotation=45)
plt.title("Class Distribution")

os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/class_distribution.png")

print("✅ EDA completed")