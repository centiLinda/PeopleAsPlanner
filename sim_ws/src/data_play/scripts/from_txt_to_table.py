#!/usr/bin/env python3
import pandas as pd
import os

scene="test_data"
# challenge="roundabout_0"
challenge="crossing_0"
# challenge="promenade_0"

# Load the data
file_path = f"/root/{scene}/{challenge}/experiment_log.csv"  # Update this path if needed
df = pd.read_csv(file_path,skiprows=1, header=None, names=["Trial_ID", "Finish", "Total_collision", "Traveling_dis", "Total_Time"])
print(df)
df = df.fillna(0)  # Fill NaN values with 0
df = df[df["Trial_ID"] != 100]

# Filter rows where Finish == 1
df_filtered = df[df["Finish"] == 1]

# Calculate the sum of Finish
sum_finish = df_filtered["Finish"].sum()

# Calculate the averages
avg_total_collision = df_filtered["Total_collision"].mean()
avg_traveling_dis = df_filtered["Traveling_dis"].mean()
avg_total_time = df_filtered["Total_Time"].mean()

# Display the results
print(f"sum_finish: {sum_finish}, avg_tcc: {avg_total_collision}, avg_travel_dis: {avg_traveling_dis}, avg_total_time: {avg_total_time}")

# Extract the directory path
dir_path = os.path.dirname(file_path)

# Define the new file path for avg.txt
avg_file_path = os.path.join(dir_path, "avg.txt")

# Content to save
content = f"sum_finish: {sum_finish}, avg_tcc: {avg_total_collision}, avg_travel_dis: {avg_traveling_dis}, avg_total_time: {avg_total_time}"

# Write to avg.txt
with open(avg_file_path, "w") as file:
    file.write(content)

print(f"File saved at: {avg_file_path}")
