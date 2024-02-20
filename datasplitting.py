import pickle

filename = "c:\\Users\\asus\\Downloads\\ApneaData.csv"

# Read and split the file content into rows
with open(filename, 'r') as f:
    rows = f.read().split("\n")

# Convert each row into a list of integers and store them in a list
data = [[int(value) for value in row.split(" ")] for row in rows]

# Save the data as a pickle file
with open('ApneaData.pkl', 'wb') as f:
    pickle.dump(data, f)