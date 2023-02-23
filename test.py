from torchvision import datasets

train_data = datasets.ImageFolder(root="data/Cells/fixed") # transforms to perform on labels (if necessary)

print(f"Train data:\n{train_data}\nTest data:\n{train_data}")