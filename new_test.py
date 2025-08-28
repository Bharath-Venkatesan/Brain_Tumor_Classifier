import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_names = torch.load(r"E:\RESUME_PROJECTS\brain_tumor_classification_model\model\class_names.pth")
# print("Class labels:", class_names) ---> for logging purposes


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Model loading for new_customcnn.pth
model = CustomCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(r"E:\RESUME_PROJECTS\brain_tumor_classification_model\model\new_customcnn.pth", map_location=device))
model.eval()

#Test image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction function(conditional)
import time

def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    predicted_label = class_names[class_idx]

    if predicted_label.replace(" ", "").lower() in ["no_tumor", "notumor"]:
        print( "No Tumor Detected") 
        time.sleep(2)
        return f"Stopping computation...You may quit now"
    else:
        print("Tumor Detected.....Please wait for classification process")
        time.sleep(2) 
        return f"Predicted Type of Tumor : {predicted_label}"




test_image = r"E:\RESUME_PROJECTS\brain_tumor_classification_model\test_image.png"  
print(predict_image(test_image))
