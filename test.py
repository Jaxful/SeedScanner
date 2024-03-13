import torch
from torchvision import transforms
from PIL import Image
from train import MinecraftCNN  # Adjust this if needed to correctly import your model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MinecraftCNN(num_classes=3)
model.load_state_dict(torch.load('minecraft_cnn_model.pth', map_location=device))
model.eval()
model = model.to(device)

# Define class names in the same order as during training
class_names = ['Bad', 'Best', 'Moderate']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4018, 0.4675, 0.4808], std=[0.1856, 0.1891, 0.2910]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    image = image.to(device)  # Ensure the image is on the correct device
    return image

def predict(image_path, model, device):
    image = preprocess_image(image_path)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()

    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

if __name__ == '__main__':
    image_path = r"Test Image Path"
    predicted_class_name = predict(image_path, model, device)
    print(f'Predicted class: {predicted_class_name}')
