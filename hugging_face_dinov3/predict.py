import torch
from PIL import Image

def predict_image(model, image_processor, image_path, class_names, threshold=0.5):
    """
    Predict labels for a single image
    """
    model.eval()

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(image, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

    # Get predictions above threshold
    predictions = probabilities > threshold
    predicted_labels = [class_names[i] for i, pred in enumerate(predictions[0]) if pred]
    predicted_probs = [probabilities[0][i].item() for i, pred in enumerate(predictions[0]) if pred]

    return predicted_labels, predicted_probs

# Example usage
# predicted_labels, predicted_probs = predict_image(model, image_processor, "path/to/image.jpg", class_names)