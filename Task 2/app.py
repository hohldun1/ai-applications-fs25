import gradio as gr
from transformers import pipeline

# Load models
vit_classifier = pipeline("image-classification", model="222dunja/vit-base-oxford-iiit-pets")
clip_detector = pipeline(model="openai/clip-vit-large-patch14", task="zero-shot-image-classification")

# Oxford Pet Class Labels
labels_oxford_pets = [
    'Siamese', 'Birman', 'shiba inu', 'staffordshire bull terrier', 'basset hound', 'Bombay', 'japanese chin',
    'chihuahua', 'german shorthaired', 'pomeranian', 'beagle', 'english cocker spaniel', 'american pit bull terrier',
    'Ragdoll', 'Persian', 'Egyptian Mau', 'miniature pinscher', 'Sphynx', 'Maine Coon', 'keeshond', 'yorkshire terrier',
    'havanese', 'leonberger', 'wheaten terrier', 'american bulldog', 'english setter', 'boxer', 'newfoundland', 'Bengal',
    'samoyed', 'British Shorthair', 'great pyrenees', 'Abyssinian', 'pug', 'saint bernard', 'Russian Blue', 'scottish terrier'
]

# Define the inference function
def classify_pet(image):
    vit_result = vit_classifier(image)[0]  # Top-1 prediction
    clip_result = clip_detector(image, candidate_labels=labels_oxford_pets)[0]  # Top-1 prediction

    return {
        "Transfer Learning Prediction (ViT)": f"{vit_result['label']} ({vit_result['score']:.2f})",
        "Zero-Shot Prediction (CLIP)": f"{clip_result['label']} ({clip_result['score']:.2f})"
    }

# Define example image paths
example_images = [
    ["example_images/dog1.jpeg"],
    ["example_images/dog2.jpeg"],
    ["example_images/leonberger.jpg"],
    ["example_images/snow_leopard.jpeg"],
    ["example_images/cat.jpg"]
]

# Launch Gradio Interface
iface = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="filepath", label="Upload a Pet Image"),
    outputs=gr.JSON(label="Model Predictions"),
    title="🐶🐱 Pet Classifier: Transfer Learning vs. Zero-Shot",
    description="This app compares a fine-tuned ViT model with CLIP for Oxford Pets classification.",
    examples=example_images
)

iface.launch()