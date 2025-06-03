import fiftyone as fo
import fiftyone.zoo as foz
import os

# Define output directory locally
output_dir = os.path.join(os.getcwd(), 'dataset', 'open_images_v7')

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Use exact class names from Open Images dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=[
        "Bottle",
        "Shelf",
        # "Shopping cart",
        # "Convenience store",
        # "Drink",
        # "Liquor store",
        "Supermarket"
    ],
    max_samples=100,  # Increased sample size
    dataset_name="shelf_monitoring",
    shuffle=True,
    seed=51
)

# Print dataset statistics
print(f"Downloaded {len(dataset)} images")
print("\nClass distribution:")
# print(dataset.count_values("detections.detections.label"))

# Filter for images that contain both shelves and bottles
both_query = {
    "$and": [
        {"detections.detections.label": "Shelf"},
        {"detections.detections.label": "Bottle"}
    ]
}
filtered_dataset = dataset.match(both_query)
print(f"\nImages with both shelves and bottles: {len(filtered_dataset)}")

# Optional: Visualize the filtered dataset
session = fo.launch_app(filtered_dataset)











# import fiftyone as fo
# import fiftyone.zoo as foz
# import os

# # Define output directory locally
# output_dir = os.path.join(os.getcwd(), 'dataset', 'open_images_v7')

# # Create directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Use exact class names from Open Images dataset
# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split="validation",
#     label_types=["detections"],
#     classes=["Bottle", "Shelf"],  # Using "Bottle" instead of "Beverage"
#     max_samples=100,
#     dataset_name="shelf_monitoring",  # Added unique dataset name
#     shuffle=True,
#     seed=51
# )

# # Print dataset info
# print(f"Downloaded {len(dataset)} images to {output_dir}")
# # print("Classes:", dataset.distinct("detections.detections.label"))

# # Optional: Visualize a sample
# session = fo.launch_app(dataset)














# import fiftyone as fo
# import fiftyone.zoo as foz
# import os

# # Define output directory locally
# output_dir = os.path.join(os.getcwd(), 'dataset', 'open_images_v7')

# # Create directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Download 100 images of "Beverage" from validation split
# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split="validation",
#     label_types=["detections"],  # Download bounding box annotations
#     classes=["Beverage", "Shelf"],  # Filter for Beverage and Shelf classes
#     max_samples=100,            # Limit to 100 images
#     dataset_dir=output_dir,     # Save locally
#     seed=51,                    # For reproducibility
#     shuffle=True                # Randomize selection
# )

# # Print dataset info
# print(f"Downloaded {len(dataset)} images to {output_dir}")

# # Optional: Visualize a sample
# session = fo.launch_app(dataset)









# import fiftyone as fo
# import fiftyone.zoo as foz
# import os

# # Define output directory in Google Drive
# output_dir = '/content/drive/MyDrive/shelfsense/open_images_v7'

# # Create directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Download 100 images of "Beverage" from validation split
# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split="validation",
#     label_types=["detections"],  # Download bounding box annotations
#     classes=["Beverage"],       # Filter for Beverage class
#     max_samples=100,            # Limit to 100 images
#     dataset_dir=output_dir,     # Save to Google Drive
#     seed=51,                    # For reproducibility
#     shuffle=True                # Randomize selection
# )

# # Print dataset info
# print(f"Downloaded {len(dataset)} images to {output_dir}")

# # Optional: Visualize a sample
# session = fo.launch_app(dataset)








# # from google.colab import drive
# # drive.mount('/content/drive')


# # from openimages.download import download_images
# # download_images(labels=["shelf", "beverage"], limit=1000, output_dir="openimages_shelf")


# from openimages.download import download_dataset

# # Define the classes you want to download
# classes = ['Shelf', 'Beverage']

# # Download the dataset
# download_dataset(
#     classes=classes,
#     data_type='train',  # Options: 'train', 'validation', 'test'
#     limit=1000,
#     dest_dir='openimages_shelf'
# )