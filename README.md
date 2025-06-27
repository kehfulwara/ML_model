# ML_model
Used the ISIC 2019 dataset, which is a well-known medical dataset specially made for skin cancer detection research.

It contains a large number of high-quality dermoscopic images of skin lesions, which include both benign (non-cancerous) and malignant (cancerous) cases.

Along with each image, the dataset also provides metadata. This includes:

The patient’s age

Their gender

The location of the lesion on the body (like back, face, etc.)

And the diagnosis method, such as whether it was confirmed by histopathology, which is the gold standard for skin cancer diagnosis.

The most important column in our dataset is called benign_malignant.
This label tells us if a particular image shows a benign or malignant skin lesion.
We used this label to train our CNN models — so that the model can learn to classify new images correctly.

Each image in the dataset has a unique ID number, which we matched with the correct label using the metadata file.
So for example, if the image with ID ISIC_0012345 has a label of “malignant”, the model knows it needs to learn patterns related to cancer from that image.

We used stratified sampling to make sure that the class distribution of benign and malignant cases stayed the same in both the training and validation sets. This helps the model learn in a balanced way and gives us more reliable performance results.

Also, using a single labeled dataset like ISIC 2019 allowed us to apply consistent preprocessing, such as resizing, cleaning, and augmentation — making sure that the input to the model is clean, accurate, and standardized.

Step 1: Data Preprocessing
First, we cleaned the dataset by:

Filling missing values (like unknown sex or age),

Removing rows without proper labels,

Enhancing image quality using filters like CLAHE and denoising,

And resizing all images to the same size: 384×384 pixels.

Step 2: Syncing Images with Metadata
Next, we made sure that the image files and their labels (like benign or malignant from the benign_malignant column) were properly matched. Any missing or unmatched images were removed.

This gave us a clean labeled dataset, ready for model training.

Step 3: Stratified Sampling
Since our dataset had more benign cases than malignant, we used stratified sampling.
This means we split the data into training and validation sets while keeping the same class ratio (74.5% benign and 25.5% malignant in both sets).

Step 4: Selecting CNN Architecture
Then, we selected our deep learning models. We used EfficientNetB0 for our initial experiments, and also tested EfficientNetB4 later to compare performance.

Both models are strong CNN architectures that work well for image classification.

Step 5: Binary Classification
The selected model takes an image and predicts whether it is benign or malignant.
This is called binary classification because there are only two output classes.

Step 6: Performance Evaluation
Finally, we evaluated the model using metrics like:

Accuracy

Precision

Recall

F1-score

AUC and PR-AUC scores
