# ridvikpal/FruitSalad

FruitSalad is a CNN used to classify different types of fruits from their images. It is trained on the [Fruits Classification Dataset from Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification/data), and takes 128x128px images as an input. Any images that are not 128x128px are resized using torchvision transforms. it supports classifying the following fruits:

- Apples
- Bananas
- Grapes
- Mangoes
- Strawberries

Please see the [FruitSalad.ipynb](./FruitSalad.ipynb) file for detailed results and information.
