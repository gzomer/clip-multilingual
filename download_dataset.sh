ANNOTATIONS=http://images.cocodataset.org/annotations/annotations_trainval2014.zip
IMAGES=http://images.cocodataset.org/zips/train2014.zip

# Download annotations
if [ ! -d "./datasets/annotations" ]; then
    echo "Downloading annotations..."
    wget $ANNOTATIONS -O ./datasets/annotations.zip
    echo "Extracting annotations..."
    unzip ./datasets/annotations.zip -d ./datasets/annotations
    rm ./datasets/annotations.zip
else
    echo "Annotations already downloaded."
fi

# Download images
if [ ! -d "./datasets/train2014" ]; then
    echo "Downloading images..."
    wget $IMAGES -O ./datasets/images.zip
    echo "Extracting images..."
    unzip ./datasets/images.zip -d ./datasets/train2014
    rm ./datasets/images.zip
else
    echo "Images already downloaded."
fi