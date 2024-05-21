import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils import get_imagenet_classes
# from tensorflow.keras.optimizers import Adam

# create mask to correspond between imagenet-a labels and imagenet-1k

all_wnids, imagenet_a_wnids, imagenet_a_mask = get_imagenet_classes()

# create a matrix mapping from imagenet-a to imagenet-1k one-hot encodings
mask_matr = np.zeros((1000,200))
mask_inds = np.argwhere(imagenet_a_mask).flatten()
for j in range(200):
    mask_matr[mask_inds[j],j]=1

#make this a tf tensor
mask_tens = tf.convert_to_tensor(mask_matr, dtype=tf.float32)

# link to parent folder of all the classes
test_dir = "C:/Users/laure/Documents/nat_advs_proj/imagenet-a-split/test"

# establish dataset, class_names defined by all_wnids
batch_size = 32
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_dir, labels='inferred',batch_size=None, label_mode="categorical",
                                                                    shuffle=False, class_names = imagenet_a_wnids)
pre_labels = np.array([y for x, y in test_dataset])
pre_labels = np.argmax(pre_labels, axis=1)
# Define the preprocessing function
def preproc(tensor, y):
    # image processing
    tensor = tf.image.resize_with_crop_or_pad(tensor, 224, 224)
    tensor = preprocess_input(tensor)

    # distribute correct labels to the 1000 imagenet classes
    y_new = tf.matmul(mask_tens,tf.reshape(y,(200,1)))
    y_new = tf.reshape(y_new,(1000,))
    return tensor, y_new

test_dataset = test_dataset.map(preproc)
image_labels = np.array([y for x, y in test_dataset])
test_dataset = test_dataset.batch(batch_size)

base_model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg')

base_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

# results = tf.keras.metrics.categorical_accuracy(image_labels, base_model.predict(test_dataset))
# print("Test 1000 Accuracy:", np.sum(results)/image_labels.shape[0])
score = base_model.evaluate(test_dataset)
print("Scores: "+str(score))

def get_predictions(dset, net=None, mask=None):

    # predict labels based on network
    outputs = net.predict(dset)

    # mask outputs to only be imagenet-a related
    mask_outputs = outputs[:,mask]
    
    # take argmax of the imagenet-a related labels ONLY
    pred = np.argmax(mask_outputs,axis=1)

    # compare to real labels
    num_correct = np.array(pred==pre_labels).sum()

    #output correct examples
    correct = np.argwhere(pred==pre_labels)
    
    return correct, num_correct


def get_imagenet_a_results(loader, net, mask):
    correct, num_correct = get_predictions(loader, net, mask)
    acc = num_correct / pre_labels.shape[0]
    print('Accuracy (%):', round(100*acc, 4))

    return correct

correct_labs = get_imagenet_a_results(test_dataset, base_model, imagenet_a_mask)

# preds = base_model.predict(test_dataset)

# print("shape: "+str(preds.shape))


