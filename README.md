# Handwriting Reognition in Pytorch

**Data Types (JPG and CSV):**
- CSVs containing JPEG filename of black and white handwriting samples
- Train-test split: divided into 331,059 training and 41,382 validation respectively
- Each row contains the true label and the label length (# of chars)
**Data Inconsistencies**:
- Illegible handwriting samples have a "NULL" label; these instances were removed
- change all labels to upper case

**Preprocessing Steps:**
**Image Sizing:**
- Fixed at 256 x64
- Larger-sized images were cropped
- Smaller-sized images were padded
- Image normalized
**Labels:**
- 30 distinct characters (26 upper-case alphabet; 4 special characters(-, ', '"', ' '))
- Set max length of labels to be 64 characters; **accounts for feeding fixed-size tensors in PyTorch**
- Create a dictionary mapping characters to a number and pad with -1 until 64 characters

## CNN_RNN Model Architecture
**Inputs:** Image, Label
**CNN Model Attributes:**.  
- 3x3 kernel size (stride 2),
- 3 downsamples (3, 32, 64, 128 channels)
- Batch Normalization Layers after conv. layers
- ReLU or Mish activation functions after Batch Normalization
- 2x2 Maxpool after conv. layers 1,2; 2x1 Maxpool after 3rd/final conv. layer
**RNN Model Attributes:**.  
- outputs of CNN fed through linear layer with (in_features=1024; output_features=64)
- bidirectinal; input_size=64; hidden_size=512
- outputs of LSTM fed through linear layer with (in_features=1024; output_features=30)
- The 30 out-features represents a probability for each character
-

## Hyperparameters
- cosine learning rate scheduler
- variable batch sizes (32, 64)
- MiSH vs ReLu activation function
- ADAM optimizer

## Loss function
- CTCLoss

## Training
- send image and labels to GPU
- take the Log SoftMax of the predicted labels
- Calculate character accuracy (compare characters the label was not -1 padded
- word accuracy (only correct if all characters within a word were correct)

## Results (20 epochs; Mish Activation; batch_size=32; )


## Results (20 epochs; Mish Activation; batch_size=32; )


## Conclusions and Lessons Learned
**CTCLoss Function** This was our first time using the CTCloss and in a couple of our trainin losses were "Nan". After some research, we
determined that CTCLoss is subject to numerical instability if it is not on the CPU. The reason why the "Nans" were sporadic was because
we reasoned that there were certain combinations within a batch that caused the numerical instability
**Keras vs Pytorch Translation**: Since we modeled our notebook from the top-performing notebook in Kaggle, we realized that Keras
functions vs PyTorch functions had minor inconsistencies. For example, we had to permute the dimensions in order to get the CTC loss
function to work. Additionally, the CTC Loss function in PyTorch requires y_prediction to be in the form (sequence length, batch_size, # of classes)
