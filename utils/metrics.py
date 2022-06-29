import torch
import numpy as np
"""
https://github.com/ihamdi/Semantic-Segmentation/blob/3d32556385bfa86d8bbfcf5704aef8e119490d52/utils/dice_score.py
https://docs.monai.io/en/stable/metrics.html#mean-dice
"""

# def dice_coef_loss(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     """
#     Kudos: https://github.com/ihamdi/Semantic-Segmentation/blob/3d32556385bfa86d8bbfcf5704aef8e119490d52/utils/dice_score.py
#     """
#     # Average of Dice coefficient for all batches, or for a single mask
#     assert input.size() == target.size()
#     if input.dim() == 2 and reduce_batch_first:
#         raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input.dim() == 2 or reduce_batch_first:
#         inter = torch.dot(input.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input) + torch.sum(target)
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
#     else:
#         # compute and average metric for each batch element
#         dice = 0
#         for i in range(input.shape[0]):
#             dice += dice_coef_loss(input[i, ...], target[i, ...])
#         return dice / input.shape[0]
def dice_coef(y_pred, y_true, epsilon=1e-6):
    """
    https://github.com/XiaowenK/UNet_Family/blob/master/train_RV.py
    """
    if y_pred.shape != y_true.shape:
        return AssertionError("check dimensions!")
    
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)
    
    intersection = (y_pred_f * y_true_f).sum()
    
    A_sum = torch.sum(y_pred_f)
    B_sum = torch.sum(y_true_f)
    
    dice_score = (2. * intersection + epsilon) / (A_sum + B_sum + epsilon)
    
    if (dice_score > 1.0) or (dice_score < 0):
        return ValueError("dice score outside the possible range. check implementation")
    
    return dice_score

def soft_dice_coef_loss(y_pred, y_true, epsilon=1e-6):
    """
    Kudos: https://github.com/XiaowenK/UNet_Family/blob/master/train_RV.py
    """
    return 1 - dice_coef(y_pred, y_true, epsilon=1e-6)



def max_prob(sampled_outputs: np.array):
    """ Calculation of the mean maximal softmax probability of the sampled outputs of one input data.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Parameters:
    -----------
    sampled_outputs : np.array(sample, classes)
        softmax-outputs of sampled network

    Returns:
    -------
    float
        Mean maximum softmax probability
    """
    mean_output = np.mean(sampled_outputs, axis=0)

    return np.max(mean_output)

def get_entropy(sampled_outputs: np.array):
    """ Calculation of the mean entropy within the sampled outputs.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Parameters:
    -----------
    sampled_outputs : np.array(sample, classes)
        softmax-outputs of sampled network

    Returns:
    -------
    float
        Mean entropy of prediction
    """
    mean_output = np.mean(sampled_outputs, axis=0)
    return -((mean_output*np.log2(mean_output)).sum())

def get_MI(sampled_outputs: np.array):
    """ Calculation of the mutual information within the sampled outputs.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Parameters:
    -----------
    sampled_outputs : np.array(sample, classes)
        softmax-outputs of sampled network

    Returns:
    -------
    float
        mutual information of prediction
    """
    mean_output = np.mean(sampled_outputs, axis=0)
    mutual_information = (get_entropy(mean_output) - 1/len(sampled_outputs) * ((sampled_outputs*np.log2(sampled_outputs)).sum(axis=1)).sum())
    return mutual_information

def get_EKL(sampled_outputs: np.array):
    """ Calculation of the Expected Kullback Leibler Divergence within the sampled outputs.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Parameters:
    -----------
    sampled_outputs_ : np.array(sample, classes)
        softmax-outputs of sampled network

    Returns:
    -------
    float
        Expected Kullback Leibler Divergence of prediction
    """
    mean_output = np.mean(sampled_outputs, axis=0)
    ekl = 0
    for sample in range(len(sampled_outputs)):
        for class_idx in range(len(sampled_outputs[sample])):
            ekl += mean_output[class_idx] * np.log(mean_output[class_idx] / max(1e-16, (sampled_outputs[sample])[class_idx]))
    ekl = 1/len(sampled_outputs) * ekl

    return ekl

def get_predictive_variance(sampled_outputs: np.array):
    """ Calculation of the Predictive Variance within the sampled outputs.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Parameters:
    -----------
    sampled_outputs : np.array(sample, classes)
        softmax-outputs of sampled network

    Returns:
    -------
    np.array(float)
        Predicted Variance of prediction
    """
    mean_output = np.mean(sampled_outputs, axis=0)
    return np.mean((sampled_outputs-mean_output)**2, axis=0)

def get_AUROC(outputs_list, target):
    """ Calculation of the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from the sampled outputs.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Useful guide for understanding: https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044

    Parameters:
    -----------
    outputs_list : list[np.array(sample, batch_size, classes)]
        list of output-arrays of sampled network for each evaluation data
    target : list[np.array(batch_size)]
        labels of the data which might live on GPU

    Returns:
    -------
    float
        Mean AUROC of outputs
    """
    # stack all outputs and labels independent from batches each to one list
    # -> from [batch1(output1, output2, output3...), batch2(output1, output2, output3...)] to [output11, output12, output13, output21, output22, output23]
    outputs_array = []
    label_array = []
    for idx in range(len(outputs_list)):
        batch_first = outputs_list[idx].transpose(1,0,2)   # transpose outputs from (sample,batch,classes) to (batch,sample,classes)
        for data in range(len(batch_first)):
            mean_output = np.mean(batch_first[data], axis=0)  # get mean output of each class over samples
            outputs_array.append(mean_output)
            label_array.append((target[idx])[data])
    outputs_array = np.array(outputs_array)
    label_array = np.array(label_array)
    # convert lable vector to one-hot-label-vector (see https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array)
    n_values = len(outputs_array[0])
    label_onehot = np.eye(n_values)[label_array]

    if(np.isnan(outputs_array).any() or np.isinf(outputs_array).any()):
        print("Warning: NaN or InF in outputs_array! - AUROC is set to 0")
        return 0
    else:
        # need try-catch-block since when data is unbalanced and there's only one class represented it throws ValueError
        try:
            auroc = sklm.roc_auc_score(label_onehot, outputs_array, average = 'macro')
            return auroc
        except ValueError as e:
            #print("ValueError in get_AUROC: " + str(e) + " AUROC is set to 0.")
            return 0

def get_precision(outputs_list, target):
    """ Calculation of the Precision-score from the sampled outputs.
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Useful guide for understanding: https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044

    Parameters:
    -----------
    outputs_list : list[np.array(sample, batch_size, classes)]
        list of output-arrays of sampled network for each evaluation data
    target : list[np.array(batch_size)]
        labels of the data

    Returns:
    -------
    float
        Mean precision score of outputs
    """
    # stack all outputs and labels independent from batches each to one list
    # -> from [batch1(output1, output2, output3...), batch2(output1, output2, output3...)] to [output11, output12, output13, output21, output22, output23]
    outputs_array = []
    label_array = []
    for idx in range(len(outputs_list)):
        batch_first = outputs_list[idx].transpose(1,0,2)   # transpose outputs from (sample,batch,classes) to (batch,sample,classes)
        for data in range(len(batch_first)):
            mean_output = np.mean(batch_first[data], axis=0)  # get mean output of each class over samples
            outputs_array.append(mean_output)
            label_array.append((target[idx])[data])
    outputs_array = np.array(outputs_array)
    label_array = np.array(label_array)

    if(np.isnan(outputs_array).any() or np.isinf(outputs_array).any()):
        print("Warning: NaN or InF in outputs_array! - Precision is set to 0")
        return 0
    else:
        # convert outputs to prediction-vector (vector of indeces of predicted classes)
        prediction_array = np.argmax(outputs_array, axis=1)
        # need try-catch-block since when data is unbalanced and there's only one class represented it throws ValueError
        try:
            precision = sklm.precision_score(label_array, prediction_array, average='macro', zero_division=0)
            if(precision is None):
                print("Precision returns None!")
            return precision
        except ValueError as e:
            print("ValueError in get_precision: " + str(e))
            return 0

def get_recall(outputs_list, target):
    """ Calculation of the Recall-score from the sampled outputs.

    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Useful guide for understanding: https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044

    Parameters:
    -----------
    outputs_list : list[np.array(sample, batch_size, classes)]
        list of output-arrays of sampled network for each evaluation data
    target : list[np.array(batch_size)]
        labels of the data

    Returns:
    -------
    np.array(float)
        Mean recall score of outputs
    """
    # stack all outputs and labels independent from batches each to one list
    # -> from [batch1(output1, output2, output3...), batch2(output1, output2, output3...)] to [output11, output12, output13, output21, output22, output23]
    outputs_array = []
    label_array = []
    for idx in range(len(outputs_list)):
        batch_first = outputs_list[idx].transpose(1,0,2)   # transpose outputs from (sample,batch,classes) to (batch,sample,classes)
        for data in range(len(batch_first)):
            mean_output = np.mean(batch_first[data], axis=0)  # get mean output of each class over samples
            outputs_array.append(mean_output)
            label_array.append((target[idx])[data])
    outputs_array = np.array(outputs_array)
    label_array = np.array(label_array)
    if(np.isnan(outputs_array).any() or np.isinf(outputs_array).any()):
        print("Warning: NaN or InF in outputs_array! - Recall is set to 0")
        return 0
    else:
        # convert outputs to prediction-vector (vector of indeces of predicted classes)
        prediction_array = np.argmax(outputs_array, axis=1)
        # need try-catch-block since when data is unbalanced and there's only one class represented it throws ValueError
        try:
            recall = sklm.recall_score(label_array, prediction_array, average='macro', zero_division=0)
            if(recall is None):
                print("Recall returns None!")
            return recall
        except ValueError as e:
            print("ValueError in get_recall: " + str(e))
            return 0

def get_accuracy(sampled_outputs: np.array, target: np.array):
    """
    Calculation of accuracy from sampled outputs vs. targets
    
    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)
    
    Parameters:
    -----------
    sampled_outputs : np.array(sample, batch_size, classes)
        list of output-arrays of sampled network for each evaluation data
    target : np.array(batch_size)
        labels of the data

    Returns:
    -------
    np.array(float)
        Mean accuracy of sampled_outputs as percentage
    
    """
    # sklm.accuracy_score(target, np.argmax(sampled_outputs, 2))
    # return (np.argmax(sampled_outputs, 2) == np.array(target)).mean()
    hits = (np.argmax(np.array(sampled_outputs), 2) == np.array(target))
    result = hits.sum() / hits.size
    assert result <= 1.0, "this cannot be"
    return result * 100 
    # return (sampled_outputs == target).sum() / len(target)
    
def get_f1(outputs_list, target):
    """ Calculation of the f1-score from the sampled outputs.

    According to Paper: Gawlikowski, J; et al.: A Survey of Uncertainty in Deep Neural Networks (https://arxiv.org/abs/2107.03342)

    Useful guide for understanding: https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044

    Parameters:
    -----------
    outputs_list : list[np.array(sample, batch_size, classes)]
        list of output-arrays of sampled network for each evaluation data
    target : list[np.array(batch_size)]
        labels of the data

    Returns:
    -------
    np.array(float)
        Mean f1 score of outputs
    """
    # stack all outputs and labels independent from batches each to one list
    # -> from [batch1(output1, output2, output3...), batch2(output1, output2, output3...)] to [output11, output12, output13, output21, output22, output23]
    outputs_array = []
    label_array = []
    for idx in range(len(outputs_list)):
        batch_first = outputs_list[idx].transpose(1,0,2)   # transpose outputs from (sample,batch,classes) to (batch,sample,classes)
        for data in range(len(batch_first)):
            mean_output = np.mean(batch_first[data], axis=0)  # get mean output of each class over samples
            outputs_array.append(mean_output)
            label_array.append((target[idx])[data])
    outputs_array = np.array(outputs_array)
    label_array = np.array(label_array)

    if(np.isnan(outputs_array).any() or np.isinf(outputs_array).any()):
        print("Warning: NaN or InF in outputs_array! - F1 score is set to 0")
        return 0
    else:
        # convert outputs to prediction-vector (vector of indeces of predicted classes)
        prediction_array = np.argmax(outputs_array, axis=1)
        # need try-catch-block since when data is unbalanced and there's only one class represented it throws ValueError
        try:
            f1 = sklm.f1_score(label_array, prediction_array, average='macro', zero_division=0)
            if f1 is None:
                print("F1 returns None!")
            return f1
        except ValueError as e:
            print("ValueError in get_f1: " + str(e))
            return 0
        
def confusion_matrix_mc(sampled_outputs, target):
    """
    Computes the confusion matrix for the different Monte Carlo samples and
    gives back the number of occurences in a class x class matrix.
    
    Parameters:
    -----------
    sampled_outputs : np.array(sample, batch_size, classes)
        list of output-arrays of sampled network for each evaluation data
    target : np.array(batch_size)
        labels of the data

    Returns:
    -------
    np.array(float)
        Number of occurances of y_pred vs y_true in a class x class matrix
    
    """
    conf_mat = np.zeros((sampled_outputs.shape[2], sampled_outputs.shape[2]))
    
    for idx in range(len(sampled_outputs)):
        y_pred = np.argmax(sampled_outputs[idx,:,:].reshape(sampled_outputs.shape[1],
                                                           sampled_outputs.shape[2]),
                           axis = 1)
        conf_mat += sklm.confusion_matrix(target, y_pred)
    
    return conf_mat

# # Defining Dice loss class
# # Source code: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# # https://gist.github.com/MBoustani/1e4a18286e091d71cc74fb000490d349#file-spinal-column-segmentation-ipynb
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):

#         inputs = torch.sigmoid(inputs)       
        
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         bce = F.binary_cross_entropy_with_logits(inputs, targets)
#         pred = torch.sigmoid(inputs)
#         loss = bce * 0.5 + dice * (1 - 0.5)
        
#         # subtract 1 to calculate loss from dice value
#         return 1 - dice
    
# def dice_coeff(input: Tensor, 
#                target: Tensor,
#                reduce_batch_first: bool = False, 
#                epsilon=1e-6):
#     # Average of Dice coefficient for all batches, or for a single mask
#     assert input.size() == target.size()
#     if input.dim() == 2 and reduce_batch_first:
#         raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input.dim() == 2 or reduce_batch_first:
#         inter = torch.dot(input.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input) + torch.sum(target)
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
#     else:
#         # compute and average metric for each batch element
#         dice = 0
#         for i in range(input.shape[0]):
#             dice += dice_coeff(input[i, ...], target[i, ...])
#         return dice / input.shape[0]

# def dice_loss(input: Tensor, target: Tensor):
#     # Dice loss (objective to minimize) between 0 and 1
#     assert input.size() == target.size()
#     return 1 - dice_coeff(input, target, reduce_batch_first=True)