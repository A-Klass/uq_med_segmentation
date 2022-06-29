
import torch
import itertools
import torch
import tqdm
import numpy as np

import torch.nn.functional as F

from utils.metrics import dice_coef

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def run_training_epoch(loader,
    model,
    criterion,
    optimizer,
    grad_scaler,
    cuda=True,
    verbose=False,
    subset=None,
    amp_enabled=False):
    """ 
    Kudos to swag.utils.train_epoch
    Within 1 training epoch, run through all batches in data loader
    """
    loss_sum = 0.0
    dice_sum = 0.0
    num_batches = len(loader)
    num_objects_current = 0

    model.train()
    
    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches, miniters = 30)
        
    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True).float()
        else:
            input = input.float()
            target = target.float()
        # assert input.shape[1] == model.in_channels
            # f'Network has been defined with {net.n_channels} input channels, ' \
            # f'but loaded images have {images.shape[1]} channels. Please check that ' \
            # 'the images are loaded correctly.'
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(input)
            if criterion.__name__ == 'soft_dice_coef_loss':
                loss = criterion(F.softmax(output, dim = 1), target)
            else:
                loss = criterion(output, target)
                # loss = criterion(output, target.squeeze(1).long())
            try:
                # dice = dice_coef(torch.sigmoid(output), torch.nn.functional.one_hot(target.squeeze(1).long()).permute(0,3,1,2).float())
                dice = dice_coef(torch.sigmoid(output), target)
            except ValueError:
                dice = torch.nan
                print("dice score calculation failed")
            except AssertionError:
                dice = torch.nan
                print("dice score calculation failed")
              
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()    
        grad_scaler.step(optimizer)         
        grad_scaler.update()
        
        loss_sum += loss.data.item() * input.size(0)
        dice_sum += dice.item() * input.size(0)
        num_objects_current += input.size(0)
            
    return {
        "dice_coef_avg": dice_sum / num_objects_current,
        "loss_avg": loss_sum / num_objects_current,
        }
    
def eval(loader,
         model, 
         criterion, 
         cuda=True, 
        #  regression=False, 
         verbose=False,
         amp_enabled=False):

    loss_sum = 0.0
    dice_sum = 0.0
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(input)
                if (criterion.__name__ == 'dice_coef_loss') or ('soft_dice_coef_loss'):
                    loss = criterion(F.softmax(output, dim = 1), target)
                else:
                    loss = criterion(output, target)           
            try:
                dice = dice_coef(torch.sigmoid(output), target)
            except ValueError:
                dice_sum = torch.nan
                print("dice score calculation failed")
            except AssertionError:
                dice_sum = torch.nan
                print("dice score calculation failed")
            loss_sum += loss.item() * input.size(0)
            dice_sum += dice.item() * input.size(0)

    return {
        "dice_coef_avg": dice_sum / num_objects_total,
        "loss_avg": loss_sum / num_objects_total
    }
    
def bn_update(loader, model, verbose=False, subset=None, cuda=False, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            if cuda:
                input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    
    
def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
        
def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
            targets.append(target.cpu().numpy())
        else:
            targets.append(target.numpy())
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())            
    return np.vstack(preds), np.concatenate(targets)