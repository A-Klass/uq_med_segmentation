
import gc
import logging
import numpy as np
import os
import time
import wandb
import warnings
import torch
import torch.optim as optim


# ---------- vanilla swag stuff ----------
# from swag.utils import save_checkpoint
# from swag.losses import cross_entropy # loss function that allows input of model
from torch.nn.functional import cross_entropy

# ---------- custom swag stuff ----------
from utils.swag_custom import SWAG
from utils.swag_utils_custom import bn_update, eval, run_training_epoch, predictions

# ---------- custom helpers ----------
from utils.tracker import TrainTracker
from utils.train_utils import load_processed_data, get_parser, assert_parser_params, make_dataloaders
from utils.metrics import soft_dice_coef_loss
from u_net.unet import UNet

# ----------------------------------------------------------------------------
def train_unet(params):   
    if torch.cuda.is_available() and params.use_cuda:
        params.device = torch.device("cuda"); cuda = True; torch.cuda.empty_cache()
    else:
        params.device = torch.device("cpu"); cuda = False
        
    # Load data            
    imgs_train, masks_train = load_processed_data(file_processed_images=params.path_train_images,
                                                  file_processed_masks=params.path_train_masks)
    assert len(imgs_train) == len(masks_train), "check dimensions and preprocessing"
    
    # check model path and create folder if not yet existing
    model_path = os.path.normpath(params.path_model)
    if not os.path.exists(model_path): os.mkdir(model_path); print(f"created folder for models: {model_path}")

    # create subset to train on, if applicable. meant for testing on weaker PC
    if params.take_small_subset:
        subset_size = 100 
        imgs_train = imgs_train[0:subset_size]
        masks_train = masks_train[0:subset_size]
        
    # classification problem with 3 or 2 target classes (depending on whether background is included)
    n_unique_classes = np.unique(masks_train).size
    if params.n_classes != n_unique_classes:
        warnings.warn(f'overwriting params.n_classes = {params.n_classes} with np.unique(masks_train).size = {np.unique(masks_train).size}',
                      stacklevel = 2)
        params.n_classes = n_unique_classes

    # make dataloaders
    loader_args = dict(batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=True)
    train_loader, val_loader, n_train, n_val = make_dataloaders(imgs = imgs_train,
                                                                masks=masks_train,
                                                                loader_args=loader_args,
                                                                criterion = params.criterion,
                                                                val_percent = params.val_percent,
                                                                seed = params.seed)
    del imgs_train
    del masks_train
    gc.collect()
    
    # Set up the model
    if params.net == "UNet":
        model = UNet(n_channels = params.n_channels, 
                     n_classes = params.n_classes).to(params.device)
    else:
        raise ValueError("currently only UNet supported.")
    # TODO set up other architectures
    
    # Initialize SWAG 
    swag_model = SWAG(
        base = UNet,
        no_cov_mat=params.no_cov_mat,
        max_num_models=params.max_num_models, 
        var_clamp=1e-30,
        n_classes=params.n_classes, 
        n_channels = params.n_channels).to(params.device)
    # model = torch.nn.DataParallel(model) # TODO investigate the usefulness of this
    
    # Set up the optimizer, the loss, the learning rate scheduler 
    match params.optm: # only python 3.10+
        case "SGD":
            optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum= params.sgd_momentum)
        case "Adam":
            optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=params.adam_betas)

    match params.criterion:
        case "cross_entropy":
            criterion = cross_entropy
        case "soft_dice_coeff":
            criterion = soft_dice_coef_loss

    best_loss = np.Inf
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=0.8, 
                                                     patience=4, 
                                                     verbose=True, 
                                                     eps=1e-6)     
    grad_scaler = torch.cuda.amp.GradScaler(enabled=params.amp) # https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float16
  
    # ---------------------------------------------------------------------------------------
    if params.wandb:
        # (Initialize logging)
        experiment = wandb.init(project='U-Net1', resume='allow', anonymous='must')
        experiment.config.update(dict(epochs=params.epochs,
                                      batch_size=params.batch_size, 
                                      learning_rate=params.lr,
                                      val_percent=params.val_percent, 
                                      time_per_epoch= int(0),
                                      amp=params.amp))
    else:
        # initialize offline trackers
        tracker_train = TrainTracker("traintrackerabc", labels = (0,1))
        # tracker_eval = EvalPlotter("train_plotter", labels = labels)

    logging.info(f'''Starting training:
                 Epochs:          {params.epochs}
                 Batch size:      {params.batch_size}
                 Learning rate:   {params.lr}
                 Training size:   {n_train}
                 Validation size: {n_val}
                 Device:          {params.device}
                 ''')
    # ---- Training loop ----
    global_step = 0
    save_checkpoint = False # TODO 8h limit einbauen
    time_stamp0 = time.time()  # TODO 8h limit einbauen
    # burn in period
    for b in range(params.burn_in):
        time_stamp1 = time.time()
        print("Epoch: ", b + 1, " at: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_stamp1)))
        train_res = run_training_epoch(train_loader,
                                       model,
                                       criterion,
                                       optimizer, 
                                       grad_scaler,
                                       cuda=cuda,
                                       verbose=True, 
                                       amp_enabled= params.amp)
        time_stamp2 = time.time()
        time_per_epoch =  time_stamp2 - time_stamp1
        print(f'At epoch {b + 1}: Loss={train_res["loss_avg"]} | dice={train_res["dice_coef_avg"]}')
        if params.wandb:
            experiment.log({
                'train loss': train_res["loss_avg"],
                'train dice': train_res["dice_coef_avg"],
                'epoch': b + 1,
                'time_per_epoch': time_per_epoch
                })
            
    # after warm up, train SWAG, too
    for e in range(params.burn_in + 1, params.epochs):
        # ---- timer ----
        time_stamp1 = time.time()
        print("Epoch: ", e, " at: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_stamp1)))
        train_res = run_training_epoch(train_loader,
                                       model,
                                       criterion,
                                       optimizer, 
                                       grad_scaler,
                                       cuda=cuda,
                                       verbose=True, 
                                       amp_enabled= params.amp)
        time_stamp2 = time.time()
        time_per_epoch =  time_stamp2 - time_stamp1
        print(f'At epoch {e}: Loss={train_res["loss_avg"]} | dice={train_res["dice_coef_avg"]}')
        if params.wandb:
            experiment.log({
                'train loss': train_res["loss_avg"],
                'train dice': train_res["dice_coef_avg"],
                'epoch': e,
                'time_per_epoch': time_per_epoch
                })
        # (re-)initialize flag for potentially saving this epoch's model if it performs better than ever before
        save_new_best_model = False             
        loss = []
        dice = []
        mc_predictions = []
        swag_model.collect_model(model)
        if ((e+1) % params.epochs_to_check_val_at == 0):
        # Validation evaluation
        # Get Monte Carlo samples from network to estimate the uncertainty (on validation set) during training 
            for mc in range(params.mc_samples):
                if params.verbose_terminal: print(f'collecting MC sample no. {mc} ')
                swag_model.sample(scale = 0.0, cuda=params.use_cuda)
                bn_update(train_loader, swag_model, cuda=params.use_cuda)
                mc_prediction,  target = predictions(val_loader, swag_model, cuda=params.use_cuda)
                swag_res = eval(val_loader, swag_model, criterion, cuda=params.use_cuda)
                loss.append(swag_res['loss_avg'])
                dice.append(swag_res['dice_coef_avg'])
                mc_predictions.append(mc_prediction)
            # tracker_train.update(mc_predictions)
            val_mean_swag_prediction_loss = np.array(loss).mean()
            val_mean_swag_prediction_dice = np.array(dice).mean()
            
            scheduler.step(val_mean_swag_prediction_loss)
            val_res_freq = eval(val_loader, model, criterion, cuda=params.use_cuda)
            
            if params.verbose_terminal:
                print(f'At epoch {e + 1} validation loss frequentist model: Loss={val_res_freq["loss_avg"]}')
                print(f'At epoch {e + 1} validation loss from SWAG: Loss={val_mean_swag_prediction_loss}')
                print(f'At epoch {e + 1} validation dice score frequentist model: Loss={val_res_freq["dice_coef_avg"]}')
                print(f'At epoch {e + 1} validation dice score from SWAG: Loss={val_mean_swag_prediction_dice}')
                
            if (val_mean_swag_prediction_loss < best_loss):
                best_loss =  val_mean_swag_prediction_loss
                save_new_best_model = True
                best_epoch = e
                
            if e - params.tolerance > best_epoch:
                break_training = True
                
            if save_new_best_model:
                
                save_new_best_model = False # reset flag
                
        division_step = (n_train // (10 * params.batch_size))
        if division_step > 0:
            if global_step % division_step == 0:
                histograms = {}
                for tag, value in model.named_parameters():
                    tag = tag.replace('/', '.')
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                if params.wandb:
                    experiment.log(
                        {
                        'learning rate': optimizer.param_groups[0]['lr'],
                        f'mean loss val.set SWAG {criterion.__name__}': val_mean_swag_prediction_loss,
                        f'mean loss val.set Freq {criterion.__name__}': val_res_freq["loss_avg"],
                        'validation Dice Bayesian': val_mean_swag_prediction_dice,
                        'validation Dice Vanilla': val_res_freq["dice_coef_avg"],
                        # 'images': wandb.Image(images[0].cpu()),
                        # 'masks': {
                        #     'true': wandb.Image(true_masks[0].float().cpu()),
                        #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        # },
                        'step': global_step,
                        'epoch': e,
                        **histograms}
                        )
        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')
    print("Done")

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # parse parameters from console
    params = get_parser().parse_args()
    assert_parser_params(params)
    train_unet(params)