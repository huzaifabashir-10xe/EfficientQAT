import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os

def update_dataset(layer, dataset, dev, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                dataset.update_data(index,new_data)

                    
def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    """
    Run block-wise quantization-aware training (QAT) on all transformer layers.

    This function implements Phase 1 of EfficientQAT (Block-AP). It quantizes each transformer
    block one at a time using fake quantization and minimizes reconstruction loss between
    full-precision and quantized outputs. The process supports training both quantization parameters
    (scale and zero-point) and optionally full-precision weights.

    Steps involved:
        1. Intercept and save inputs to each transformer block for calibration.
        2. Replace all nn.Linear layers with QuantLinear for QAT.
        3. Optimize quantization parameters (and optionally weights) using MSE loss.
        4. Finalize weights using in-place quantization.
        5. Optionally convert to real integer weights if --real_quant is specified.

    Args:
        model (transformers.PreTrainedModel): The pretrained language model to quantize.
        args (argparse.Namespace): Parsed command-line arguments with quantization settings.
        trainloader (List[Tuple[Tensor]]): Calibration dataset used to fit quantization.
        valloader (List[Tuple[Tensor]]): Validation dataset used for early stopping.
        logger (logging.Logger, optional): Logger to print progress and metrics.

    Returns:
        transformers.PreTrainedModel: The quantized model with updated layers.
    """
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False                  # turn-off cache (no KV cache) because it is not needed djuring calibration
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    # Model to be used to collect inputs to block-0
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        # Define unique disk cache paths using timestamp
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None

    # Create training and validation datasets to store intermediate layer inputs
    # They will store input tensors that reach block-0
    # They are not traditional token based datasets
    # In this case, since it is block-0 so all inputs will be saved in this form
    #✅ Why Do You Need This?
        # Because later, during block-wise training:
            # You need to pass a quantized version of the block the same input the original FP block received
            # And compare its output to the FP output
        # So you have to:
            # Run the model once to collect all inputs reaching block 0
            # Save them → That’s what this dataset is for
        # Without this, you would:
            # Have to rerun the full model multiple times
            # Waste compute and memory
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        """
        Custom wrapper that intercepts and saves the input to the first transformer block,
        then raises a ValueError to halt the forward pass early.
        """
        
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            # The model runs 1 sample
            # Catcher intercepts the input to block 0
            # That input is saved into this dataset at index i
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module # restore original block and removes the catcher 

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    layers[0] = layers[0].module
    if attention_mask is not None:
        # Expand attention mask to batch
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input (they are same at the first layer)
    # Why Do We Need quant_train_inps if it’s the Same?
    # Because during training:
        # You will modify quant_train_inps for block 1, block 2, etc.
        # It’s not the same as FP anymore
        # So quant_train_inps serves as a reusable buffer that gets updated block-by-block after each is quantized.
        # if saving to disk → copy FP .pt files to quant dataset path
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # If using RAM → clone the actual tensors from fp_train_inps into quant_train_inps
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # step 6: start training
    # layers refer to blocks  
    loss_func = torch.nn.MSELoss()

    # Loop overall blocks and quantize them one-by-one
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
                #replacing specifics of model blocks with quantlinear
                set_op_by_name(qlayer, name, quantlinear)  
                del module  
        qlayer.to(dev)
        # A version of block is created, which is ready for QAT
                
        # step 6.2: obtain output of full-precision model for MSE, to be considered as target label
            # Temporarily disable quantization.
            # Pass all calibration inputs through the FP block.
            # Save the output — this will act as the label.
        set_quant_state(qlayer,weight_quant=False) # deactivate quantization for obtaining ground truth, that will be used as target label

        if args.epochs > 0:
            # prepares ground-truth full precision outputs of current block
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids)
        set_quant_state(qlayer,weight_quant=True)  # activate quantization
        
        
        if args.epochs > 0:
            with torch.no_grad():
                # Even though the model is fake-quantized, we do:
                # Training with AMP (Automatic Mixed Precision)
                # So we switch weights to float32 (which AMP expects internally)
                # If you leave weights in float16, you risk numerical instability during backprop.
                qlayer.float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule

            # We will fill this list with:
            #     Quantization parameter group (if enabled)
            #     Weight parameter group (if enabled)
            # These will be passed to AdamW
            param = []

            # You must train something — either quantization params (scale, zero_point) or weights (actual linear weights).
            # If both are 0, that’s a misconfiguration.
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0

            # This is used to configure the learning rate scheduler.
            # E.g., if: epochs = 2, train_size = 4096, batch_size = 8. Then: total_training_iteration = 2 * 4096 / 8 = 1024 steps
            total_training_iteration = args.epochs * args.train_size / args.batch_size

            if args.quant_lr > 0:
                # Enable gradients for quantization parameters 
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                # PyTorch requires an optimizer object to use with CosineAnnealingLR. We create a dummy one just to drive the LR schedule.
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index     # to remeber quant_param index in optimizer param group
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if args.weight_lr > 0:
                # add weight parameters to optimizer
                set_weight_parameters(qlayer,True)
                param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index    # Track which param group refers to weights in the actual optimizer.
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)

            # Instantiate Real Optimizer
            # This optimizer is now responsible for:
                # Updating quant params (if enabled)
                # Updating weights (if enabled)
            # Each group may have its own LR.
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            # This is for mixed-precision (AMP) training.
            # It scales the loss to prevent underflows in half-precision.
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")
            
            # init Early stopping trackers
            best_val_loss = 1e6
            early_stop_flag = 0
            # this loop trains the current block(qlayer) for epochs number of times
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []      # stores gradient norms
                norm_list = []
                start_time = time.time()

                # Loop over training dataset (quantized inputs and full-precision targets)
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)): # quant_imps are input to block, fp_inps are target output
                    # obtain output of quantization model i.e Forward + Loss Computation (with AMP)
                    with torch.cuda.amp.autocast(): # Mixed-precision (autocast) saves memory.
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0] # quant_out is the quantized block's output
                        reconstruction_loss = loss_func(label, quant_out) # calculation of MSE
                        loss =  reconstruction_loss
                    
                    # safety check - if loss explodes, debugging kicks in
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    # gradients are scaled by loss_scaler to avoid underflows
                    # refers to a technique used in mixed-precision training (typically with torch.cuda.amp) 
                    # to prevent numerical underflows in gradients when using lower-precision (like float16 ot bfloat16)
                    # The idea is to temporarily multiply the loss (and hence gradients) by a large number, called the loss scale 
                    # (e.g., 1024 or 2¹⁵), to keep gradients in a representable range in float16.
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    # adjust lr. 
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                # step 6.5: calculate validation loss
                # Same idea as training loop, except:
                    # Wrapped in torch.no_grad()
                    # Just compute validation reconstruction loss
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())


                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                
                # If validation stops improving for N epochs, exit early.           
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        
        
        qlayer.half()                               # Convert weights to float16
        quant_inplace(qlayer)                       # Apply fake quantization in-place to the weights
        set_quant_state(qlayer,weight_quant=False)  # weight has been quantized inplace, so turn the quantization mode off

        # step 6.7: update inputs of quantization model for next block
        if args.epochs>0:
            # Feed the quant_train/val_inps  through the trained qlayer
            # Save the output as the new quant_train/val_inps
            # That becomes the input to the next block in the next iteration of the outer loop
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,quant_val_inps,dev,attention_mask,position_ids)
        layers[block_index] = qlayer.to("cpu")

        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        # If the user set --real_quant, this block replaces each QuantLinear with a packed integer version that holds: INT8/INT4 weights
        # Stored using scale + zero_point in a compressed form
        # this updates the next block for quantization
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
