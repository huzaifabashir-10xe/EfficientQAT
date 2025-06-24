# This file is modified from https://github.com/artidoro/qlora/blob/main/qlora.py 
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np
import importlib
from packaging import version

import torch
import transformers
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    LlamaTokenizer
)


from datautils_block import test_ppl
from datautils_e2e import make_data_module
from bitsandbytes.optim import AdamW
import os
import utils
from quantize.int_linear_real import load_quantized_model,QuantLinear
from pathlib import Path




def is_ipex_available():
    """
    Checks whether Intel Extension for PyTorch (IPEX) is installed and compatible with the current PyTorch version.

    Returns:
        bool: True if IPEX is available and version-compatible with installed PyTorch, else False.
    """

    def get_major_and_minor_from_version(full_version):
        """
        Extracts 'major.minor' version from a full semantic version string (e.g., '2.1.0' -> '2.1').

        Args:
            full_version (str): Full version string (e.g., '2.1.0').

        Returns:
            str: 'major.minor' part of the version.
        """
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    # Get the installed PyTorch version
    _torch_version = importlib.metadata.version("torch")

    # Check if IPEX is even importable (installed)
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False

    # Try to get the installed IPEX version
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False  # IPEX not actually installed

    # Extract only major.minor for compatibility comparison
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)

    # Warn if IPEX and PyTorch versions don't match in major.minor
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False

    return True  # IPEX is installed and version-compatible

    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    """
    Holds configuration arguments related to model loading and family metadata.

    This dataclass is used by HuggingFace's `HfArgumentParser` to parse command-line args
    for model selection and trust/auth settings during loading.

    Attributes:
        quant_model_path (str): Filesystem path to the quantized model produced by Block-AP.
            Used when loading a pre-quantized model for evaluation or fine-tuning.
        
        model_family (str): A label used for organizing and caching datasets,
            useful to distinguish models like 'llama-2', 'opt', etc.
        
        trust_remote_code (bool): Whether to trust and execute arbitrary code
            from the model repo's `modeling.py` (via `from_pretrained()`).
            Needed for some custom implementations on HuggingFace Hub.
        
        use_auth_token (bool): Whether to use a HuggingFace authentication token (e.g., for private models).
    """
    quant_model_path: Optional[str] = field(
        default="",
        metadata={"help": "path of the quantization model by Block-AP."}
    )
    model_family: Optional[str] = field(
        default="llama-2",
        metadata={"help": "for the saving of dataset cache for faster experiments"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    """
    Stores all dataset-related configuration options used during training and evaluation.

    This dataclass controls dataset selection, sample limits, sequence length, 
    evaluation protocols, and preprocessing parallelism.
    """
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    conv_temp: str = field(
        default='llama-2',
        metadata={"help": "Conversation template, only useful with deita datasets"}
    )
    mask_use: bool = field(
        default=True, metadata={"help": "mask the loss to role in dialogue datas"}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|redpajama]"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    """
    Extends HuggingFace's Seq2SeqTrainingArguments with custom training options
    for EfficientQAT quantization-aware fine-tuning and evaluation.

    This class adds support for quantization configuration, memory limits,
    optimizer choices, and evaluation-specific toggles like MMLU or perplexity.
    """
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_ppl_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    pt_context_len: int = field(
        default=1024,
        metadata={"help": "language modeling length."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    wbits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    group_size: int = field(
        default=64,
        metadata={"help": "How many group size to use."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    resume_from_checkpoint: str = field(default=None, metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=0, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-5, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='cosine', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=5, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    """
    Stores generation-related configuration options used during evaluation or prediction.

    These hyperparameters control decoding strategy and token-level behavior for models
    that use `generate()` — e.g., for text completion, summarization, instruction following, etc.
    """
    
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def get_accelerate_model(args, checkpoint_dir):
    """
    Loads and prepares a quantized model (via EfficientQAT) for training or evaluation using Accelerate.
    
    Key operations:
        - Detects available compute device (GPU/XPU).
        - Loads the quantized model and tokenizer.
        - Sets appropriate compute dtype (fp16/bf16/fp32).
        - Ensures model is parallelizable and correctly tokenized.
        - Applies parameter freezing, casting, and gradient checkpointing if required.

    Args:
        args (Namespace): Parsed command-line arguments (TrainingArguments or similar).
        checkpoint_dir (str): Path to the checkpoint directory.

    Returns:
        model (nn.Module): Prepared and configured model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer ready for input formatting.
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    
    model, tokenizer = load_quantized_model(args.quant_model_path,args.wbits, args.group_size)
    tokenizer.model_max_length = args.pt_context_len
    
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))        
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.cuda()
    model.train()
        
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )    

    # TODO
    # if 'llama1' in args.model_name_or_path or 'llama2' in args.model_name_or_path or 'llama-1' in args.model_name_or_path or 'llama-2' in args.model_name_or_path:
    if isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })


    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)
            
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)    
            
        model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        # if isinstance(module, QuantLinear):
        #     # transfer trainable step size into float32
        #     module.scales.data = module.scales.data.to(torch.float32)
        if 'norm' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                    # module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*'*80)
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print('*'*80)
    if args.wbits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg






def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    """
    Main training entry point for end-to-end fine-tuning of a quantized model.

    This function does the following:
    1. Parses model, data, training, and generation arguments.
    2. Loads a quantized model using `get_accelerate_model()`.
    3. Initializes dataset and trainer objects.
    4. Configures optimizer to train quantization parameters (scales).
    5. Optionally evaluates perplexity or benchmark accuracy (e.g., MMLU).
    6. Trains and/or evaluates the model, and saves predictions/metrics to disk.

    It supports:
        - Loading quantized models (Block-AP outputs).
        - Training only scale parameters in QuantLinear.
        - Perplexity evaluation (PPL) on wikitext2/C4.
        - LM-eval tasks (PIQA, ARC, HellaSwag, etc.).
        - MMLU benchmark evaluation.
    """

    # Step 1: Parse arguments into dataclasses
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # Convert GenerationArguments to Hugging Face config object
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    
    # Combine all argument namespaces into a single one
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # Step 2: Prepare logging directory and logger
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(args.output_dir)
    logger.info(args)
    
    # Step 3: Check if training was already completed
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    # Step 4: Load quantized model and tokenizer
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False      # disable caching for training
    print('loaded model')
    
    # Step 5: Set random seed for reproducibility
    set_seed(args.seed)

    # Step 6: Load dataset (train/eval/predict splits)
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    # Step 7: Configure optimizer to train only quantization parameters (scales)
    optimizer_grouped_parameters = []
    
    # Enable training for scale parameters inside QuantLinear layers
    for name, module in model.named_modules():
        # if isinstance(module, LoraLayer):
        if isinstance(module, QuantLinear) and not 'head' in name:
            module.scales.requires_grad = True
    
    # Collect all scale parameters for optimizer
    optimizer_grouped_parameters.append({
        'params': [p for n, p in model.named_parameters() if 'scale' in n],
        'weight_decay': 0.0, 'lr': args.learning_rate
        })
    
    optimizer = AdamW(optimizer_grouped_parameters)

    # Step 8: Initialize Hugging Face trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, None),
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},   # exclude test dataset
    )

    # Step 9: Optional perplexity evaluation callback (on wikitext2 and c4)
    if args.do_ppl_eval:
        class PPLvalCallback(transformers.TrainerCallback):
            @torch.no_grad()
            def on_evaluate(self, args=None, state=None, control=None, model=None, **kwargs):
                results = test_ppl(trainer.model, trainer.tokenizer, datasets=['wikitext2','c4'],ppl_seqlen=2048)
                logger.info(results)
                trainer.log(results)

        trainer.add_callback(PPLvalCallback)
    
    # Verifying the datatypes and parameter counts before training.
    # Step 10: Print number of trainable parameters by dtype
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}


    # Step 11: Training
    print(args.output_dir)
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Step 12: Evaluation on validation set
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    #  Step 13: Prediction on test set and decoding output
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Save predictions as JSONL
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    # Step 14: Save combined metrics (train/eval/predict)
    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    # Step 15: Run LM-eval tasks if requested
    if args.eval_tasks != "" or args.do_mmlu_eval:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

    if args.eval_tasks != "":
        task_list = args.eval_tasks.split(',')
        lm_eval_model = HFLM(pretrained=model, batch_size=32)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')

    # Step 16: Run MMLU (Massive Multitask Language Understanding) evaluation.
    # One of the most comprehensive and challenging benchmarks used to evaluate the zero-shot reasoning and generalization ability of language models.
    if args.do_mmlu_eval:
        lm_eval_model = HFLM(pretrained=model, batch_size=16)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=['mmlu'],
        num_fewshot=5,
        task_manager=task_manager,
        cache_requests=True,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in results['results']:
            total_acc += results['results'][task]['acc,none']
        logger.info(f"Average MMLU Acc: {total_acc/len(results['results'])*100:.2f}%")

if __name__ == "__main__":
    train()
