# Credit： We Implement Decaying LoRA by modifying the codes of Original LoRA study

## Steps to reproduce our results
### Create and activate conda env
```console
conda env create -f environment.yml
```
### Install the pre-requisites
lora:
```console
pip install -e ..
```
NLU:
```console
pip install -e .
```
### Start the experiments
```console
roberta_base_mnli.sh
roberta_base_sst2.sh
roberta_base_mrpc.sh
roberta_base_cola.sh
roberta_base_qnli.sh
roberta_base_qqp.sh
roberta_base_rte.sh
roberta_base_stsb.sh
```

### Evaluate the checkpoints
```console
python -m torch.distributed.launch --nproc_per_node=1 examples/text-classification/run_glue.py \
--model_name_or_path microsoft/roberta_base \
--lora_path ./roberta_base_mnli_lora.bin\
--task_name mnli \
--do_eval \
--output_dir ./output \
--apply_lora \
--lora_r 16 \
--lora_alpha 32
```

### Enable Cutoff/R-drop for data augmentation
```console
mnli.cutoff.sh
mnli.rdrop.sh
```

## Citation
```
@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
