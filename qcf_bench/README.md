# QCF (Quantum Chemistry Federation) Bench

This is a test bench that allows for the easy testing of quantum chemistry models over federated learning. It is light-weight and feature sparse, designed for easy extension and clear directions to add custom datasets/models.

## Running as-is

### Setup

First, you need some version of conda installed on your computer (we recommend [miniforge](https://github.com/conda-forge/miniforge)). Then, you will need to create and setup the testbench environment by running the commands below.

```bash
conda env create -f environment.yml
conda activate qcfb
pip install fedml # fedml is not available on conda, that's why we install it via pip
```

Next, you might want to ensure the dataset is downloaded correctly. We have files for the ogb_molhiv dataset in the repository, but we have run into odd issues before about this. To do so, follow the commands below

```bash
python
>>> dataset = MyGraphPropPredDataset(name="ogbg-molhiv") # run inside the interactive python environment
```

### Parameters

You may adjust your parameters in the `config/fedml_hiv.yaml` file (or any other file, and then modify the script within `run_bench.sh`). This includes model settings, dataset (should you add custom datasets and use this argument), GPU options, and wandb args.

Adding custom parameters is as easy as adding a line in the file, no other requirement needed. To use these arguments within the test bench, ensure that the variable `args` is available within your scope and use the syntax `args.parameter_name` for the value.

#### GPU specific options

If you want to only use CPU, simply turn the parameter `using_gpu` to false.

The default settings we have included run with 1 GPU with one worker. If you want to adjust any GPU settings, here are some explanations:

- `worker_num`: the amount of *processes* (not clients) that you want to run, can save time with parallelization (we believe, not tested thoroughly)
- `using_gpu`: whether you want to use the GPU or just CPU
- `gpu_mapping_file`: the file for num_processes-to-GPU mappings
- `gpu_mapping_key`: the mapping to use within `gpu_mapping_file`

The GPU mappings in `gpu_mapping_file` work like such:

```yaml
mapping_key_name:
    host_name: [num_processes_on_GPU0, num_processes_on_GPU1, ..., num_processes_on_GPUn]
```

`host_name` should not have affect, since we are running simulation-based federated learning. The list should be sum to `worker_num + 1` (workers and master).

#### wandb

To get wandb working, you may need to run `wandb login` in your terminal, and then change the appropriate wandb settings to connect tot he correct key/team/project/run endpoint. Refer to [wandb documentation](https://docs.wandb.ai/) for more details.

### Training

To run the test_bench with the specified args within `config/fedml_hiv.yaml` (or custom yaml file, and edit the `run_bench.sh` script), use the following command (where `worker_num` is the value of that argument).

```bash
./run_bench.sh [worker_num]
```

So if `workers_num` is 1, then the command would be `./run_bench.sh 1`

## Running with a custom dataset/model

### Adding a dataset

To add a custom dataset, you need to modify `data/data_loader.py`. Currently, the dataset is hardcoded to be the `ogb-molhiv` dataset, but you can either hard code it to your dataset or use the argument `args.dataset` to allow this to be modified via the config file. Here is where you would split the dataset either with your own sorting method or pre-selected indices (we have the latter).

See `data/data_loader.py` for more specific documentation.

You also need to modify the way the trainers/aggregators evaluate and the criterion. It's hardcoded specific to the `ogb-molhiv` dataset, but you can once again either re-hardcode it or use the dataset argument to change the evaluator. See the `ogb` trainer/aggregator examples for more documentation.

### Adding a model

To add a model, there are three steps.

First, add the model version that *both* inherits from `nn.Module` and does not need any outside commands to run (e.g., `fairseq-train`). You may need to modify your model to do so--it sucks--but it is the only way to drag and drop into FedML as far as we know.

Then, create a trainer and an aggregator into the `trainer` folder. The trainer will train the model (funny enough) and the aggregator will then test the all the models against the test set(s), run FedAvg, and compute the final score for that communication round. Mostly, you can copy the examples that are already given, only with some tweaks to ensure the model training steps are correct, the score is computed correctly, and any extra logging you want with logging/wandb is added. See thre `ogb` examples for documentation.

Finally, modify `fedml_bench.py` to enable the model option. It is self-explanatory to follow suit with what is given within `create_model`, just ensure you are also adding the arguments you want to your config file (e.g., `fedml_hiv.yaml`).

## Directories

- `config`: place to store config files
- `data`: files relating to downloading/loading the dataset
- `dataset`: automatically downloaded files for the `ogb-molhiv` dataset
- `model`: the models installed on this test bench
- `trainer`: the trainers/aggregators for the installed models within `model`
- `wandb`: automatically generated file for the most-recent run