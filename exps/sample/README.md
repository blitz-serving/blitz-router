# How to reproduce the results of a feature

## 🎯 Objective

Example: optimize the loading process of model parameters.

## 📌 Reproduce Instructions

Follow the instructions step by step.

If you want to generate configurations and run test cases, run the following command **under workspace root directory**. The runner will create a directory to store logs and snapshot all uncommitted files.

```bash
python scripts/batchv2/main.py --templates exps/sample/run.toml
```

Add `--dry-run` in the end to skip running executions but just generate the configurations.

```bash
python scripts/batchv2/main.py --templates exps/sample/run.toml --dry-run
```

The runner has the checkpoint feature, which can be used to continue interrupted or failed jobs. Each time you run a template, the runner will generate a checkpoint file under it working directory and record finished jobs. If the execution is interrupted, you can resume it by running the following command:

```bash
python scripts/batchv2/main.py --checkpoint log_home/202506251046_eval
```

To force the runner to run all jobs in checkpoint mode, add `--force` flag. Attention the the `--force` flag will remove the checkpoint file and your next run will not skip previous finished jobs.

```bash
python scripts/batchv2/main.py --force --checkpoint log_home/202506251046_eval
```
