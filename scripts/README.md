## how to test(really one button run!)
```bash
./scripts/start_azure.sh /nvme/zkx/blitz-infer-pack/config/dense_vllm_dp6.toml /nvme/zkx/blitz-infer-pack/config/dense_router.toml /nvme/zkx/blitz-infer-pack/config/dense_24clients.toml 
```
向start_azure脚本传入三个参数，分别对应启动vllm、router和client的配置路径

start_azure的逻辑是在/nvme/lmetric/logs路径下根据datetime创建目录，拷贝三个配置文件，并创建tmux session按照顺序启动vllm、router、client。并且每十分钟将router.log和client.jsonl checkpoint

## smart runner
给定的toml通过修改的smart runner启动，在这里能够对原有的runner通过命令行传入全局变量，替换掉toml中定义的全局配置，toml里的全局配置项又能替换掉每个app的配置项，因此不需要修改toml中的任何配置(例如log路径、vllm端口)即可运行一次新的测试

对于client可能需要修改toml全局变量中的trace路径，从而更改使用的trace


## env
```bash
git clone git@github.com:blitz-serving/modified-vllm.git
git checkout step-reporter-v2

VLLM_USE_PRECOMPILED=1 pip install -e .
```

修改config/dense_vllm_dp6.toml config/dense_router.toml config/dense_24clients.toml 中所有可执行文件路径、日志写入路径等。