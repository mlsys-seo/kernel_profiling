export CUDA_VISIBLE_DEVICES=3
models="densenet121 resnet50 mobilenet_v3_large"
for model in ${models}; do
    python kernel_profiling.py --model_name=${model} --batch_size=1
done
unset CUDA_VISIBLE_DEVICES