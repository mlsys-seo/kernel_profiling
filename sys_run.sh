
models="densenet121 densenet201 mobilenet_v3_large resnet18 resnet50 efficientnet_v2_m efficientnet_v2_l convnext_tiny convnext_small convnext_base convnext_large shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0"
train_batch_sizes="128"
infer_batch_sizes="32"

for model in ${models}
do
    export CUDA_VISIBLE_DEVICES=0
    nsys profile \
    -t cuda,osrt,nvtx,cudnn,cublas \
    --cuda-graph-trace=node \
    -o ./data/nsys/train_${model}_${train_batch_sizes} \
    --export sqlite \
    -w true \
    -f true \
    python kernel_profiling.py --model_name ${model} --batch_size 64 --do_train True &
    unset CUDA_VISIBLE_DEVICES

    export CUDA_VISIBLE_DEVICES=1
    nsys profile \
    -t cuda,osrt,nvtx,cudnn,cublas \
    --cuda-graph-trace=node \
    -o ./data/nsys/infer_${model}_${infer_batch_sizes} \
    --export sqlite \
    -w true \
    -f true \
    python kernel_profiling.py \
    --model_name ${model} --batch_size 16 --do_train False &
    unset CUDA_VISIBLE_DEVICES
    wait
done
