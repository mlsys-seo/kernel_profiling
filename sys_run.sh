export CUDA_VISIBLE_DEVICES=0
models="densenet121 densenet201 mobilenet_v3_large resnet18 resnet50 efficientnet_v2_m efficientnet_v2_l convnext_tiny convnext_small convnext_base convnext_large shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0"
train_batch_sizes="64"
infer_batch_sizes="16"

for model in ${models}
do
    for train_batch_size in $train_batch_sizes
    do
        nsys profile \
        -t cuda,osrt,nvtx,cudnn,cublas \
        --cuda-graph-trace=node \
        -o train_${model}_${train_batch_sizes} \
        --export sqlite \
        -w true \
        -f true \
        python kernel_profiling.py --model_name ${model} --batch_size ${train_batch_size} --do_train True
    done
done 
unset CUDA_VISIBLE_DEVICES &

export CUDA_VISIBLE_DEVICES=1
for model in ${models}
do
    for infer_batch_size in $infer_batch_sizes
    do
        nsys profile \
        -t cuda,osrt,nvtx,cudnn,cublas \
        --cuda-graph-trace=node \
        -o infer_${model}_${train_batch_sizes} \
        --export sqlite \
        -w true \
        -f true \
        python kernel_profiling.py \
        --model_name ${model} --batch_size ${infer_batch_size} --do_train False
    done
done
unset CUDA_VISIBLE_DEVICES