export CUDA_VISIBLE_DEVICES=3
# models="convnext_base densenet121 efficientnet_v2_s mobilenet_v3_large resnet50"
models="densenet121"
# python kernel_profiling.py --model_name=${model} --batch_size=1 --do_train=True #> ${model}.txt
for model in ${models}; do
    nsys profile \
    -t cuda,osrt,nvtx,cudnn,cublas \
    --cuda-graph-trace=node \
    -o train_${model}_${train_batch_sizes} \
    -w true \
    -f true \
    python kernel_profiling.py --model_name=${model} --batch_size=1 --do_train=True #> ${model}.txt
done
unset CUDA_VISIBLE_DEVICES