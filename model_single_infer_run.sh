do_train_or_infers="infer"
# models="densenet121 mobilenetv31.0 resnet50 efficientnet_v2_m convnext_base" 
models="mobilenetv31.0 resnet50 efficientnet_v2_m convnext_base" 
# models="densenet201 resnet18 efficientnet_v2_l convnext_tiny convnext_small convnext_large shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0" 
batch_sizes="1 4 8 16 32 64 128"
percentiles="10 20 30 40 50 60 70 80 90 100"

> model_single_error.txt

for do_train_or_infer in $do_train_or_infers
do
    for model in $models
    do
        for batch_size in $batch_sizes
        do
            for percentile in $percentiles
            do
                export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentile
                echo ====================
                echo mps_usage: $percentile%
                python model_single.py \
                    --model $model \
                    --batch_size $batch_size \
                    --do_train_or_infer $do_train_or_infer \
                    --mps_percent $percentile
            done
        done
    done
done


#                 nsys profile \
#                 -t cuda,osrt,nvtx,cudnn,cublas \
#                 --cuda-graph-trace=node \
#                 -o $do_train_or_infer'_'$percentile'%_'$model'_'$batch_size \
#                 --export sqlite \
#                 -w true \
#                 -f true \