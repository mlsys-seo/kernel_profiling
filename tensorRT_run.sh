models="densenet121"
batch_sizes="64"

for model in ${models}
do
    nsys profile \
    -t cuda,osrt,nvtx,cudnn,cublas \
    -o test \
    -w true \
    -f true \
    python test_tensorRT2.py
done


    # --cuda-graph-trace=node \