
SCRIPT_NAME="predictv2.py"
cd /home/mcipriano/projects/sirius/src/sirius
mamba activate sirius

# Run the script with CUDA_VISIBLE_DEVICES=1 for shard 0 to 9
for i in {0..9}; do
  CUDA_VISIBLE_DEVICES=1 python $SCRIPT_NAME --shard $i &
done

# Run the script with CUDA_VISIBLE_DEVICES=2 for shard 10 to 19
for i in {10..19}; do
  CUDA_VISIBLE_DEVICES=2 python $SCRIPT_NAME --shard $i &
done

# Run the script with CUDA_VISIBLE_DEVICES=3 for shard 20 to 29
for i in {20..29}; do
  CUDA_VISIBLE_DEVICES=3 python $SCRIPT_NAME --shard $i &
done

# Wait for all background processes to complete
wait
