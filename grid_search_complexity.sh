#!/bin/bash

START_FOLD=0
END_FOLD=4
COMPLEXITY_SCRIPT="complexity_model.py"
GEN_DATA_SCRIPT="generate_dataset.py"
CONTRASTIVE_SCRIPT="contrastive_embedding_model.py"
VISUALIZE_SCRIPT="visualization.py"
RANKED_SCRIPT="multilabel_rank_model.py"
EVAL_SCRIPT="eval_multilabel_model.py"
CHECKPOINT_DIR="checkpoints"
DATASET="APPS"
MODEL="gpt-3.5-turbo"
COMPLEXITY=0.1452
STEP=2
START_TOP_K=1
END_TOP_K=1


for FOLD in $(seq $START_FOLD $END_FOLD); do
  RESULT_DIR="result/PET_model_each_fold_result/multilabel_$DATASET\_$MODEL\_fold$FOLD"
  mkdir -p $RESULT_DIR


  echo "Running $COMPLEXITY_SCRIPT with --dataset $DATASET --model $MODEL --fold $FOLD..."
  python3 $COMPLEXITY_SCRIPT --dataset $DATASET --model $MODEL --fold $FOLD --complexity $COMPLEXITY

  
  echo "Running $GEN_DATA_SCRIPT..."
  python3 $GEN_DATA_SCRIPT --dataset $DATASET --model $MODEL --fold $FOLD --complexity $COMPLEXITY
  echo "Completed $GEN_DATA_SCRIPT"


  echo "Running $CONTRASTIVE_SCRIPT..."
  CUDA_VISIBLE_DEVICES=3 python3 $CONTRASTIVE_SCRIPT > $RESULT_DIR/contrastive_embedding_output.txt
  echo "Completed $CONTRASTIVE_SCRIPT."

  echo "Running $VISUALIZE_SCRIPT..."
  python3 $VISUALIZE_SCRIPT --dataset $DATASET --model $MODEL --fold $FOLD> $RESULT_DIR/visualization_output.txt
  echo "Completed $VISUALIZE_SCRIPT."

  echo "Running $RANKED_SCRIPT..."
  python3 $RANKED_SCRIPT > $RESULT_DIR/ranked_model_output.txt
  echo "Completed $RANKED_SCRIPT."

  for TOP_K in $(seq $START_TOP_K $STEP $END_TOP_K); do
    echo "Running $EVAL_SCRIPT with --top_k $TOP_K..."
    python3 $EVAL_SCRIPT --top_k $TOP_K > $RESULT_DIR/eval_model_output_${COMPLEXITY}_top_k_${TOP_K}.txt
    echo "Completed $EVAL_SCRIPT with --top_k $TOP_K."
  done

  echo "Completed all steps for complexity $COMPLEXITY."
  
done
echo "All scripts have been executed for all specified complexities. Results are saved in the $RESULT_DIR directory."
