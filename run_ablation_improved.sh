#!/bin/bash
cd "./$(dirname "$0")"
CONFIG="configs/improved_25plus.yaml"
SEED=42
RESULTS_DIR="results"
ABLATION_CSV="$RESULTS_DIR/ablation_results.csv"

# 写入 CSV 头部
echo "exp,bleu,val_loss" > $ABLATION_CSV

run_exp() {
    local name=$1
    local ablation_flag=$2

    echo "========== Running: $name =========="
    if [ -z "$ablation_flag" ]; then
        python3 src/train_improved.py --config $CONFIG --seed $SEED
    else
        python3 src/train_improved.py --config $CONFIG --seed $SEED --ablation $ablation_flag
    fi

    # 找到最新的运行文件夹
    # (使用 ls -td ... | head -n 1 总是能找到最新的)
    latest_dir=$(ls -td $RESULTS_DIR/run_${name}/*/ 2>/dev/null | head -n 1)
    if [ -z "$latest_dir" ]; then
        # 兼容 baseline 文件夹名称
        latest_dir=$(ls -td $RESULTS_DIR/run_baseline/*/ 2>/dev/null | head -n 1)
    fi

    # --- 这是修复的部分 ---
    if [ -f "$latest_dir/train_log.csv" ]; then
        
        # 使用 awk 和 sort 来查找最佳 val_loss (第3列) 对应的行
        # NR > 1: 跳过 CSV 头部
        # $3 != "": 确保 val_loss 列不为空
        # sort -t',' -k3,3n: 按第3列 (val_loss) 进行数值 (n) 排序
        # head -n 1: 选取最小 val_loss 对应的行
        best_line=$(awk -F',' 'NR > 1 && $3 != "" {print $0}' "$latest_dir/train_log.csv" | sort -t',' -k3,3n | head -n 1)

        if [ -z "$best_line" ]; then
            echo "$name,,," >> $ABLATION_CSV
            echo "⚠️ No valid log data found for $name in $latest_dir/train_log.csv"
        else
            # 从 $best_line 中提取数据
            val_loss=$(echo "$best_line" | cut -d',' -f3)
            bleu=$(echo "$best_line" | cut -d',' -f4)
            echo "$name,$bleu,$val_loss" >> $ABLATION_CSV
            echo "✅ Recorded $name (best val_loss) → BLEU=$bleu, val_loss=$val_loss"
        fi
    else
        echo "$name,,," >> $ABLATION_CSV
        echo "⚠️ No train_log.csv found for $name in $latest_dir"
    fi
}

# --- 运行所有实验 ---
run_exp "baseline" ""
run_exp "no_positional" "no_positional"
run_exp "no_residual" "no_residual"
run_exp "reduce_heads" "reduce_heads"

echo "==============================="
echo "✅ All ablation runs complete."
echo "📊 Results saved to: $ABLATION_CSV"
cat $ABLATION_CSV