#!/bin/bash

# Constantes com valores padrão
DEFAULT_INPUT_DIR="inputs"
DEFAULT_SOLVER="glpk"
DEFAULT_TIME_LIMIT=""
DEFAULT_THREADS=""

OUTUPUT_RESULT_DIR="output_DQN/"
OUTUPUT_LOG_DIR="log_DQN/"

# Uso do script
show_usage() {
    echo "Usage: $0 [input_directory] [solver] [timelimit]  "
    echo "  input_directory: Directory with input files (required)"
    echo "  solver:          DQN"
    echo "  timelimit:       Time limit in seconds (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 ./inputs"
}


input_directory="${1:-$DEFAULT_INPUT_DIR}"
solver="DQN"
timelimit="${3:-$DEFAULT_TIME_LIMIT}"

# Verificar se o diretório existe
if [ ! -d "$input_directory" ]; then
    echo "Error: Directory '$input_directory' not found!"
    exit 1
fi

mkdir -p $OUTUPUT_RESULT_DIR
mkdir -p $OUTUPUT_LOG_DIR

#'Maximize_SFCs'
#for mipgap in  0.2 0.0001 ; do
    mipgap_scaled=$(echo "$mipgap * 10000" | bc | cut -d. -f1)
    # Pad with zeros for consistent width
    mipgap_value=$(printf "%05d" "$mipgap_scaled")
    # Processar todos os arquivos .txt no diretório
    for arquivo in "$input_directory"/*.txt; do
        if [ -f "$arquivo" ]; then
            # Extrair nome base do arquivo sem extensão
            nome_base=$(basename "$arquivo" .txt)
            nome_saida="${OUTUPUT_RESULT_DIR}/${nome_base}_DQN_results.json"
            log_saida="${OUTUPUT_LOG_DIR}/${nome_base}_DQN_log.txt"

            echo "Processing: $arquivo"
            echo "Output: $nome_saida"
            echo "Log: $log_saida"
            echo "Solver: $solver"
            echo "Objective: $objective"
            echo "MIP Gap: $mipgap"

            # Construir comando base
            comando="python3 SolverDQN/SolverWithDQN.py --input \"$arquivo\" --output \"$nome_saida\" "

            # Adicionar timelimit se fornecido e não vazio
            if [ -n "$timelimit" ]; then
                comando="$comando --time-limit \"$timelimit\""
                echo "Time limit: ${timelimit}s"
            fi


            # Adicionar tee para log
            comando="$comando | tee \"$log_saida\""

            echo "Command: $comando"
            echo "---"

            # Executar comando
            eval $comando

            # Verificar se executou com sucesso
            if [ $? -eq 0 ]; then
                echo "✓ Successfully processed: $arquivo $solver $objective"
            else
                echo "✗ Failed to process: $arquivo $solver $objective"
            fi

            echo "=========================================="
        fi
    done
#done

echo "All files processed!"
