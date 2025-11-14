#!/bin/bash

# Constantes com valores padrão
DEFAULT_INPUT_DIR="inputs"
DEFAULT_SOLVER="glpk"
DEFAULT_TIME_LIMIT=""
DEFAULT_THREADS=""

OUTUPUT_RESULT_DIR="output_milp/"
OUTUPUT_LOG_DIR="log_milp/"

# Uso do script
show_usage() {
    echo "Usage: $0 [input_directory] [solver] [timelimit] [threads]"
    echo "  input_directory: Directory with input files (required)"
    echo "  solver:          Solver to use {cbc,cplex,gurobi,glpk,scip} (default: $DEFAULT_SOLVER)"
    echo "  timelimit:       Time limit in seconds (optional)"
    echo "  threads:         Number of threads (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 ./inputs"
    echo "  $0 ./inputs gurobi"
    echo "  $0 ./inputs cbc 300"
    echo "  $0 ./inputs gurobi 600 4"
}


input_directory="${1:-$DEFAULT_INPUT_DIR}"
solver="${2:-$DEFAULT_SOLVER}"
timelimit="${3:-$DEFAULT_TIME_LIMIT}"
threads="${4:-$DEFAULT_THREADS}"



# Verificar se o diretório existe
if [ ! -d "$input_directory" ]; then
    echo "Error: Directory '$input_directory' not found!"
    exit 1
fi

mkdir -p $OUTUPUT_RESULT_DIR
mkdir -p $OUTUPUT_LOG_DIR

#'Maximize_SFCs'
for mipgap in  0.2 0.0001 ; do
    mipgap_scaled=$(echo "$mipgap * 10000" | bc | cut -d. -f1)
    # Pad with zeros for consistent width
    mipgap_value=$(printf "%05d" "$mipgap_scaled")
    # Processar todos os arquivos .txt no diretório
    for arquivo in "$input_directory"/*.txt; do
        if [ -f "$arquivo" ]; then
            # Extrair nome base do arquivo sem extensão
            nome_base=$(basename "$arquivo" .txt)
            nome_saida="${OUTUPUT_RESULT_DIR}/${nome_base}_${solver}_${mipgap_value}_results.json"
            log_saida="${OUTUPUT_LOG_DIR}/${nome_base}_${solver}_${mipgap_value}_log.txt"

            echo "Processing: $arquivo"
            echo "Output: $nome_saida"
            echo "Log: $log_saida"
            echo "Solver: $solver"
            echo "Objective: $objective"
            echo "MIP Gap: $mipgap"

            # Construir comando base
            comando="python3 SolverMILP.py --solver \"$solver\" --input \"$arquivo\" --output \"$nome_saida\" --mip-gap \"$mipgap\" "

            # Adicionar timelimit se fornecido e não vazio
            if [ -n "$timelimit" ]; then
                comando="$comando --time-limit \"$timelimit\""
                echo "Time limit: ${timelimit}s"
            fi

            # Adicionar threads se fornecido e não vazio
            if [ -n "$threads" ]; then
                comando="$comando --threads \"$threads\""
                echo "Threads: $threads"
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
done

echo "All files processed!"
