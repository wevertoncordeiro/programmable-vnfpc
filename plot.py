import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot SFC acceptance rate vs total SFCS')
    parser.add_argument('--input_dir', "-i", type=str, required=True,
                       help='Diretório de entrada contendo os arquivos JSON')
     
    return parser.parse_args()

 

def process_json_files(input_dir, y_variable):
    """Processa todos os arquivos JSON no diretório e organiza os dados"""
    # Estrutura para armazenar os dados: {total_nodes: {total_sfcs: [accepted_sfcs]}}
    # data = defaultdict(lambda: defaultdict(list))
    # data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Encontra todos os arquivos JSON no diretório
    json_pattern = os.path.join(input_dir, '**/*.json')
    json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        print(f"Nenhum arquivo JSON encontrado em {input_dir}")
        return None
    
    print(f"Encontrados {len(json_files)} arquivos JSON")
    
    for file_path in json_files:
        print(f"\n\n\n######################### {file_path} #######\n")
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            metadata = content.get('metadata', {})
            print(f"metadata: {metadata}")
            print(file_path)
            # Extrai os dados necessários
            input_file = file_path.split("/")[-1] #metadata.get('input_file', "").split("/")[-1]
            print(f"\n\tinput_file: {input_file}")
            seed = None
            # Split by underscores and find the part containing "seed"
            for part in input_file.split('_'):
                if part.startswith('seed'):
                    seed = part[4:]  # Remove "seed" prefix
                    print(int(seed))  # Output: 2
                if part.startswith('nodes'):
                    switches = int(part[5:])  # Remove "seed" prefix
                    npops = switches 
                    endpoints = switches*.5

            if "_DQN_" in input_file:
                solver = "DQN"
            else: 
                solver = "GUROBI"

            print(f"\tseed {seed} input_file: {input_file}")
            print(f"\nodes {switches} input_file: {input_file}")

            
            solution = content.get('solution', {})
            print(f"\nsolution: {solution}")
            total_sfcs = int(solution.get('total_sfcs'))
            print(f"\n\ttotal_sfcs: {total_sfcs}")

            if solver == "GUROBI":
                y_value = solution.get(y_variable)
                if y_value is None:
                    y_value = metadata.get(y_variable)

                mip_gap = metadata.get('mip_gap') 
                total_nodes = ( metadata['physical_topology_summary']['number_endpoints'] +
                                metadata['physical_topology_summary']['number_physical_switches'] +
                                metadata['physical_topology_summary']['number_network_pops'])
                    
                print(f"\n\tmip_gap: {mip_gap}")

            else:
                total_nodes = switches + npops + endpoints
                mip_gap = 0.0
                if y_variable == "accepted_sfcs":
                    y_value = metadata.get("accepted_sfcs")

                if y_variable == "sum_vnf_vs_instances_count":
                    y_value = metadata.get("vnf_instances") +  metadata.get("vs_instances")
                if y_variable == "vnf_instances_count":
                    y_value = metadata.get("vnf_instances")
                if y_variable == "vs_instances_count":
                    y_value = metadata.get("vs_instances")

                if y_variable == "elapsed_time":
                    y_value = metadata.get("solving_time")

            print(f"\n\t total_nodes: {total_nodes}")
            # accepted_sfcs = solution.get('accepted_sfcs')
            # print(f"accepted_sfcs: {accepted_sfcs}")
            # sum_vnf_vs_instances_count = solution.get('sum_vnf_vs_instances_count')
            # print(f"sum_vnf_vs_instances_count: {sum_vnf_vs_instances_count}")
            
            
            selected_total_nodes = {125, 250}
            selected_mip_gap = {0.0001, 0.2, 0.0}
            selected_total_sfcs = {1, 5, 10, 15, 20, 25}
             
            if total_sfcs is not None and y_value is not None and mip_gap is not None:
                
                if total_sfcs in selected_total_sfcs and total_nodes in selected_total_nodes and mip_gap in selected_mip_gap:
                    # data[total_nodes][total_sfcs][mip_gap].append(y_value)
                    data[total_nodes][total_sfcs][mip_gap][seed].append(y_value)
                

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Erro ao processar {file_path}: {e}")
            continue
    
    return data

def prepare_plot_data(data):
    """Prepara os dados para o plot, calculando média e intervalo de confiança 95%"""
    plot_data = {}
    
    for total_nodes, sfcs_data in data.items():
        for total_sfcs, mip_gaps_data in sfcs_data.items():
            for mip_gap, seeds_data in mip_gaps_data.items():
                # Collect all values across different seeds
                all_values = []
                for seed, values_list in seeds_data.items():
                    # If there are multiple values for the same seed, take their mean
                    if len(values_list) > 0:
                        seed_mean = np.mean(values_list)
                        all_values.append(seed_mean)
                
                if len(all_values) == 0:
                    continue
                
                # Calculate statistics
                mean_value = np.mean(all_values)
                std_value = np.std(all_values, ddof=1)  # sample standard deviation
                n = len(all_values)
                
                # Calculate 95% confidence interval
                if n > 1:
                    # confidence_interval = stats.t.interval(0.95, n-1, loc=mean_value, scale=std_value/np.sqrt(n))
                    # ci_lower = confidence_interval[0]
                    # ci_upper = confidence_interval[1]
                    std_error = std_value / np.sqrt(n)
                    ci_lower = mean_value - std_error * 1.96 #* std_error  # 95% CI with normal approximation
                    ci_upper = mean_value + std_error * 1.96 #* std_error
                else:
                    # If only one sample, we can't calculate proper CI
                    ci_lower = mean_value
                    ci_upper = mean_value
                
                # Create unique key: (total_nodes, mip_gap)
                key = (total_nodes, mip_gap)
                
                if key not in plot_data:
                    plot_data[key] = {
                        'x': [], 
                        'y_mean': [], 
                        'y_ci_lower': [], 
                        'y_ci_upper': [],
                        'n_samples': []
                    }
                
                plot_data[key]['x'].append(total_sfcs)
                plot_data[key]['y_mean'].append(mean_value)
                plot_data[key]['y_ci_lower'].append(ci_lower)
                plot_data[key]['y_ci_upper'].append(ci_upper)
                plot_data[key]['n_samples'].append(n)
                
                print(f"Nodes {total_nodes}, mip_gap {mip_gap}, SFCs {total_sfcs}: {n} amostras, média = {mean_value:.2f}, CI95% = [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Ordena por total_nodes e mip_gap
    plot_data = dict(sorted(plot_data.items(), key=lambda x: (x[0][0], x[0][1])))
    return plot_data

def create_plot(plot_data, output_file, y_label, y_max):
    """Cria o plot com os dados organizados por nodes e mip_gap, incluindo intervalos de confiança com zoom inset"""
    # Cria figura com um único subplot
    fig, ax_main = plt.subplots(figsize=(8, 5))
    
    # Cria o gráfico inset no canto inferior direito
    if "deployeda" in output_file:
        ax_inset = fig.add_axes([0.65, 0.22, 0.3, 0.3])  # [left, bottom, width, height]
        
    # Cores e marcadores diferentes para cada combinação
    tab20_colors = plt.cm.tab20c.colors
     
    # Mapeamento de nodes para pares de cores
    unique_nodes = sorted(set([key[0] for key in plot_data.keys()]))
    unique_nodes.append("dqn1")
    unique_nodes.append("dqn2")
    unique_nodes.append("dqn3")
    unique_nodes.append("dqn4")
    node_color_pairs = {}
    for i, node in enumerate(unique_nodes):
        if i * 4 < len(tab20_colors):
            node_color_pairs[node] = (
                tab20_colors[i*4], 
                tab20_colors[i*4 + 1], 
                tab20_colors[i*4 + 2], 
                tab20_colors[i*4 + 3]
            )
        else:
            # Fallback - ciclo através das cores
            base_idx = (i % (len(tab20_colors) // 4)) * 4
            node_color_pairs[node] = (
                tab20_colors[base_idx],
                tab20_colors[base_idx + 1],
                tab20_colors[base_idx + 2], 
                tab20_colors[base_idx + 3]
            )

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    fontsize = 20

    for i, ((total_nodes, mip_gap), data_dict) in enumerate(plot_data.items()):
        x_values = data_dict['x']
        y_mean = data_dict['y_mean']
        y_ci_lower = data_dict['y_ci_lower']
        y_ci_upper = data_dict['y_ci_upper']
        
        c1, c2, c3, c4 = node_color_pairs[total_nodes]
        
        # Label único para cada combinação
        label = f'{int(total_nodes)} G{mip_gap}'
        marker = markers[hash(mip_gap) % len(markers)]
        # Escolhe a tonalidade baseada no mip_gap
        if mip_gap == 0.0001:
            color = c1 
        elif mip_gap == 0.2:
            color = c2
        elif mip_gap == 0.1:
            color = c3
        elif mip_gap == 0.01:
            color = c4
        elif mip_gap == 0.00: #DQN
            c1aa, c2aa, c3aa, c4aa = node_color_pairs["dqn2"]
            d1aa, d2aa, c3aa, c4aa = node_color_pairs["dqn1"]
            if total_nodes == 125:
                color = c1aa
                marker = markers[3]
            else:
                color = d1aa 
                marker = markers[3]
            label = f'{int(total_nodes)} DQN'
        else:
            color = c1  # default

        # Ordena os pontos por x_values para garantir linha contínua
        sorted_indices = np.argsort(x_values)
        x_sorted = np.array(x_values)[sorted_indices]
        y_mean_sorted = np.array(y_mean)[sorted_indices]
        y_ci_lower_sorted = np.array(y_ci_lower)[sorted_indices]
        y_ci_upper_sorted = np.array(y_ci_upper)[sorted_indices]
        
        # Plota no gráfico principal
        line = ax_main.plot(x_sorted, y_mean_sorted, 
                marker=marker, 
                linestyle='-', 
                linewidth=2,
                markersize=fontsize/2,
                color=color,
                label=label)
        
        ax_main.fill_between(x_sorted, y_ci_lower_sorted, y_ci_upper_sorted,
                        color=color, alpha=0.2, label=f'_nolegend_')
        
        # Plota no gráfico inset (zoom)
        if "deployeda" in output_file:
            ax_inset.plot(x_sorted, y_mean_sorted, 
                    marker=marker, 
                    linestyle='-', 
                    linewidth=1.5,
                    markersize=fontsize/3,
                    color=color,
                    label='_nolegend_')  # Remove labels do inset
            
            ax_inset.fill_between(x_sorted, y_ci_lower_sorted, y_ci_upper_sorted,
                            color=color, alpha=0.2, label=f'_nolegend_')

    # Configurações do gráfico principal
    if output_file == "sfc_time_log.pdf":
        ax_main.set_yscale('log')
         

    ax_main.set_xlabel('Requested SFCs', fontsize=fontsize-2)
    ax_main.set_ylabel(y_label, fontsize=fontsize-2)
    ax_main.legend(fontsize=fontsize-3, loc='best')
    
    if y_max == None:
        ax_main.set_ylim(bottom=0)
    else:
        ax_main.set_ylim(bottom=0, top=y_max)
    ax_main.set_ylim(bottom=0)
    
    ax_main.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    if "deployeda" in output_file:
        # Configurações do gráfico inset (zoom)
        ax_inset.set_xlim(18, 26)
        ax_inset.set_ylim(18, 26)
        #ax_inset.set_title('Zoom: [19-26]', fontsize=fontsize-4)
        # ax_inset.grid(False, alpha=0.3)
        ax_inset.tick_params(axis='both', which='major', labelsize=fontsize-4)
    
        # Adiciona uma moldura ao inset
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.5)
    
    # Ajusta o layout
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as: {output_file}")


def create_combined_plot(all_plots_data, output_file="combined_plot.pdf"):
    """Cria um plot combinado com todos os subplots e uma legenda única"""
    # Cria figura com subplots - 3x2 para os 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(8, 4))
    axes = axes.flatten()
    
    # Configurações de estilo consistentes
    tab20_colors = plt.cm.tab20c.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    fontsize = 14  # Um pouco menor para subplots
    
    # Prepara o mapeamento de cores (usando o mesmo para todos os subplots)
    plot_data_sample = all_plots_data[0][0]  # Primeiro conjunto de dados para extrair nodes
    unique_nodes = sorted(set([key[0] for key in plot_data_sample.keys()]))
    unique_nodes.extend(["dqn1", "dqn2", "dqn3", "dqn4"])
    
    node_color_pairs = {}
    for i, node in enumerate(unique_nodes):
        if i * 4 < len(tab20_colors):
            node_color_pairs[node] = (
                tab20_colors[i*4], 
                tab20_colors[i*4 + 1], 
                tab20_colors[i*4 + 2], 
                tab20_colors[i*4 + 3]
            )
        else:
            base_idx = (i % (len(tab20_colors) // 4)) * 4
            node_color_pairs[node] = (
                tab20_colors[base_idx],
                tab20_colors[base_idx + 1],
                tab20_colors[base_idx + 2], 
                tab20_colors[base_idx + 3]
            )

    # Lista para coletar handles e labels da legenda
    all_handles = []
    all_labels = []
    legend_added = False  # Para evitar duplicatas na legenda

    # Plota cada conjunto de dados em um subplot
    for idx, (plot_data, y_label, y_max) in enumerate(all_plots_data):
        if idx >= len(axes):
            break
            
        ax = axes[idx]

        if idx == 0:
            # ax_inset = ax.add_axes([0.65, 0.22, 0.3, 0.3]) 
            ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower right')
        
        for i, ((total_nodes, mip_gap), data_dict) in enumerate(plot_data.items()):
            x_values = data_dict['x']
            y_mean = data_dict['y_mean']
            y_ci_lower = data_dict['y_ci_lower']
            y_ci_upper = data_dict['y_ci_upper']
            
            c1, c2, c3, c4 = node_color_pairs[total_nodes]
            
            # Label único para cada combinação
            label = f'{int(total_nodes)} G{mip_gap}'
            marker = markers[hash(mip_gap) % len(markers)]
            
            # Escolhe a tonalidade baseada no mip_gap
            if mip_gap == 0.0001:
                color = c1 
            elif mip_gap == 0.2:
                color = c2
            elif mip_gap == 0.1:
                color = c3
            elif mip_gap == 0.01:
                color = c4
            elif mip_gap == 0.00:  # DQN
                c1aa, c2aa, c3aa, c4aa = node_color_pairs["dqn2"]
                d1aa, d2aa, c3aa, c4aa = node_color_pairs["dqn1"]
                if total_nodes == 125:
                    color = c1aa
                    marker = markers[3]
                else:
                    color = d1aa 
                    marker = markers[3]
                label = f'{int(total_nodes)} DQN'
            else:
                color = c1  # default

            # Ordena os pontos
            sorted_indices = np.argsort(x_values)
            x_sorted = np.array(x_values)[sorted_indices]
            y_mean_sorted = np.array(y_mean)[sorted_indices]
            y_ci_lower_sorted = np.array(y_ci_lower)[sorted_indices]
            y_ci_upper_sorted = np.array(y_ci_upper)[sorted_indices]
            
            # Plota a linha com marcadores
            line = ax.plot(x_sorted, y_mean_sorted, 
                    marker=marker, 
                    linestyle='-', 
                    linewidth=2,
                    markersize=fontsize/2,
                    color=color,
                    label=label)
            
            # Plota a área do intervalo de confiança
            ax.fill_between(x_sorted, y_ci_lower_sorted, y_ci_upper_sorted,
                          color=color, alpha=0.2, label=f'_nolegend_')
            
            # Coleta handles e labels apenas do primeiro subplot para legenda única
            if idx == 0 and not legend_added:
                all_handles.extend(line)
                all_labels.append(label)
            
            # Plota no gráfico inset (zoom)
            if idx==0:
                ax_inset.plot(x_sorted, y_mean_sorted, 
                        marker=marker, 
                        linestyle='-', 
                        linewidth=1.5,
                        markersize=fontsize/3,
                        color=color,
                        label='_nolegend_')  # Remove labels do inset
                
                ax_inset.fill_between(x_sorted, y_ci_lower_sorted, y_ci_upper_sorted,
                                color=color, alpha=0.2, label=f'_nolegend_')
        
        # Configurações do subplot
        if "sfc_time_log" in output_file and idx == len(all_plots_data)-1:  # Último plot se for time_log
            ax.set_yscale('log')
        
        ids = ["a", "b", "c", "d"]
        
        ax.set_xlabel(f'({ids[idx]}) Requested SFCs', fontsize=fontsize-2)
        ax.set_ylabel(y_label, fontsize=fontsize-2)
        
        if idx ==1:
            ax.set_yscale('log')
        # if y_max is None:
        #     ax.set_ylim(bottom=0)
        # else:
        #     ax.set_ylim(bottom=0, top=y_max)
        
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.grid(True, alpha=0.3)
        
        # Marca a legenda como adicionada após o primeiro subplot
        if idx == 0:
            legend_added = True

        
        
        
        if idx==0: 
            # Configurações do gráfico inset (zoom)
            ax_inset.set_xlim(18, 26)
            ax_inset.set_ylim(18, 26)
            #ax_inset.set_title('Zoom: [19-26]', fontsize=fontsize-4)
            # ax_inset.grid(False, alpha=0.3)
            ax_inset.tick_params(axis='both', which='major', labelsize=fontsize-4)
        
            # Adiciona uma moldura ao inset
            for spine in ax_inset.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1.5)

    # Remove subplots vazios se houver menos que 6
    for idx in range(len(all_plots_data), len(axes)):
        fig.delaxes(axes[idx])

    # Legenda única para toda a figura
    if all_handles:
        fig.legend(all_handles, all_labels, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=6 , #len(all_handles),  # TODOS OS ITENS EM UMA LINHA
                  fontsize=fontsize-4,    # Um pouco menor para caber
                  frameon=True,
                  fancybox=False,
                  shadow=False,
                  columnspacing=0.8,    # ↓ Espaço entre colunas
          handletextpad=0.5, )

    

    plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.1, right=0.95, 
                       wspace=0.4, hspace=0.4)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as: {output_file}")
    plt.show()
    
    return fig, axes



def create_plot2(plot_data, output_file, y_label, y_max):
    """Cria o plot com dados organizados por nodes e mip_gap, plotando apenas a melhor interpolação"""
    # Cria figura com um único subplot
    fig, ax_main = plt.subplots(figsize=(12, 8))
    
    # Cria o gráfico inset no canto inferior direito se necessário
    if "deployeda" in output_file:
        ax_inset = fig.add_axes([0.65, 0.22, 0.3, 0.3])
    
    # Cores e marcadores
    tab20_colors = plt.cm.tab20c.colors
     
    # Mapeamento de nodes para pares de cores
    unique_nodes = sorted(set([key[0] for key in plot_data.keys()]))
    unique_nodes.extend(["dqn1", "dqn2", "dqn3", "dqn4"])
    node_color_pairs = {}
    for i, node in enumerate(unique_nodes):
        if i * 4 < len(tab20_colors):
            node_color_pairs[node] = (
                tab20_colors[i*4], 
                tab20_colors[i*4 + 1], 
                tab20_colors[i*4 + 2], 
                tab20_colors[i*4 + 3]
            )
        else:
            base_idx = (i % (len(tab20_colors) // 4)) * 4
            node_color_pairs[node] = (
                tab20_colors[base_idx],
                tab20_colors[base_idx + 1],
                tab20_colors[base_idx + 2], 
                tab20_colors[base_idx + 3]
            )

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    fontsize = 20
    
    for i, ((total_nodes, mip_gap), data_dict) in enumerate(plot_data.items()):
        x_values = np.array(data_dict['x'])
        y_mean = np.array(data_dict['y_mean'])
        y_ci_lower = np.array(data_dict['y_ci_lower'])
        y_ci_upper = np.array(data_dict['y_ci_upper'])
        
        c1, c2, c3, c4 = node_color_pairs[total_nodes]
        label_base = f'{total_nodes} G{mip_gap}'
        # Define cor e marcador baseado nos parâmetros
        if mip_gap == 0.0001:
            color = c1 
        elif mip_gap == 0.2:
            color = c2
        elif mip_gap == 0.1:
            color = c3
        elif mip_gap == 0.01:
            color = c4
        elif mip_gap == 0.00:  # DQN
            c1aa, c2aa, c3aa, c4aa = node_color_pairs["dqn2"]
            d1aa, d2aa, c3aa, c4aa = node_color_pairs["dqn1"]
            color = c1aa if total_nodes == 125 else d1aa
            marker = markers[3]
            label_base = f'{total_nodes} DQN'
        else:
            color = c1
            marker = markers[hash(mip_gap) % len(markers)]
            label_base = f'{total_nodes} G{mip_gap}'

        # Ordena os pontos
        sorted_indices = np.argsort(x_values)
        x_sorted = x_values[sorted_indices]
        y_mean_sorted = y_mean[sorted_indices]
        y_ci_lower_sorted = y_ci_lower[sorted_indices]
        y_ci_upper_sorted = y_ci_upper[sorted_indices]
 

        # PLOT: Pontos originais
        ax_main.scatter(x_sorted, y_mean_sorted, 
                       marker=marker, 
                       s=100,
                       color=color,
                       alpha=0.8,
                       label=f'{label_base}',
                       zorder=5)
        
        # Barras de erro para intervalos de confiança
        ax_main.errorbar(x_sorted, y_mean_sorted, 
                        yerr=[y_mean_sorted - y_ci_lower_sorted, y_ci_upper_sorted - y_mean_sorted],
                        fmt='none',
                        color=color, 
                        alpha=0.4,
                        capsize=3,
                        zorder=4)

        # TESTAR TODOS OS AJUSTES E SELECIONAR O MELHOR
        if len(x_sorted) > 1:
            x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            best_fit = None
            best_error = float('inf')
            best_type = None
            best_params = None
            best_equation = None
            
            # 1. REGRESSÃO LINEAR (y = ax + b) - LINHA DE TENDÊNCIA
            try:
                # Usar polyfit para regressão linear (grau 1)
                linear_coeffs = np.polyfit(x_sorted, y_mean_sorted, 1)
                linear_func = np.poly1d(linear_coeffs)
                y_linear = linear_func(x_dense)
                
                # Calcular erro (RMSE)
                y_pred_linear = linear_func(x_sorted)
                error_linear = np.sqrt(np.mean((y_mean_sorted - y_pred_linear)**2))
                
                if error_linear < best_error:
                    best_fit = (x_dense, y_linear)
                    best_error = error_linear
                    best_type = 'Linear'
                    best_params = f"RMSE: {error_linear:.4f}"
                    best_equation = f"y = {linear_coeffs[0]:.3f}x + {linear_coeffs[1]:.3f}"
                    print(f"{label_base}: Linear fit - a={linear_coeffs[0]:.3f}, b={linear_coeffs[1]:.3f}, RMSE={error_linear:.4f}")
            except Exception as e:
                print(f"Linear regression failed for {label_base}: {e}")

            # 2. REGRESSÃO QUADRÁTICA (y = ax² + bx + c)
            try:
                if len(x_sorted) >= 3:
                    quad_coeffs = np.polyfit(x_sorted, y_mean_sorted, 2)
                    quad_func = np.poly1d(quad_coeffs)
                    y_quadratic = quad_func(x_dense)
                    
                    y_pred_quadratic = quad_func(x_sorted)
                    error_quadratic = np.sqrt(np.mean((y_mean_sorted - y_pred_quadratic)**2))
                    
                    if error_quadratic < best_error:
                        best_fit = (x_dense, y_quadratic)
                        best_error = error_quadratic
                        best_type = 'Quadratic'
                        best_params = f"RMSE: {error_quadratic:.4f}"
                        best_equation = f"y = {quad_coeffs[0]:.3f}x² + {quad_coeffs[1]:.3f}x + {quad_coeffs[2]:.3f}"
                    print(f"{label_base}: Quadratic fit -   RMSE={error_linear:.4f}")
            except Exception as e:
                print(f"Quadratic regression failed for {label_base}: {e}")

            # 3. REGRESSÃO CÚBICA (y = ax³ + bx² + cx + d)
            try:
                if len(x_sorted) >= 4:
                    cubic_coeffs = np.polyfit(x_sorted, y_mean_sorted, 3)
                    cubic_func = np.poly1d(cubic_coeffs)
                    y_cubic = cubic_func(x_dense)
                    
                    y_pred_cubic = cubic_func(x_sorted)
                    error_cubic = np.sqrt(np.mean((y_mean_sorted - y_pred_cubic)**2))
                    
                    if error_cubic < best_error:
                        best_fit = (x_dense, y_cubic)
                        best_error = error_cubic
                        best_type = 'Cubic'
                        best_params = f"RMSE: {error_cubic:.4f}"
                        best_equation = f"y = {cubic_coeffs[0]:.3f}x³ + {cubic_coeffs[1]:.3f}x² + {cubic_coeffs[2]:.3f}x + {cubic_coeffs[3]:.3f}"
                    print(f"{label_base}: Cubic fit -   RMSE={error_cubic:.4f}")
            except Exception as e:
                print(f"Cubic regression failed for {label_base}: {e}")

            # 4. AJUSTE EXPONENCIAL (y = a * exp(b * x))
            try:
                if np.all(y_mean_sorted > 0):
                    # Usar polyfit no log(y) para regressão exponencial
                    log_y = np.log(y_mean_sorted)
                    exp_coeffs = np.polyfit(x_sorted, log_y, 1)
                    a = np.exp(exp_coeffs[1])
                    b = exp_coeffs[0]
                    
                    y_exponential = a * np.exp(b * x_dense)
                    y_pred_exponential = a * np.exp(b * x_sorted)
                    error_exponential = np.sqrt(np.mean((y_mean_sorted - y_pred_exponential)**2))
                    
                    if error_exponential < best_error:
                        best_fit = (x_dense, y_exponential)
                        best_error = error_exponential
                        best_type = 'Exponential'
                        best_params = f"RMSE: {error_exponential:.4f}"
                        best_equation = f"y = {a:.3f} * exp({b:.3f}x)"
                    print(f"{label_base}: Exponential fit -   RMSE={error_exponential:.4f}")
            except Exception as e:
                print(f"Exponential fit failed for {label_base}: {e}")

            # 5. AJUSTE LOGARÍTMICO (y = a + b * ln(x))
            try:
                if np.all(x_sorted > 0):
                    log_coeffs = np.polyfit(np.log(x_sorted), y_mean_sorted, 1)
                    a = log_coeffs[1]
                    b = log_coeffs[0]
                    
                    y_log = a + b * np.log(x_dense)
                    y_pred_log = a + b * np.log(x_sorted)
                    error_log = np.sqrt(np.mean((y_mean_sorted - y_pred_log)**2))
                    
                    if error_log < best_error:
                        best_fit = (x_dense, y_log)
                        best_error = error_log
                        best_type = 'Logarithmic'
                        best_params = f"RMSE: {error_log:.4f}"
                        best_equation = f"y = {a:.3f} + {b:.3f}·ln(x)"
                    print(f"{label_base}: Logarithmic fit -   RMSE={error_log:.4f}")
            except Exception as e:
                print(f"Logarithmic fit failed for {label_base}: {e}")

            # PLOTAR APENAS O MELHOR AJUSTE
            if best_fit is not None:
                x_dense_best, y_dense_best = best_fit
                
                # Estilo baseado no tipo de ajuste
                if best_type == 'Linear':
                    linestyle = '--'
                    linewidth = 2.5
                elif best_type == 'Quadratic':
                    linestyle = '-.'
                    linewidth = 2
                elif best_type == 'Cubic':
                    linestyle = ':'
                    linewidth = 2
                elif best_type == 'Exponential':
                    linestyle = '-'
                    linewidth = 2
                elif best_type == 'Logarithmic':
                    linestyle = '-'
                    linewidth = 2
                    linewidth = 1.5
                
                ax_main.plot(x_dense_best, y_dense_best, 
                           linestyle=linestyle,
                           color=color, 
                           alpha=0.8, 
                           linewidth=linewidth,
                           label=f'{label_base} {best_type}\n{best_equation}',
                           zorder=3)
                
                print(f"{label_base}: Best fit = {best_type}, RMSE = {best_error:.6f}")

        

    # Configurações do gráfico principal
    
    ax_main.set_yscale('log')

    ax_main.set_xlabel('Requested SFCs', fontsize=fontsize)
    ax_main.set_ylabel(y_label, fontsize=fontsize)
    
    # Legenda organizada
    ax_main.legend(fontsize=fontsize-6, loc='best', ncol=1, framealpha=0.7)
    
    # Configurações de limites
    if y_max is None:
        ax_main.set_ylim(bottom=0)
    else:
        ax_main.set_ylim(bottom=0, top=y_max)
    
    ax_main.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    
    # Configurações do gráfico inset (se aplicável)
    if "deployeda" in output_file:
        ax_inset.set_xlim(18, 26)
        ax_inset.set_ylim(18, 26)
        ax_inset.tick_params(axis='both', which='major', labelsize=fontsize-4)
        ax_inset.grid(True, alpha=0.2, linestyle=':')
        
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.5)

    # Ajusta o layout e salva
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as: {output_file}")
 

def main():
    args = parse_arguments()
    #objectives=['Maximize_SFCs', 'Maximize_SFCs_Minimize_Resources']
    # Processa os arquivos JSON
    ymax=80 
    #⬆ ⬇
    y = [
         ("sfc_deployed.pdf", "SFCs Deployed ➡  ", "accepted_sfcs", None),
         ("sfc_time_log.pdf", "Solving secs ⬅", "elapsed_time", None),
    #    # ("sfc_resources.pdf", "Virtual Instances (Lower is better)", "sum_vnf_vs_instances_count", ymax),
         ("sfc_vnf.pdf", "VNF Instances ⬅", "vnf_instances_count", ymax),
         ("sfc_vs.pdf", "VS Instances ⬅", "vs_instances_count", ymax),
     #   #  ("sfc_time.pdf", "Elapsed secs (Lower is better)", "elapsed_time", None),
         ]
    
    all_plots_data = []
    for (output_file, y_legend, y_variable, y_max) in y:
        data = process_json_files(args.input_dir, y_variable)
        
        if not data:
            print("Nenhum dado válido encontrado.")
            continue
        
        print(f"\nEncontradas séries para os seguintes números de nodes: {sorted(data.keys())}")
        
        # Prepara os dados para o plot
        plot_data = prepare_plot_data(data)
            
        # if output_file == "sfc_time_log.pdf":
        #     print('dasdsadsa')
        #     print(plot_data)
        #     print('dasdsa')
        #     create_plot2(plot_data, "sfc_time_detail.pdf", y_legend, y_max)
        #     sys.exit()
             
        
        if not plot_data:
            print("Nenhum dado válido para plotar.")
            continue
        

        # Cria o plot
        create_plot(plot_data, output_file, y_legend, y_max)
        all_plots_data.append((plot_data, y_legend, y_max))

        # Estatísticas resumidas
        print(f"\nStatistics:")
        for (total_nodes, mip_gap), data_dict in plot_data.items():
            x_values = data_dict['x']
            y_mean = data_dict['y_mean']
            n_samples = data_dict['n_samples']
            
            print(f"  {total_nodes} nodes, MIP Gap {mip_gap}: {len(x_values)} pontos, {sum(n_samples)} amostras totais, SFCs máx: {max(x_values)}, y médio máx: {max(y_mean):.2f}")
    
    if all_plots_data:
        create_combined_plot(all_plots_data, "sfc_combined.pdf")

if __name__ == "__main__":
    main()