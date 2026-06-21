"""
Comparação Metodológica: DBSCAN vs. HDBSCAN
Painel principal para artigo científico (formato ACM, 5 páginas)

Avalia três cenários topológicos desafiadores que expõem as limitações do DBSCAN
e a flexibilidade hierárquica do HDBSCAN.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Estética acadêmica global
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
})

NOISE_COLOR = '#BDBDBD'
NOISE_SIZE  = 12
DATA_SIZE   = 22
CMAP_NAME   = 'tab10'


# ===========================================================================
# 1. GERAÇÃO DOS DATASETS
# ===========================================================================

def gerar_cenario_1(seed: int = 42) -> np.ndarray:
    """
    Cenário 1 — Densidades Extremamente Heterogêneas.
    Dois clusters lado a lado com variâncias distintas,
    mais ruído uniforme de fundo.
    """
    rng = np.random.default_rng(seed)
    centers    = [[-4.5, 0.0], [4.5, 0.0]]
    cluster_std = [0.30, 2.80]          # Cluster A ultra-compacto, B extremamente esparso
    X, _ = make_blobs(
        n_samples   = [200, 350],
        centers     = centers,
        cluster_std = cluster_std,
        random_state= seed,
    )
    noise = rng.uniform(low=-9, high=9, size=(120, 2))
    return np.vstack([X, noise])


def gerar_cenario_2(seed: int = 42) -> np.ndarray:
    """
    Cenário 2 — Geometria Não-Convexa (make_moons).
    Forma lunar com ruído de fundo severo e uniforme, dificultando
    fronteiras de densidade global.
    """
    rng = np.random.default_rng(seed)
    X, _ = make_moons(n_samples=500, noise=0.07, random_state=seed)
    # Escala do ruído cobre a caixa delimitadora das luas + margem
    noise = rng.uniform(low=[-0.6, -0.6], high=[1.6, 1.4], size=(130, 2))
    return np.vstack([X, noise])


def gerar_cenario_3(seed: int = 42) -> np.ndarray:
    """
    Cenário 3 — Clusters Conectados por Ponte de Ruído.
    Duas massas densas unidas por um 'fio' de pontos menos densos.
    O DBSCAN com eps alto funde tudo; com eps baixo parte a ponte.
    """
    rng = np.random.default_rng(seed)

    # Massa densa esquerda
    left  = rng.normal(loc=[-4, 0], scale=0.45, size=(200, 2))
    # Massa densa direita
    right = rng.normal(loc=[ 4, 0], scale=0.45, size=(200, 2))

    # Fio de pontos esparsos conectando as duas massas (x ∈ [-3, 3])
    bridge_x = rng.uniform(-3, 3, size=55)
    bridge_y = rng.normal(0, 0.15, size=55)
    bridge   = np.column_stack([bridge_x, bridge_y])

    # Ruído de fundo espalhado
    noise = rng.uniform(low=[-6, -3], high=[6, 3], size=(60, 2))

    return np.vstack([left, right, bridge, noise])


# ===========================================================================
# 2. AVALIAÇÃO DE MÉTRICAS
# ===========================================================================

def calcular_metricas(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcula Silhouette Score, Davies-Bouldin Index e Proporção de Ruído.
    Retorna NaN para métricas que exigem ≥ 2 clusters válidos.
    """
    mascara_validos = labels != -1
    n_clusters      = len(set(labels[mascara_validos]))
    n_ruido         = np.sum(labels == -1)
    proporcao_ruido = n_ruido / len(labels)

    if n_clusters >= 2 and mascara_validos.sum() >= n_clusters + 1:
        sil = silhouette_score(X[mascara_validos], labels[mascara_validos])
        dbi = davies_bouldin_score(X[mascara_validos], labels[mascara_validos])
    else:
        sil = float('nan')
        dbi = float('nan')

    return {
        'n_clusters'      : n_clusters,
        'silhouette'      : sil,
        'davies_bouldin'  : dbi,
        'proporcao_ruido' : proporcao_ruido,
    }


# ===========================================================================
# 3. EXECUÇÃO DOS ALGORITMOS
# ===========================================================================

CENARIOS = [
    {
        'nome'          : 'Cenário 1\nDensidades Heterogêneas',
        'gerador'       : gerar_cenario_1,
        'dbscan_params' : {'eps': 0.55, 'min_samples': 8},
        'hdbscan_params': {'min_cluster_size': 20, 'min_samples': 5},
    },
    {
        'nome'          : 'Cenário 2\nGeometria Não-Convexa',
        'gerador'       : gerar_cenario_2,
        'dbscan_params' : {'eps': 0.13, 'min_samples': 7},
        'hdbscan_params': {'min_cluster_size': 15, 'min_samples': 4},
    },
    {
        'nome'          : 'Cenário 3\nPonte de Ruído',
        'gerador'       : gerar_cenario_3,
        'dbscan_params' : {'eps': 0.55, 'min_samples': 8},
        'hdbscan_params': {'min_cluster_size': 20, 'min_samples': 5},
    },
]

resultados = []

for cfg in CENARIOS:
    X_raw = cfg['gerador']()
    X     = StandardScaler(copy=True).fit_transform(X_raw)

    # --- DBSCAN ---
    db     = DBSCAN(**cfg['dbscan_params']).fit(X)
    db_met = calcular_metricas(X, db.labels_)

    # --- HDBSCAN (copy=True obrigatório para evitar warning do sklearn) ---
    hdb     = HDBSCAN(**cfg['hdbscan_params'], copy=True).fit(X)
    hdb_met = calcular_metricas(X, hdb.labels_)

    resultados.append({
        'cenario'     : cfg['nome'].replace('\n', ' '),
        'X'           : X,
        'db_labels'   : db.labels_,
        'hdb_labels'  : hdb.labels_,
        'db_params'   : cfg['dbscan_params'],
        'hdb_params'  : cfg['hdbscan_params'],
        'db_metricas' : db_met,
        'hdb_metricas': hdb_met,
    })


# ===========================================================================
# 4. RELATÓRIO ACADÊMICO NO TERMINAL
# ===========================================================================

separador = '=' * 72

print(f'\n{separador}')
print('  RELATÓRIO COMPARATIVO: DBSCAN vs. HDBSCAN')
print(f'{separador}\n')

linhas = []

for r in resultados:
    cenario = r['cenario']

    for algo, met, params in [
        ('DBSCAN' , r['db_metricas'],  r['db_params']),
        ('HDBSCAN', r['hdb_metricas'], r['hdb_params']),
    ]:
        linhas.append({
            'Cenário'          : cenario,
            'Algoritmo'        : algo,
            'Parâmetros'       : str(params),
            'N° Clusters'      : met['n_clusters'],
            'Silhouette ↑'     : f"{met['silhouette']:.4f}" if not np.isnan(met['silhouette']) else 'N/A',
            'Davies-Bouldin ↓' : f"{met['davies_bouldin']:.4f}" if not np.isnan(met['davies_bouldin']) else 'N/A',
            'Ruído (%)'        : f"{met['proporcao_ruido']*100:.1f}%",
        })

df_rel = pd.DataFrame(linhas)
print(df_rel.to_string(index=False))
print(f'\n{separador}\n')


# ===========================================================================
# 5. VISUALIZAÇÃO — PAINEL 3 × 3
# ===========================================================================

def _estilizar_ax(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#888888')


def plotar_clusters(ax: plt.Axes, X: np.ndarray, labels: np.ndarray) -> None:
    """Renderiza um subplot de clustering — título e métricas definidos externamente."""
    cmap = matplotlib.colormaps[CMAP_NAME]

    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        ax.scatter(
            X[mask, 0], X[mask, 1],
            color=cmap(cid % 10), s=DATA_SIZE,
            linewidths=0.3, edgecolors='white', alpha=0.88, zorder=2,
        )

    mask_ruido = labels == -1
    if mask_ruido.any():
        ax.scatter(
            X[mask_ruido, 0], X[mask_ruido, 1],
            color=NOISE_COLOR, s=NOISE_SIZE, marker='x',
            linewidths=0.7, alpha=0.65, zorder=1,
        )

    _estilizar_ax(ax)


# ---------------------------------------------------------------------------
# Layout: 3 linhas × 3 colunas
#
#   • Cabeçalhos de coluna → set_title() na linha 0, pad generoso
#   • Rótulos de linha     → fig.text() horizontal, centralizado na altura
#                            de cada linha — evita rotação e cramping
#   • Métricas             → set_xlabel() em 2 linhas abaixo de cada subplot
#     (nunca acima → sem sangramento entre linhas)
# ---------------------------------------------------------------------------
ROW_LABELS = [
    'C1: Densidades Heterogêneas',
    'C2: Geometria Não-Convexa',
    'C3: Ponte de Ruído',
]
COL_TITLES = ['Dados Originais', 'DBSCAN', 'HDBSCAN']

fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(7.5, 8.2),
    gridspec_kw={'hspace': 0.40, 'wspace': 0.10},
)

fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.09)

def _fmt_sil(m: dict) -> str:
    return f"{m['silhouette']:.3f}" if not np.isnan(m['silhouette']) else 'N/A'

for linha, r in enumerate(resultados):
    X           = r['X']
    db_p, hdb_p = r['db_params'], r['hdb_params']
    db_m, hdb_m = r['db_metricas'], r['hdb_metricas']

    # --- Coluna 0: Dados Originais ---
    # xlabel vazio garante que todos os subplots da linha tenham a mesma
    # altura de caixa — sem isso, as colunas 1 e 2 ficam mais baixas que a 0.
    ax0 = axes[linha, 0]
    ax0.scatter(X[:, 0], X[:, 1], c='#2c3e50', s=DATA_SIZE * 0.8,
                alpha=0.45, linewidths=0, zorder=2)
    # Placeholder com fontes idênticas aos xlabels reais → mesma métrica de
    # altura, grade perfeitamente alinhada; cor 'none' torna invisível.
    ax0.set_xlabel('placeholder\nplaceholder',
                   fontsize=7, linespacing=1.6, color='none')
    _estilizar_ax(ax0)

    # --- Coluna 1: DBSCAN ---
    plotar_clusters(axes[linha, 1], X, r['db_labels'])
    axes[linha, 1].set_xlabel(
        f"eps={db_p['eps']},  min_samples={db_p['min_samples']}\n"
        f"k={db_m['n_clusters']}   Sil={_fmt_sil(db_m)}   "
        f"Ruído={db_m['proporcao_ruido']*100:.0f}%",
        fontsize=7, linespacing=1.6, color='#333333',
    )

    # --- Coluna 2: HDBSCAN ---
    plotar_clusters(axes[linha, 2], X, r['hdb_labels'])
    axes[linha, 2].set_xlabel(
        f"min_cluster_size={hdb_p['min_cluster_size']},  "
        f"min_samples={hdb_p['min_samples']}\n"
        f"k={hdb_m['n_clusters']}   Sil={_fmt_sil(hdb_m)}   "
        f"Ruído={hdb_m['proporcao_ruido']*100:.0f}%",
        fontsize=7, linespacing=1.6, color='#333333',
    )

# Cabeçalhos de coluna — aplicados por último para não serem sobrescritos
for col, titulo in enumerate(COL_TITLES):
    axes[0, col].set_title(titulo, pad=7, fontsize=10,
                            fontweight='bold', color='#1a1a2e')

# Rótulos de linha via fig.text com rotation=90.
# A posição y usa a bbox do ax0 *depois* de subplots_adjust, garantindo
# alinhamento preciso ao centro vertical de cada linha de subplots.
for linha, label in enumerate(ROW_LABELS):
    bbox = axes[linha, 0].get_position()
    y_center = (bbox.y0 + bbox.y1) / 2
    fig.text(
        0.01, y_center, label,
        ha='left', va='center',
        fontsize=8, fontweight='bold', color='#1a1a2e',
        rotation=90,
    )

# suptitle posicionado acima do top=0.91 com y fixo
fig.suptitle(
    'Análise Comparativa: DBSCAN vs. HDBSCAN em Cenários Topológicos Desafiadores',
    fontsize=10, fontweight='bold', color='#1a1a2e',
    y=0.975,
)

# Legenda de ruído centralizada abaixo do grid, dentro da margem bottom=0.09
patch_ruido = mpatches.Patch(
    facecolor=NOISE_COLOR, edgecolor='gray', linewidth=0.5,
    label='Ruído  (label = −1)',
)
fig.legend(
    handles=[patch_ruido],
    loc='lower center',
    ncol=1,
    fontsize=8,
    framealpha=0.9,
    edgecolor='#cccccc',
    bbox_to_anchor=(0.55, 0.01),
)

# ---------------------------------------------------------------------------
# Exportação
# ---------------------------------------------------------------------------
OUTPUT = 'painel_comparativo.pdf'
plt.savefig(OUTPUT, format='pdf', dpi=300, bbox_inches='tight')
print(f'Figura exportada → {OUTPUT}')
plt.show()
