# Análise Comparativa: DBSCAN vs HDBSCAN

Repositório contendo o código-fonte e o laboratório de testes gerado para o artigo da disciplina de Metodologia Científica. O objetivo deste experimento é demonstrar as limitações do parâmetro de distância global do DBSCAN em topologias de densidade variável e contrastá-las com a robustez hierárquica do HDBSCAN.

**Autor:** Gustavo Kim Alcantara (RA: 820763)  
**Instituição:** Universidade Federal de São Carlos (UFSCar) - Ciência da Computação  

## Estrutura do Repositório

- `experimento_dbscan_hdbscan.py`: Script principal contendo a geração dos datasets sintéticos (Cenários de Densidades Variáveis, Formas Geométricas Não-Convexas e Clusters com Ruído de Conexão), execução dos algoritmos e cálculo das métricas.
- `painel_comparativo.pdf`: Imagem vetorial gerada pelo script, utilizada para discussão visual no artigo.
- `requirements.txt`: Lista de dependências e versões exatas para garantia de reprodutibilidade.

## Como Reproduzir o Experimento

Para garantir que o ambiente de execução seja idêntico, recomenda-se a utilização de um ambiente virtual Python.

**1. Clone este repositório:**
```bash
git clone [https://github.com/SEU-USUARIO/dbscan-hdbscan-comparativo-mc.git](https://github.com/SEU-USUARIO/dbscan-hdbscan-comparativo-mc.git)
cd dbscan-hdbscan-comparativo-mc
```

**2. Crie e ative um ambiente virtual:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Instale as dependências rigorosas:**
```bash
pip install -r requirements.txt
```

**4. Execute o laboratório de testes:**
```bash
python3 experimento_dbscan_hdbscan.py
```

O script irá imprimir no terminal o relatório com as métricas calculadas (Silhouette Score, Davies-Bouldin e Proporção de Ruído) e gerar o painel vetorial em PDF no mesmo diretório.