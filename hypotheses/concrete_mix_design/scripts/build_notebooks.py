from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_concrete_mix_design.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


CELLS = [
    markdown(
        """
        # AED guiado pelo DAMICORE: projeto de misturas de concreto

        **Problema.** Escolher quanto colocar de cada um dos 7 componentes de um m³ de
        concreto (cimento, escória, cinza volante, água, superplastificante, agregado
        graúdo e miúdo) para obter **a maior resistência aos 28 dias pelo menor custo**.

        **Hipótese.** Um Algoritmo de Estimação de Distribuição (AED) cujo modelo
        probabilístico é construído inteiro pelo DAMICORE (quais variáveis andam juntas,
        quais soluções formam cada região da frente e a tabela de cada grupo) aproxima a
        frente de Pareto verdadeira avaliando só 0,06% das misturas possíveis.

        É o exemplo da extrusora do slide (`docs/AED Proposito Geral.pdf`) trocado por
        concreto:

        | Extrusora (slide) | Concreto |
        | --- | --- |
        | Variáveis: P, N, D1, D3, L1… | 7 componentes, cada um com 8 níveis de kg/m³ |
        | Minimizar a energia E | Minimizar o custo (US$/m³) |
        | Maximizar a massa processada Q | Maximizar a resistência aos 28 dias (MPa) |
        | Constrói/simula e testa | Modelo treinado em 1.030 ensaios de laboratório |
        | Valores das soluções promissoras, um arquivo por variável | Igual |
        | DAMICORE identifica os grupos de variáveis | Igual, e também os nichos de soluções |
        | Tabela de probabilidades de cada grupo | Uma por grupo dentro de cada nicho |
        | Amostragem gera novas extrusoras | Amostragem gera novas misturas |
        """
    ),
    markdown(
        """
        ## 1. Dados e o "laboratório simulado"

        Base UCI *Concrete Compressive Strength* (Yeh, 1998): 1.030 ensaios de 428
        misturas diferentes, em várias idades. Um *gradient boosting* aprende a
        resistência a partir da mistura e da idade; ele faz o papel de "constrói e testa".
        Na validação cruzada, todas as idades de uma mesma mistura ficam no mesmo fold,
        para medir o erro em misturas que o modelo nunca viu.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import toytree
        from IPython.display import display

        PROJECT_ROOT = Path.cwd().resolve()
        while (
            PROJECT_ROOT != PROJECT_ROOT.parent
            and not (PROJECT_ROOT / "pyproject.toml").exists()
        ):
            PROJECT_ROOT = PROJECT_ROOT.parent
        if not (PROJECT_ROOT / "pyproject.toml").exists():
            raise RuntimeError("Open this notebook from the repository environment.")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from hypotheses.concrete_mix_design.scripts.config import (
            COMPONENTS,
            DEFAULT_CONFIG,
            PRICES_USD_PER_KG,
        )
        from hypotheses.concrete_mix_design.scripts.experiment import execute_experiment
        from hypotheses.concrete_mix_design.scripts.landscape import describe_levels, load_tests

        tests = load_tests()
        print(f"{len(tests)} ensaios, {len(tests[list(COMPONENTS)].drop_duplicates())} misturas")
        display(tests.describe().T.round(1))
        """
    ),
    markdown(
        """
        ## 2. O AED e o experimento

        O espaço de busca tem 8 níveis por componente, ou seja 8⁷ ≈ 2,1 milhões de
        misturas. Os níveis são quantis das misturas reais; escória, cinza volante e
        superplastificante têm o nível 0 (ausente).

        **Restrição.** Uma mistura só é aceita se estiver perto de alguma mistura real
        (distância padronizada ≤ 1,0). Longe dos dados o modelo estaria inventando.
        Misturas inviáveis gastam avaliação e ficam atrás de todas as viáveis na seleção.

        **Uma geração do AED** (20 gerações de 50 misturas novas, depois de 200 misturas
        iniciais; 1.200 avaliações por semente, 20 sementes):

        1. **Soluções promissoras:** as 100 melhores avaliadas até agora, pela frente de
           Pareto.
        2. **DAMICORE nas soluções → nichos.** Cada mistura vira um arquivo em que cada
           componente é uma letra repetida conforme o nível (`A.BBB.C…`); misturas
           parecidas compartilham trechos e o NCD as aproxima. O corte FastGreedy do
           DAMICORE é usado se der no máximo 6 nichos de pelo menos 10 misturas; senão a
           árvore é cortada em menos grupos até isso valer. Cada nicho é uma região da
           frente (misturas baratas e fracas, caras e fortes…).
        3. **DAMICORE nas variáveis → grupos.** Um arquivo por componente. Em vez de
           escrever o nível de cada mistura (um caractere por mistura), o arquivo lista
           **os números das misturas agrupados por nível**. Assim dois componentes que
           dividem as misturas do mesmo jeito escrevem os mesmos trechos — inclusive
           relações inversas, como "muito cimento, pouca água" —, que é o que o NCD
           consegue enxergar. Isso é feito com todas as promissoras e de novo dentro de
           cada nicho.
        4. **Tabelas:** em cada nicho, uma tabela de probabilidade conjunta por grupo.
        5. **Amostragem:** cada mistura nova escolhe um nicho ao acaso e sorteia cada
           grupo da tabela dele. Não há mistura uniforme: se o nicho só repete misturas
           já avaliadas, tenta os outros nichos e depois o modelo de todas as
           promissoras. Um sorteio uniforme é o último recurso e fica registrado.

        Como o espaço cabe na memória, todas as 2,1 milhões de misturas são pontuadas uma
        vez e guardadas. A busca só lê o que avalia; a tabela completa é o **gabarito**,
        usado apenas para medir o quão perto o AED chegou.
        """
    ),
    code(
        """
        results = execute_experiment(DEFAULT_CONFIG)
        audit = results.audit
        print(f"Artefatos: {results.output_dir}")
        print(f"Modelo de resistência: R² = {audit['surrogate']['cv_r2_mean']:.3f}, "
              f"erro médio = {audit['surrogate']['cv_mae_mpa']:.1f} MPa")
        print(f"Misturas possíveis: {audit['search_space_size']:,}; "
              f"viáveis: {audit['feasible_mixes']:,}")
        print(f"Cada execução avalia {audit['evaluations_per_run']} misturas "
              f"({audit['share_of_space_per_run']:.3%} do espaço)")
        display(describe_levels(results.landscape.levels))
        display(pd.Series(PRICES_USD_PER_KG, name="US$/kg (ilustrativo)").to_frame())
        """
    ),
    markdown(
        """
        ## 3. Uma geração por dentro

        Primeira geração da primeira semente: o começo de cada arquivo de componente, a
        árvore das variáveis, os nichos e as tabelas de probabilidade de um nicho.
        """
    ),
    code(
        """
        seed = DEFAULT_CONFIG.seeds[0]
        generation_dir = results.output_dir / "searches" / f"seed-{seed}" / "generation-01"
        archive_dir = generation_dir / "archive"
        names = {f"component-{index}.txt": name for index, name in enumerate(COMPONENTS)}
        for path in sorted((archive_dir / "component_series").iterdir()):
            series = path.read_bytes()
            print(f"{names[path.name]:<18} {series[:60].decode()}…  ({len(series)} bytes)")

        tree_path = archive_dir / "tree.nwk"
        if tree_path.exists():
            tree = toytree.tree(tree_path.read_text(encoding="utf-8"))
            canvas, _, _ = tree.draw(width=480, height=320)
            display(canvas)
            display(pd.read_csv(archive_dir / "ncd_matrix.csv", index_col=0).round(3))

        first = (results.niches["seed"] == seed) & (results.niches["generation"] == 1)
        display(results.niches.loc[first].round(1))
        blocks = results.blocks
        display(
            blocks.loc[
                (blocks["seed"] == seed) & (blocks["generation"] == 1),
                ["scope", "block_id", "components"],
            ]
        )
        display(
            pd.read_csv(generation_dir / "probability_tables_niche-1.csv")
            .sort_values(["block_id", "probability"], ascending=[True, False])
            .groupby("block_id")
            .head(5)
        )
        """
    ),
    markdown(
        """
        ## 4. Convergência

        **Hipervolume** é a área do plano custo × resistência coberta pelas melhores
        misturas encontradas. Dividido pelo hipervolume do gabarito, 1,0 significa ter
        encontrado a frente de Pareto verdadeira.
        """
    ),
    code(
        """
        curve = (
            results.convergence.groupby("evaluations")["hypervolume_ratio"]
            .agg(["mean", "std"])
            .reset_index()
        )
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(curve["evaluations"], curve["mean"], "o-", label="AED + DAMICORE")
        axis.fill_between(
            curve["evaluations"], curve["mean"] - curve["std"], curve["mean"] + curve["std"], alpha=0.2
        )
        axis.axvline(DEFAULT_CONFIG.population_size, color="gray", linestyle=":",
                     label="Fim da população inicial")
        axis.set(
            xlabel="Avaliações",
            ylabel="Hipervolume / hipervolume do gabarito",
            title="Convergência (média ± desvio, 20 sementes)",
        )
        axis.legend()
        plt.show()
        """
    ),
    markdown(
        """
        ## 5. Resultado final

        Para cada semente: hipervolume, e o **custo extra** da mistura mais barata que
        atinge 30, 40, 50 e 60 MPa em relação à mais barata do gabarito (US$/m³, menor é
        melhor). Também: de qual modelo vieram as misturas novas.
        """
    ),
    code(
        """
        display(results.summary.set_index("metric").round(3))
        display(results.origins.round(3))
        print(f"Sorteios uniformes (último recurso): {audit['uniform_fallback_offspring']}")

        gap_columns = [f"cost_gap_{target:g}mpa" for target in DEFAULT_CONFIG.strength_targets]
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        metrics = results.run_metrics
        axes[0].hist(metrics["hypervolume_ratio"], bins=10)
        axes[0].set(xlabel="Hipervolume / gabarito", ylabel="Sementes",
                    title="Hipervolume final por semente")
        axes[1].boxplot([metrics[c].dropna() for c in gap_columns],
                        tick_labels=[f"{t:g} MPa" for t in DEFAULT_CONFIG.strength_targets])
        axes[1].set(ylabel="Custo extra (US$/m³)", title="Distância ao mais barato do gabarito")
        figure.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 6. O que o DAMICORE encontrou

        **Grupos de variáveis.** Com que frequência dois componentes caíram no mesmo grupo
        (todas as promissoras, todas as gerações e sementes), contra a frequência
        esperada se grupos do mesmo tamanho fossem sorteados. O teste binomial é só uma
        triagem: gerações da mesma semente não são independentes.

        **Nichos.** Quanto os nichos separam a frente: 0 = cada nicho cobre toda a faixa
        de resistência, 1 = nichos sem sobreposição.
        """
    ),
    code(
        """
        print(f"Cimento e água no mesmo grupo: {audit['cement_water_same_block_share']:.1%} das gerações")
        print(f"Qualquer par, ao acaso:        {audit['same_block_share_by_chance']:.1%}")
        print(f"Execuções do DAMICORE nas variáveis: {audit['damicore_component_runs']}, "
              f"sem árvore: {audit['damicore_component_runs_without_tree']}")
        display(results.pair_tests.head(8).round(4))

        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(results.block_cooccurrence.to_numpy(), cmap="magma", vmin=0, vmax=1)
        axis.set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=45, ha="right")
        axis.set_yticks(range(len(COMPONENTS)), COMPONENTS)
        axis.set_title("Frequência de dois componentes no mesmo grupo DAMICORE")
        figure.colorbar(image, ax=axis, shrink=0.8)
        figure.tight_layout()
        plt.show()

        print(f"Separação dos nichos na resistência: {audit['niche_strength_separation']:.2f}")
        print("Número de nichos por geração:", audit["niche_count_share"])
        print("Como o número foi escolhido:", audit["niche_status_share"])
        """
    ),
    markdown(
        """
        ## 7. Frente de Pareto e misturas recomendadas

        Frente do AED (todas as sementes juntas) sobre a frente verdadeira. A tabela
        mostra a mistura mais barata que atinge cada resistência, sua razão água/cimento
        e, como conferência do modelo, a resistência medida em laboratório da mistura
        real mais próxima.
        """
    ),
    code(
        """
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.step(results.true_front["cost_usd_m3"], results.true_front["strength_mpa"],
                  where="post", color="black", linewidth=2, label="Gabarito")
        axis.plot(results.front["cost_usd_m3"], results.front["strength_mpa"], "o", alpha=0.7,
                  label="AED + DAMICORE")
        axis.set(xlabel="Custo (US$/m³)", ylabel="Resistência prevista aos 28 dias (MPa)",
                 title="Frentes de Pareto")
        axis.legend()
        plt.show()

        display(
            results.recommendations[
                [
                    "strength_target_mpa",
                    "source",
                    "strength_mpa",
                    "cost_usd_m3",
                    "cost_gap_usd_m3",
                    *COMPONENTS,
                    "water_cement_ratio",
                    "water_binder_ratio",
                    "nearest_real_strength_at_design_age",
                ]
            ].round(2)
        )
        """
    ),
    markdown(
        """
        ## 8. Limites

        - Não há outro método no experimento: o AED é medido contra o gabarito, não contra
          alternativas. Sem um controle, não dá para dizer quanto do resultado vem do
          DAMICORE e quanto viria de qualquer AED.
        - A resistência é a **prevista** pelo modelo (erro médio ~4 MPa), não medida.
          A coluna da mistura real mais próxima mostra que o modelo pode ser otimista.
        - Os preços são ilustrativos; mudar os preços muda a frente.
        - A restrição de distância aos dados reduz, mas não elimina, a extrapolação.
        - O número de nichos (até 6, de pelo menos 10 misturas) foi escolhido num piloto
          com 7 das 20 sementes.
        - O gabarito só existe porque o espaço tem 2,1 milhões de misturas; com mais
          componentes ou níveis, só a busca resolveria.
        """
    ),
]


def build_notebook() -> Path:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build_notebook())
