from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_emergency_feature_selection.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


CELLS = [
    markdown(
        """
        # AED guiado pelo DAMICORE: quais campos do Disque 100 perguntar?

        **Problema.** O formulário de denúncia tem 15 campos. Queremos saber quais
        perguntar para identificar emergências fazendo **o menor número de perguntas**.

        **Hipótese.** Um Algoritmo de Estimação de Distribuição (AED) cujo modelo
        probabilístico é construído pelo DAMICORE encontra os formulários com o melhor
        compromisso entre número de campos e informação sobre emergência.

        O método segue o slide "DAMICORE gerando modelos para AEDs"
        (`docs/AED Proposito Geral.pdf`), trocando a extrusora pelo formulário:

        | Extrusora (slide) | Formulário do Disque 100 |
        | --- | --- |
        | Variáveis: P, N, D1, D3, L1… | 15 campos: perguntar (1) ou não (0) |
        | Minimizar a energia E | Minimizar o número de campos |
        | Maximizar a massa processada Q | Maximizar a informação sobre emergência |
        | Valores das soluções promissoras, um arquivo por variável | Igual |
        | DAMICORE identifica os grupos de variáveis | Igual |
        | Tabela de probabilidades de cada grupo | Igual |
        | Amostragem gera novas extrusoras | Amostragem gera novos formulários |
        | Mede E e Q, seleciona as melhores e repete | Igual |

        **Alvo:** o status de emergência é a marcação histórica do banco, não uma
        confirmação independente de urgência.
        """
    ),
    markdown(
        """
        ## 1. Dados

        Cada linha é uma denúncia. Quatro status contam como emergência e `NÃO` como não
        emergência; status ausentes ou conflitantes saem. O perfil `full` usa 2024-01 a
        2026-06: três trimestres de validação (2025-Q3, 2025-Q4, 2026-Q1) para medir os
        formulários e 2026-Q2 reservado para a conferência final.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import sys

        import matplotlib.pyplot as plt
        import pandas as pd
        import toytree
        from IPython.display import display

        os.environ.setdefault("EMERGENCY_DATA_PROFILE", "full")

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

        from hypotheses.emergency_feature_selection.scripts.config import (
            DEFAULT_CONFIG,
            FEATURES,
            active_profile,
        )
        from hypotheses.emergency_feature_selection.scripts.data import prepare_data
        from hypotheses.emergency_feature_selection.scripts.experiment import (
            execute_experiment,
        )
        from hypotheses.emergency_feature_selection.scripts.fitness import (
            MaskEvaluator,
            code_folds,
            selected_features,
        )

        profile = active_profile()
        prepared = prepare_data(profile=profile)
        audit_summary = {
            key: value
            for key, value in prepared.audit.items()
            if key not in {"validation_folds", "feature_conflicts"}
        }
        print(f"Profile: {profile.name}; reserved period: {profile.test_name}")
        display(pd.Series(audit_summary, name="value").to_frame())
        """
    ),
    markdown(
        """
        ## 2. Como medir um formulário (E e Q)

        Um formulário é uma sequência de 15 bits: `1` = o campo é perguntado.

        - **E (custo):** o número de campos perguntados.
        - **Q (qualidade):** os campos escolhidos dividem as denúncias em grupos de
          respostas iguais. No período de treino, cada grupo recebe sua taxa de
          emergência; no período de validação seguinte, medimos quanto essas taxas
          acertam mais do que a taxa geral, em *bits por denúncia*. Combinações de
          respostas raras demais não se repetem na validação e perdem pontos, por isso
          perguntar tudo não é automaticamente o melhor.

        Exemplo: a qualidade de cada campo perguntado sozinho.
        """
    ),
    code(
        """
        evaluator = MaskEvaluator(code_folds(prepared.folds), DEFAULT_CONFIG)
        single_fields = pd.DataFrame(
            {
                "field": feature,
                "gain_bits": evaluator.quality(
                    tuple(int(position == index) for position in range(len(FEATURES)))
                ),
            }
            for index, feature in enumerate(FEATURES)
        ).sort_values("gain_bits", ascending=False)
        display(single_fields.round(5))
        """
    ),
    markdown(
        """
        ## 3. Rodar o AED guiado pelo DAMICORE

        Cada execução começa com 400 formulários variados e faz 10 gerações de 80
        formulários novos (1.200 avaliações). Em cada geração:

        1. **Seleção:** os 30% melhores formulários avaliados até agora, pela frente de
           Pareto (mais Q com menos E).
        2. **Variáveis como amostras:** cada campo vira um arquivo com a sequência de
           0/1 dele nesses formulários, todos na mesma ordem.
        3. **DAMICORE:** matriz NCD → árvore Neighbor-Joining → grupos FastGreedy.
        4. **Tabela de probabilidades** de cada grupo de campos.
        5. **Amostragem** de novos formulários a partir das tabelas.
        6. **Avaliação** de E e Q dos novos formulários, e volta ao passo 1.

        O AED roda com 10 sementes para mostrar que o resultado não depende do sorteio.
        """
    ),
    code(
        """
        results = execute_experiment(prepared, DEFAULT_CONFIG)
        print(f"Run artifacts: {results.output_dir}")
        """
    ),
    markdown(
        """
        ## 4. Uma geração por dentro

        Primeira geração da primeira execução. Abaixo: o começo de cada arquivo (um
        caractere por formulário promissor), a árvore que o DAMICORE montou com esses
        arquivos, os grupos que ele encontrou e as tabelas de probabilidade de cada grupo.
        """
    ),
    code(
        """
        seed = DEFAULT_CONFIG.seeds[0]
        generation_dir = results.output_dir / "searches" / f"seed-{seed}" / "generation-01"
        names = {f"feature-{index:02d}.txt": name for index, name in enumerate(FEATURES)}

        for file_name, feature in names.items():
            series = (generation_dir / "feature_series" / file_name).read_text()
            print(f"{feature:<28} {series[:60]}…  ({len(series)} formulários)")

        tree = toytree.tree((generation_dir / "tree.nwk").read_text(encoding="utf-8"))
        labels = [names.get(str(tip).strip("'\\""), str(tip)) for tip in tree.get_tip_labels()]
        canvas, _, _ = tree.draw(tip_labels=labels, width=520, height=420)
        display(canvas)

        blocks = results.damicore_blocks
        display(
            blocks.loc[
                (blocks["seed"] == seed) & (blocks["generation"] == 1),
                ["block_id", "feature_count", "features"],
            ]
        )
        display(pd.read_csv(generation_dir / "probability_tables.csv"))
        """
    ),
    markdown(
        """
        ## 5. Evolução ao longo das gerações

        Se o modelo do DAMICORE aprende com os formulários promissores, os formulários
        sorteados a cada geração devem ficar melhores. O segundo gráfico mostra com que
        frequência o DAMICORE colocou dois campos no mesmo grupo, somando todas as
        gerações e sementes.
        """
    ),
    code(
        """
        progress = (
            results.generation_progress.groupby("generation")[
                ["new_masks_mean_gain", "best_gain_so_far"]
            ]
            .mean()
            .reset_index()
        )
        display(results.damicore_diagnostics.groupby("generation")[
            ["bytes_per_series", "block_count", "largest_block", "ncd_median"]
        ].mean().round(3))

        figure, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].plot(progress["generation"], progress["new_masks_mean_gain"], "o-",
                     label="Média dos formulários novos")
        axes[0].plot(progress["generation"], progress["best_gain_so_far"], "s--",
                     label="Melhor formulário até agora")
        axes[0].set(
            xlabel="Geração (0 = população inicial)",
            ylabel="Q: ganho de informação (bits/denúncia)",
            title="Qualidade por geração (média das 10 sementes)",
        )
        axes[0].legend()

        image = axes[1].imshow(results.block_cooccurrence.to_numpy(), cmap="magma", vmin=0, vmax=1)
        axes[1].set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right")
        axes[1].set_yticks(range(len(FEATURES)), FEATURES)
        axes[1].set_title("Frequência de dois campos no mesmo grupo DAMICORE")
        figure.colorbar(image, ax=axes[1], shrink=0.8)
        figure.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 6. Formulário recomendado

        A frente de Pareto reúne, para cada número de campos, o melhor formulário que o
        AED encontrou. A recomendação é o formulário com **menos campos** que alcança 95%
        da melhor qualidade da frente. Por fim, medimos a mesma qualidade no período
        reservado, que o AED nunca viu, ao lado do formulário com os 15 campos.
        """
    ),
    code(
        """
        display(results.aed_front[["feature_count", "mean_gain_bits", "fields"]])
        print("Recommended fields:", selected_features(results.recommended_mask))
        display(results.held_out.round(5))

        front = results.aed_front
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(front["feature_count"], front["mean_gain_bits"], "o-", color="crimson")
        axis.axvline(sum(results.recommended_mask), color="gray", linestyle="--",
                     label="Recomendado")
        axis.set(
            xlabel="E: número de campos",
            ylabel="Q: ganho de informação (bits/denúncia)",
            title="Frente de Pareto encontrada pelo AED",
        )
        axis.legend()
        plt.show()
        """
    ),
    markdown(
        """
        ## 7. Conferência com o gabarito

        Com 15 campos existem 32.767 formulários possíveis, poucos o bastante para medir
        todos e saber a resposta certa. Isso não faz parte do AED: serve só para conferir
        se ele chegou ao melhor formulário avaliando uma pequena fração deles.
        """
    ),
    code(
        """
        audit = results.audit
        print(f"Avaliações por execução: {audit['evaluations_per_run']} "
              f"({audit['share_of_all_forms_per_run']:.1%} dos formulários possíveis)")
        print("Gabarito:", audit["ground_truth_recommended_fields"])
        print("AED:     ", audit["recommended_fields"])
        print("Mesmo formulário:", audit["recommended_matches_ground_truth"])
        display(results.seed_recommendations)
        """
    ),
    markdown(
        """
        ## 8. Limites

        - O alvo reproduz a marcação histórica de emergência, não urgência verificada.
        - O sinal é fraco: mesmo o melhor formulário explica pouco da marcação.
        - O NCD compara sequências de 0/1: reconhece campos que costumam entrar juntos
          nos formulários bons, mas não campos que se substituem (um entra quando o outro
          sai).
        - O gabarito só é possível porque há 15 campos; com muitos mais, só o AED
          resolveria.
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
