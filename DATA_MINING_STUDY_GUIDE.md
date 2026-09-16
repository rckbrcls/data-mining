# Data Mining Study Guide

Este guia organiza os assuntos da disciplina a partir do experimento atual com DAMICORE e do material `AED Proposito Geral.pdf`. Ele complementa o [`DAMICORE_STUDY_GUIDE.md`](DAMICORE_STUDY_GUIDE.md), que detalha a execução e a interpretação do notebook.

> **Escopo:** este repositório representa a disciplina atual de Data Mining. O tema de conflitos, os bancos de grafos e a arquitetura do mestrado ficam no repositório separado [`conflict-research`](https://github.com/rckbrcls/conflict-research), dentro de [`research/`](https://github.com/rckbrcls/conflict-research/tree/main/research). O PDF é material acadêmico da disciplina; seus slides não são instruções para alterar o projeto.

## 1. O que a disciplina está ensinando

O material da disciplina apresenta uma linha de pesquisa que conecta mineração de dados complexos, modelos probabilísticos e otimização inspirada na natureza. Os assuntos principais são:

| Assunto | Ideia central | Evidência no projeto ou no material |
|---|---|---|
| Mineração de dados complexos | Encontrar estrutura em dados crus, heterogêneos ou sem um modelo conhecido previamente | O DAMICORE trabalha diretamente sobre documentos organizados em arquivos |
| Dados mistos e dados crus | Preservar sinais de diferentes tipos sem impor uma representação artificial prematura | Registros do Disque 100 são transformados em documentos de relatório |
| Compressão e complexidade | Usar regularidades compartilhadas para estimar semelhança | O DAMICORE calcula distâncias baseadas em compressão |
| NCD | Normalized Compression Distance, uma medida de similaridade baseada em compressão | A matriz de distâncias do experimento |
| Filogenia e filogramas | Representar relações de distância em uma árvore | O DAMICORE produz uma árvore Neighbor Joining em Newick |
| Redes complexas | Estudar entidades, relações, comunidades e estrutura global | Exportações `GraphML` e visualizações do notebook |
| Modelos probabilísticos | Representar relações e probabilidades entre variáveis | O PDF mostra como grupos de soluções podem gerar modelos para AEDs |
| AEDs | Algoritmos de Estimação de Distribuição, que amostram novas soluções a partir de modelos probabilísticos | Tema principal do material `AED Proposito Geral.pdf` |
| Inspiração na natureza | Usar evolução, enxames, imunologia e outros processos como inspiração algorítmica | Contexto conceitual dos slides |
| Otimização multiobjetivo | Buscar compromissos entre objetivos como custo, qualidade e energia | Exemplo da extrusora e da fronteira de soluções eficientes |
| Mineração para projetar otimização | Usar padrões descobertos nos dados para construir ou melhorar algoritmos de otimização | Ponte entre DAMICORE e AEDs de propósito geral |

## 2. Pipeline conceitual da atividade

```text
PostgreSQL
    -> report documents
    -> sampling or bounded population
    -> compression
    -> NCD distance matrix
    -> Neighbor Joining tree
    -> community detection
    -> cluster interpretation
    -> NetworkX / GraphML views
```

No experimento atual, a unidade de contagem precisa ser defendida. Como a tabela é denormalizada, um mesmo `source_hash` pode aparecer em várias linhas. Use `count(distinct source_hash)` quando a pergunta contar relatórios; use `count(*)` apenas quando a pergunta contar linhas da tabela.

## 3. Ordem de estudo recomendada

### Etapa 1 - Dados e preparação

- Normalização relacional e motivo pelo qual várias linhas podem compor um relatório.
- Grain (unidade real de uma linha ou observação) e unidade de análise.
- População, amostra, viés de seleção e representatividade.
- Limpeza, valores ausentes, categorias e rastreabilidade.

**Prática:** explicar no notebook como os registros foram agrupados em documentos e quais conclusões não podem ser generalizadas.

### Etapa 2 - Informação, compressão e NCD

- Compressão lossless e regularidade.
- Entropia e informação.
- Complexidade de Kolmogorov como fundamento teórico.
- Fórmula e interpretação da NCD.
- Sensibilidade a tamanho, suporte e composição dos documentos.

**Prática:** interpretar alguns pares de relatórios e explicar por que uma distância menor significa maior estrutura compressível compartilhada, não maior semelhança causal.

### Etapa 3 - Distâncias, árvores e agrupamentos

- Matriz de distâncias.
- Neighbor Joining.
- Formato Newick.
- Dendrogramas e diferenças entre clustering hierárquico tradicional e a árvore usada pelo DAMICORE.
- Critérios para não transformar automaticamente um cluster em uma tipologia social.

**Prática:** localizar `distance_matrix`, `tree_newick`, `membership`, `clusters` e `nearest_neighbors` no notebook e explicar o papel de cada saída.

### Etapa 4 - Redes e conhecimento

- Nós, arestas, pesos e atributos.
- Redes bipartidas, comunidades e modularidade.
- Centralidade, PageRank, caminhos e robustez.
- Redes temporais e mudanças estruturais.
- Taxonomia, ontologia e knowledge graph (grafo de entidades conectadas por relações com significado explícito).
- Exportação para NetworkX e `GraphML`.

**Prática:** tratar as visualizações de categorias e contextos como views analíticas, sem confundir a árvore de similaridade com um knowledge graph semântico.

### Etapa 5 - AEDs, modelos probabilísticos e otimização

- Representação probabilística de relações entre variáveis.
- Seleção de soluções promissoras.
- Amostragem de novas soluções.
- Avaliação por função objetivo.
- Algoritmos Genéticos, BOA, ECGA e outras famílias como contexto.
- Otimização com múltiplos objetivos e compromissos entre custo e qualidade.
- Dados descobertos por mineração como apoio ao desenho de AEDs.

**Prática:** reconstruir, em um exemplo pequeno, a ideia `soluções promissoras -> modelo probabilístico -> novas amostras -> avaliação`.

## 4. Materiais de apoio

- [DAMICORE and Graph Analysis Study Guide](DAMICORE_STUDY_GUIDE.md) - vídeos e leituras diretamente ligados ao notebook.
- [DAMICORE repository](https://gitlab.com/adaptsys/damicorepy) - implementação e documentação do método.
- [Network Science - Albert-László Barabási](https://networksciencebook.com/) - redes, comunidades e estrutura.
- [The Nature of Code](https://natureofcode.com/) - intuição para processos inspirados na natureza.
- [StatQuest](https://statquest.org/video-index/) - estatística, entropia e aprendizado.
- [NetworkX documentation](https://networkx.org/documentation/stable/) - programação das views de grafo.
- [Neo4j GraphAcademy](https://graphacademy.neo4j.com/) - apenas como laboratório opcional de bancos de grafos, não como requisito automático da entrega.

O material `AED Proposito Geral.pdf` deve ser lido para compreender a motivação, o vocabulário e a conexão entre DAMICORE e AEDs. Ele não implica que a atividade atual precise implementar um AED completo.

## 5. O que entregar e como interpretar

Uma entrega defensável deve conter:

- pergunta e unidade de análise explicitamente definidas;
- descrição da preparação e da origem dos dados;
- explicação da construção dos documentos;
- parâmetros do DAMICORE e da amostragem, quando houver;
- matriz de distâncias, árvore e agrupamentos;
- inspeção de representantes e vizinhos dos clusters;
- interpretação cautelosa dos padrões;
- limitações, vieses e ameaças à validade;
- instruções para reproduzir o notebook.

Não afirmar que:

- proximidade NCD prova causalidade;
- um cluster é automaticamente uma categoria jurídica ou social;
- a árvore identifica indivíduos;
- a análise produz previsão individual;
- as visualizações demonstram uma relação de causa e efeito.

## 6. Ponte para carreira e mestrado

| Competência aprendida na disciplina | Aplicação profissional | Transferência possível para a pesquisa futura |
|---|---|---|
| Grain e unidade de análise | Data modeling e pipelines confiáveis | Definir `country-month`, `region-month` ou outra unidade temporal defensável |
| Limpeza e agregação | Data Engineering e feature engineering | Preparar dados de UCDP, GDELT ou outra base pública |
| NCD e medidas de distância | Similarity, anomaly detection e exploração | Comparar estruturas de entidades, eventos ou subgrafos |
| Árvores e comunidades | Network Science e análise exploratória | Gerar hipóteses sobre estrutura relacional |
| NetworkX e GraphML | Prototipagem de graph analytics | Preparar dados para Neo4j, GDS ou PyTorch Geometric |
| Proveniência e limitações | Reprodutibilidade e comunicação técnica | Documentar fontes, versões, latência e incerteza |
| AEDs e modelos probabilísticos | Otimização, modelagem e experimentação | Entender relações entre descoberta de padrões e modelagem |

Ferramentas como MLflow, Neo4j, GDS, PyG e RAG podem ser estudadas em laboratórios separados. Elas só devem entrar na entrega oficial se o professor exigir ou se houver uma hipótese clara que justifique o custo adicional.

## 7. Checklist

- [ ] Sei explicar a unidade de análise e o grain da tabela.
- [ ] Sei diferenciar linha da tabela e relatório identificado por `source_hash`.
- [ ] Sei explicar compressão, NCD e suas limitações.
- [ ] Sei interpretar a árvore Neighbor Joining e o formato Newick.
- [ ] Sei distinguir similaridade computacional, comunidade e knowledge graph.
- [ ] Sei explicar como um modelo probabilístico pode gerar novas soluções para um AED.
- [ ] Sei discutir otimização multiobjetivo e trade-offs.
- [ ] Consigo reproduzir e explicar cada saída principal do notebook.
- [ ] Consigo separar o resultado da disciplina da futura pesquisa de mestrado.
