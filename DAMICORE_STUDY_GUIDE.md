# DAMICORE and Graph Analysis Study Guide

This guide follows the project pipeline:

`PostgreSQL -> report documents -> sampling -> compression -> NCD -> distance matrix -> Neighbor Joining tree -> community detection -> cluster membership -> interpretation -> knowledge graphs`

## Part 1 — Data preparation and sampling

- [ ] [Learn Database Normalization — 1NF, 2NF, 3NF, 4NF, 5NF](https://www.youtube.com/watch?v=GFQaEYEc8_8)
  - **Focus:** Understand why one report can occupy multiple related database rows and why the notebook groups rows by `source_hash` to create one document.

- [ ] [Sampling Methods and Bias with Surveys — Crash Course Statistics #10](https://www.youtube.com/watch?v=Rf-fIpB4D50)
  - **Focus:** Understand populations, samples, representativeness, and sampling bias.
  - **Project connection:** Clarifies why the notebook can analyze a bounded sample instead of processing every report and what conclusions can be generalized.

## Part 2 — Information, compression, and NCD

- [ ] [Compression — Computerphile](https://www.youtube.com/watch?v=Lto-ajuqW3w)
  - **Focus:** Learn how lossless compression identifies repeated structure in data.
  - **Project connection:** DAMICORE uses compressed document sizes to estimate similarity.

- [ ] [Entropy for Data Science — StatQuest](https://www.youtube.com/watch?v=YtebGVx-Fxw)
  - **Focus:** Understand entropy as uncertainty or information content.
  - **Project connection:** Repetitive and predictable reports generally contain structures that a compressor can encode more efficiently.

- [ ] [Kolmogorov Complexity — A Practical Introduction with Examples](https://www.youtube.com/watch?v=4UlgFjFXCFc)
  - **Focus:** Understand complexity as the size of the shortest description capable of reproducing an object.
  - **Project connection:** This is the theoretical foundation behind using compression as an approximation of document complexity.

- [ ] [Identifying Cover Songs Using Normalized Compression Distance](https://www.merlot.org/merlot/viewMaterial.htm?id=945491)
  - **Focus:** See a practical application of Normalized Compression Distance (NCD).
  - **Project connection:** The data domain is different, but the method of finding shared information through compression is the same.

- [ ] [Interactive Normalized Compression Distance Demonstration](https://namvdo.github.io/complearn/)
  - **Type:** Interactive companion, not a video.
  - **Focus:** Explore the NCD formula, a small distance matrix, and structural views of the result.

## Part 3 — Distances, clustering, and trees

- [ ] [Hierarchical Clustering — StatQuest](https://www.youtube.com/watch?v=7xHsRkOdVwo)
  - **Focus:** Build intuition for distance matrices, heatmaps, dendrograms, and groups of similar observations.
  - **Important:** DAMICORE does not use this exact clustering algorithm. The video is useful for visual and conceptual intuition.

- [ ] [Neighbour Joining Method: Phylogenetic Tree — Step-by-Step Guide](https://www.youtube.com/watch?v=mgU5aioDGqk)
  - **Focus:** Understand how a distance matrix is converted into a tree.
  - **Project connection:** DAMICORE uses Neighbor Joining to construct the report tree.

- [ ] [Newick Tree Format](https://www.youtube.com/watch?v=bXyUomrqD7s)
  - **Focus:** Learn how parentheses, commas, labels, and branch lengths represent a tree.
  - **Project connection:** Explains the value returned as `result.tree_newick`.

- [ ] [Visualizing Newick Trees with iTOL](https://www.youtube.com/watch?v=YEKuKr9Qc4g)
  - **Focus:** Turn a Newick string into a readable tree and annotate groups, branches, and labels.

## Part 4 — Networks, communities, and cluster interpretation

- [ ] [Networks: Insights into Complex Systems and Social Structures — Mark Newman](https://www.classcentral.com/course/youtube-mark-newman-2-what-networks-can-tell-us-about-the-world-228774)
  - **Focus:** Understand networks, communities, and modularity.
  - **Project connection:** Provides the conceptual foundation for detecting communities in the DAMICORE tree.

- [ ] [Interpreting the Results of Community Detection Algorithms — Neo4j](https://neo4j.com/videos/70-interpreting-the-results-of-community-detection-algorithms/)
  - **Focus:** Evaluate whether detected communities are cohesive and meaningful.
  - **Project connection:** Helps distinguish a computational cluster from a defensible latent report typology.

## Part 5 — Graph programming and knowledge representation

- [ ] [Introduction to NetworkX in Python for Graph Analysis](https://www.youtube.com/watch?v=x48uVgRnXg4)
  - **Focus:** Learn how Python represents nodes, edges, weights, graph attributes, and GraphML files.

- [ ] [What Is a Knowledge Graph? — IBM Technology](https://www.youtube.com/watch?v=y7sXDpffzQQ)
  - **Focus:** Understand how entities and relationships form a contextual information graph.
  - **Project connection:** Relates to connecting an anonymized report with its contextual entities.

- [ ] [Taxonomy, Ontology, Knowledge Graph, and Semantics](https://www.youtube.com/watch?v=sr257blfdY8)
  - **Focus:** Distinguish hierarchical classifications, formal concepts, semantic relationships, and knowledge graphs.
  - **Project connection:** Helps separate the violation hierarchy from the report similarity clusters and the knowledge graph.

- [ ] [Graph Data Modeling — Neo4j](https://www.youtube.com/watch?v=GB4TL8fMcg4)
  - **Focus:** Decide what should be modeled as an entity, attribute, or relationship.

- [ ] [Two-Mode Networks and Bipartite Graphs](https://www.youtube.com/watch?v=iigs3p0-GX4)
  - **Focus:** Understand networks connecting two distinct node types.
  - **Project connection:** Applies to views such as `abuse category <-> age group` and `abuse category <-> relationship`.

- [ ] [Temporal Networks, Where PageRank Meets Lord of the Rings — Computerphile](https://www.youtube.com/watch?v=ZP6Bh3iWklQ)
  - **Focus:** Understand how network structure changes over time.
  - **Project connection:** Explains why splitting the graph by month can reveal patterns hidden in an aggregated network.

- [ ] [Automatic Graph Layout — yWorks](https://www.youtube.com/watch?v=AkR6r1FbRdY)
  - **Focus:** Understand graph layout, label placement, overlap reduction, and visual readability.

## Part 6 — Reading the notebook outputs

After completing the core sections, verify that you can explain each output without looking at the notebook notes:

- [ ] **`distance_matrix`** — Pairwise NCD values. Lower values indicate more shared compressible structure.
- [ ] **`tree_newick`** — The Neighbor Joining tree serialized in Newick format.
- [ ] **`membership`** — A table assigning each source document to a detected cluster.
- [ ] **`clusters`** — A cluster-centered summary of the documents assigned to each group.
- [ ] **`nearest_neighbors`** — The reports with the smallest NCD values relative to a selected report.
- [ ] **Heatmap** — A visual representation of the distance matrix that can expose low-distance blocks.
- [ ] **Latent typology** — A human interpretation supported by recurring patterns inside a cluster; it is not a label automatically produced by DAMICORE.

## Recommended study order

### Core path

Watch Parts 1 through 4 in order. This path explains the complete DAMICORE workflow, from database records to interpreted clusters.

### Graph extension

Study Part 5 after the DAMICORE core. These videos explain how the results can be represented, enriched, compared over time, and visualized as graphs.

### Practical review

After every two or three videos, reopen the notebook and identify where each concept appears. Do not interpret a cluster as a social typology until representative reports, nearest neighbors, and results from additional samples have been examined.
