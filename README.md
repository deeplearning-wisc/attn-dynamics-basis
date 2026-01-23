# How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability

[**Shawn Im**](https://shawn-im.github.io/)<sup>1</sup>, [**Changdae Oh**](https://changdaeoh.github.io/)<sup>1</sup>, [**Zhen Fang**](https://fang-zhen.github.io/)<sup>2</sup>, [**Yixuan Li**](https://pages.cs.wisc.edu/~sharonli/)<sup>1</sup>

<sup>1</sup>University of Wisconsin–Madison, <sup>2</sup>University of Technology Sydney

> Semantic associations such as the link between ``bird'' and ``flew'' are foundational for language modeling as they enable models to go beyond memorization and instead generalize and generate coherent text. Understanding how these associations are learned and represented in language models is essential for connecting deep learning with linguistic theory and developing a mechanistic foundation for large language models. In this work, we analyze how these associations emerge from natural language data in attention-based language models through the lens of training dynamics. By leveraging a leading-term approximation of the gradients, we develop closed-form expressions for the weights at early stages of training that explain how semantic associations first take shape. Through our analysis, we reveal that each set of weights of the transformer has closed-form expressions as simple compositions of three basis functions--bigram, token-interchangeability, and context mappings--reflecting the statistics in the text corpus and uncover how each component of the transformer captures the semantic association based on these compositions. Experiments on real-world LLMs demonstrate that our theoretical weight characterizations closely match the learned weights, and qualitative analyses further guide us on how our theorem shines light on interpreting the learned association in transformers.

## Tiny Stories Experiments

For Tiny Stories, first tokenize the dataset using either [tiny-tokenize-natural.py](./tiny-tokenize-natural.py) which uses common words as tokens or [tiny-tokenize-bpe.py](./tiny-tokenize-bpe.py) for Byte-pair encoding (BPE) tokenization. 

The corresponding training script [tiny-self-attn-natural.py](./tiny-self-attn-natural.py) or [tiny-self-attn-bpe.py](./tiny-self-attn-bpe.py) can be used to train a 3-layer attention-based Transformer. Plots can be created from the training logs using [tiny-plot-logs.py](./visualizations/tiny-plot-logs.py).

## Pythia Experiments

The comparison between the theoretical features and Pythia weights can be run using [pythia.py](./pythia.py) and visualized using [plot-cosine-similarity.py](./visualizations/plot-cosine-similarity.py).

The comparison for individual attention heads can be collected using [pythia-per-head.py](./pythia-per-head.py) and visualized using [plot-per-head-similarity.py](./visualizations/plot-per-head-similarity.py).


