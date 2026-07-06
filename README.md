# MicroPoetry

This projects extends Andrej Karpathy's **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** by building a hardware-accelerated GPT model for classical Chinese poetry. 
We target specifically at the 七言律诗 (seven-character regulated verse) format, where a poem contains 8 lines, with each line containing 7 characters. 
The main goal of the project is to build an inference engine in RTL for the quantized transformer model, implemented on a Xilinx Zynq-7020 FPGA. 

## Model

The model is adjusted from microgpt to support batch training using PyTorch, retaining the multi-head attention transformer architecture while increasing the number of layers to 4.  

The training data contains 68968 poems from Tang and Song dynasties, all in the format of 七言律诗 (seven-character regulated verse). Due to hardware constraints, top 3000 most frequent characters are used as the vocabulary, with rare characters replaced by a special token `<UNK>`. The start and end of poem are marked by `<BOS>` and `<EOS>` tokens, respectively, and the title and body of the poem are separated by `<SEP>`. This yields a total vocabulary size of 3004. To avoid too long titles, the maximum title length is set to 20 characters. 

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Vocabulary Size | 3004 | 3000 most frequent characters + `<UNK>`, `<BOS>`, `<EOS>`, and `<SEP>` |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 4 | Number of transformer layers |
| `block_size` | 96 | Maximum sequence length (title + 56-char body + special tokens, with margin) |
| Parameter Size | 395,702 | Total number of trainable parameters |
| Weight Tying | On | Input token embedding weights tied to language model head weights |

The model is trained with a batch size of 64 until the validation loss stabilizes. Inference remains sequential with the following constraints applied:

- The tone (平仄) of the characters must satisfy the [tonal patterns](./model/Data/templates.json) of 七言律诗 (seven-character regulated verse).
- The characters that rhyme must belong to the same 平水韵 (Ping Shui Yun) [rhyme group](./model/Data/rhyme_table.json).

It is also restricted that the rhyming characters should not repeat, and repetition in general is penalized. When the user provides a title, the model will generate a poem with relevant content. The entire script is available on [Google Colab](https://colab.research.google.com/drive/1t-YuAHcHE9ubuFYf_acDLS3uS68gfpKT?usp=sharing).

## Acknowledgments

This project builds on:

- **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** by Andrej Karpathy — the minimal, dependency-free GPT implementation that this project's architecture is derived from.
- **[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)** by jackeyGao — the Tang and Song poetry corpus (全唐诗, 全宋诗) used for training data.
- **[chinese_word_rhyme](https://github.com/charlesix59/chinese_word_rhyme)** by charlesix59 — 平水韵 rhyme category and tone (平仄) reference tables used for structural constraints.

See [LICENSE](LICENSE) and individual source repos for their respective licensing terms.
