# MicroPoetry

This projects extends Andrej Karpathy's **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** by building a hardware-accelerated GPT model for classical Chinese poetry. 
We target specifically at the 七言律诗 (seven-character regulated verse) format, where a poem contains 8 lines, with each line containing 7 characters. 
The main goal of the project is to build an inference engine in RTL for the quantized transformer model, implemented on a Xilinx Zynq-7020 FPGA. 

## Model

The model is adjusted from microgpt to support batch training using PyTorch, retaining the multi-head attention transformer architecture while increasing the number of layers to 4.  

The training data contains 68968 poems from Tang and Song dynasties, all in the format of 七言律诗 (seven-character regulated verse). Due to hardware constraints, top 3000 most frequent characters are used as the vocabulary, with rare characters replaced by a special token `<UNK>`. The start and end of poem are marked by `<BOS>` and `<EOS>` tokens, respectively, and the title and body of the poem are separated by `<SEP>`. There is also a `<PAD>` token that is currently unused but reserved for future use, if needed. This yields a total vocabulary size of 3005. To avoid too long titles, the maximum title length is set to 20 characters. 

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Vocabulary Size | 3005 | 3000 most frequent characters + `<UNK>`, `<BOS>`, `<EOS>`, `<SEP>`, and `<PAD>` |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 4 | Number of transformer layers |
| `block_size` | 96 | Maximum sequence length (title + 56-char body + special tokens, with margin) |
| Parameter Size | 395,702 | Total number of trainable parameters |
| Weight Tying | On | Input token embedding weights tied to language model head weights |

The model is trained with a batch size of 64 until the validation loss stabilizes. Inference remains sequential with the following constraints applied:

- The tone (平仄) of the characters must satisfy the [tonal patterns](./model/Data/templates.json) of 七言律诗 (seven-character regulated verse).
- The characters that rhyme must belong to the same 平水韵 (Ping Shui Yun) [rhyme group](./model/Data/rhyme_table.json).

The rhyming characters are restricted not to repeat, and repetition in general is penalized. For a user-provided title, the model will generate a poem with relevant content. The script is available on [Google Colab](https://colab.research.google.com/drive/1t-YuAHcHE9ubuFYf_acDLS3uS68gfpKT?usp=sharing).

## Quantization

With the model trained in `Float32`, we explored how quantization in inference affects the quality of the generated poems, which is evaluated by the validation loss. The model weights are first quantized to `INT8`, `INT4`, and a mixed precision of `INT8` for embedding and `INT4` for the transformer layers. For each of the data types, we applied per-channel and per-tensor quantization. For each of the quantization schemes, we evaluated the validation loss on 100 randomly selected batches of size 64 from the validation set. The results are summarized below:

| Quantization Scheme | Validation Loss | Memory Footprint |
|---------------------|-----------------|------------------|
| `float32`           | ~4.905          | ~1.54MB          |
| `INT8` per-channel  | ~4.886          | ~407KB           |
| `INT8` per-tensor   | ~4.896          | ~386KB           |
| Mixed per-channel   | ~5.057          | ~311KB           |
| Mixed per-tensor    | ~5.541          | ~290KB           |
| `INT4`per-channel   | ~5.134          | ~214KB           |
| `INT4`per-tensor    | ~8.344          | ~193KB           |

`INT8` per-channel and per-tensor quantizations show similar performance compared to the `float32`. Here we choose `INT8` per-tensor quantization for its memory efficiency, with a footprint of ~386KB, which is within the 560KB on-chip BRAM available on the Zynq-7020 FPGA. Per-tensor `INT8` quantization is performed for the activations as well. 

## Acknowledgments

This project builds on:

- **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** by Andrej Karpathy — the minimal, dependency-free GPT implementation that this project's architecture is derived from.
- **[gateGPT](https://github.com/fguzman82/gateGPT/tree/main)** by fguzman82 — an RTL implmenentation of microgpt running on Vertex-5 FPGA, referenced for architectural inspiration.
- **[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)** by jackeyGao — the Tang and Song poetry corpus (全唐诗, 全宋诗) used for training data.
- **[chinese_word_rhyme](https://github.com/charlesix59/chinese_word_rhyme)** by charlesix59 — 平水韵 rhyme category and tone (平仄) reference tables used for structural constraints.

See [LICENSE](LICENSE) and individual source repos for their respective licensing terms.
