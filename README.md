# MicroPoetry

This projects extends Andrej Karpathy's **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** by building a hardware-accelerated GPT model for classical Chinese poetry. 
We target specifically at the 七言律诗 (seven-character regulated verse) format, where a poem contains 8 lines, with each line containing 7 characters. 
The main goal of the project is to build an inference engine in RTL for the quantized transformer model, implemented on a Xilinx Zynq-7020 FPGA. 


## Acknowledgments

This project builds on:

- **[microgpt](https://karpathy.github.io/2026/02/12/microgpt/)** by Andrej Karpathy — the minimal, dependency-free GPT implementation that this project's architecture is derived from.
- **[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)** by jackeyGao — the Tang and Song poetry corpus (全唐诗, 全宋诗) used for training data.
- **[chinese_word_rhyme](https://github.com/charlesix59/chinese_word_rhyme)** by charlesix59 — 平水韵 rhyme category and tone (平仄) reference tables used for structural constraints.

See [LICENSE](LICENSE) and individual source repos for their respective licensing terms.
