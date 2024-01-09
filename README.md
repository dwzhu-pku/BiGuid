# BiGuid
This repository presents the code for the paper "Probing Bilingual Guidance for Cross-Lingual Summarization" (NLPCC 2023)


## Reproduction

- `fairseq` version: 0.12.0
- Python version: 3.10.12
- PyTorch version: 2.0.1


To reproduce the results, please follow these steps:

1. Create a conda environment and install the required packages using pip:
    ```bash
    conda env create -n your_env_name python=3.10 -y
    pip install -r requirements.txt
    ```

2. Install `fairseq` in editable mode by navigating to the `fairseq` directory and running the following command:
    ```bash
    cd fairseq
    pip install --editable ./
    ```

3. Download the content of [`mix_data`](https://drive.google.com/drive/folders/1gnrDD9g-F_EAzCevscBx1GcoAdMLlFXR?usp=sharing) and [`mix_data-bin`](https://drive.google.com/drive/folders/1rKaIOAFnODoumdS1DAsS6X3YcW1WfgSB?usp=sharing) under `data/BiNews`.

4. Navigate to the corresponding folder and start training via shell script. Take performing English-to-Chinese summarization with Chinese guidance in encoder side as an example:
    ```bash
    cd enzh2zh
    bash enzh2zh_transformer.sh
    ```

5. Do inference and evaluation using `inference.sh`. Note that `files2rouge` need to be installed for for calculating ROUGE scores.

## Citation

You may cite our paper as follows:

```bibtex
@inproceedings{zhu2023probing,
  title={Probing Bilingual Guidance for Cross-Lingual Summarization},
  author={Zhu, Dawei and Wu, Wenhao and Li, Sujian},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={749--760},
  year={2023},
  organization={Springer}
}
```

## Contact

If you have any questions, feel free to open an issue, or contact [dwzhu@pku.edu.cn;](mailto:dwzhu@pku.edu.cn)

