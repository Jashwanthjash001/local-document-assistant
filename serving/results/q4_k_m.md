| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | MTL,BLAS   |       6 |           pp512 |       204.19 ± 20.00 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | MTL,BLAS   |       6 |           tg128 |         24.43 ± 0.36 |

build: 6f1e65437 (12119)
