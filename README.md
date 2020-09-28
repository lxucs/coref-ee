# Coreference Resolution with Entity Equalization

This repository implements the coreference resolution with SpanBERT and **Entity Equalization**, and it is based on the following two repositories:
* https://github.com/mandarjoshi90/coref
* https://github.com/kkjawz/coref-ee

Usage is the same as https://github.com/mandarjoshi90/coref.

For implementation of other higher-order inference methods, please see https://github.com/lxucs/coref-hoi.

##### Note

The implementation of Entity Equalization requires O(k^2) memory with k being the number of extracted spans, while other HOI approaches only requires O(k) memory (the maximum number of antecedents for all models is set to 50 which is constant. To keep the GPU memory usage within 32GB, we limit the maximum number of span candidate to be 300 (see [independent.py](independent.py) L309).
