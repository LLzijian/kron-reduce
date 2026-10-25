# Synthetic implicit least-squares evidence: scope clarification

The rank-deficient example in the original receipt was checked only with a matrix-rank assertion. It was **not** passed to the reference least-squares solver or another solver to observe a rejection. The receipt therefore does not establish a production solver's rank-deficiency rejection behavior.

The checks also do not cover weak-direction gradients, repeated positive singular values, or an end-to-end/full-chain gradient. The original script and receipt are preserved byte-for-byte; this addendum contains no new numerical calculation.

Original receipt SHA-256: `A5F79A96EDB74356838D45AD107238AA55C7C86C749AB3C1720589F2D5F368FE`

Original script SHA-256: `2B735A56A4BFE6A54E9CED7DE1265294F383615F9A57C3F49306643DBC7D0062`
