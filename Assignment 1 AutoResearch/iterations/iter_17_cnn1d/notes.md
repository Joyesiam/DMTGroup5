# Iteration 17: 1D-CNN Temporal Model

## Hypothesis
1D-CNN captures local patterns through convolutional filters and has fewer
parameters than GRU. On short sequences (7 days), CNNs can outperform
recurrent models.

## Change
- temporal="cnn1d" (was "gru")
- 2 conv layers, 32 filters, kernel_size=3, global avg pooling
