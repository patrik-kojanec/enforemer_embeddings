# Instructions to extract Enformer's embeddings

1. Install Python 3.8 and the requirements.
2. Clone the [Enformer repository](https://github.com/deepmind/deepmind-research/tree/master/enformer) to `./enformer/`.
3. Download the [pretrained weigths](https://tfhub.dev/deepmind/enformer/1) and store them to `./weights/`.
4. Add to the following function to the Enformer class in the `./enformer/enformer.py` module:

```
### ADDITIONAL FUNCTION TO EXTRACT EMBEDDINGS ###
  @tf.function(input_signature=[
      tf.TensorSpec([None, SEQUENCE_LENGTH, 4], tf.float32)])
  def extract_features(self, x):
    return self.trunk(x, is_training=False)
```
5. Set the correct `SEQUENCE_LENGTH = ` variable (196_608 or 393_216, depending on the version of the weights).
6. In the `embed_input_sequences.ipynb` notebook, set the paths to the sequences and set the functions that read the sequences (TODOs).


## Additional testing:
Download the [testing data](https://drive.google.com/drive/folders/18UubaZRCAlJIQzBnexifWIUWJUrLb39B?usp=sharing) and run the examples present in the notebook.