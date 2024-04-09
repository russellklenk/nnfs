- ch02-02/03 and ch2-04.py produce very slightly different results even though they implement the same computation.
  These scripts all use the same input and structure, but ch2-04 uses numpy. You've tried changing dtype, and
  changing argument order to no effect. Why is there a very slight difference? Does it matter? Try to produce the
  same result in C (easier to look at disassembly).

- ch03-05 (and the book) implement the forward function as `np.dot(inputs, self.weights) + self.biases`. There is no transpose
  of the weights array, while previously there was. At the top of page 67 the book says 'Note that we're initialzing weights to
  be (inputs, neurons) rather than (neurons, inputs). We're doing this ahead instead of transposing every time we perform a
  forward pass, as explained in the previous chapter.' Does this match ch03-03 and ch03-04?
  ANSWER: Yes - the weights in prior scripts have shape (neurons, inputs):
  N0 [ I00,  I01,  I02,  I03 ]
  N1 [ I10,  I11,  I12,  I13 ]
  N2 [ I20,  I21,  I22,  I23 ]

- ch03-05 during random initialization of the weights, the weights are scaled down by a constant factor to reduce their 
  magnitude. The book says that this reduces training time. Why?
  ANSWER: I believe this is because there is less space to 'search' during optimization when the values are smaller.

