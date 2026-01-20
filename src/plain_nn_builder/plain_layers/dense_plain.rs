use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

use rayon::iter::*;
use rayon::scope;
use rayon::prelude::*;

pub struct PlainDenseLayer<T: PlainElement> {
    pub id: String,
    pub weights: PlainTensor<T>,
    pub biases: PlainTensor<T>,
    pub grad_weights: Option<PlainTensor<T>>,
    pub grad_biases: Option<PlainTensor<T>>
}

impl<T: PlainElement> PlainDenseLayer<T>{
    pub fn _new(id: String, weights: PlainTensor<T>, biases: PlainTensor<T>) -> Self {
        Self {
            id,
            weights,
            biases, 
            grad_weights: None,
            grad_biases: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainDenseLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + PlainMulInf + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default, 
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        // 1. Flatten NCHW to (N, H*W*C) for matmul
        let batch_size = input.shape[0];
        let in_features = input.data.len() / batch_size;
        let out_features = self.biases.shape[3]; // Assuming weights are [out_features, in_features]
        let flatten_input = input.flatten_hw_to_1d();

        // 2. Perform Matrix Multiplication: (N, in_features) x (in_features, out_features)
        // Note: If your matmul is already parallelized, this is the heaviest part.
        // We assume self.weights is already [Out, In], so we matmul with Transpose.
        let mut output = flatten_input.matmul(&self.weights.transpose());

        // 3. Optimized Parallel Bias Addition
        // Instead of creating 'expanded_biases' (which allocates batch_size * output_dim memory),
        // we add the bias directly to each row of the output matrix in parallel.
        let bias_data = &self.biases.data;

        // Use par_chunks_exact_mut to process each batch (row) in parallel
        output.data.par_chunks_exact_mut(out_features)
            .for_each(|row| {
                for i in 0..out_features {
                    row[i] = row[i].add(bias_data[i]);
                }
            });

        output
    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let mut grad_weights_opt = None;
        let mut grad_biases_opt = None;
        let mut grad_input_opt = None;

        scope(|s| {
            s.spawn(|_| {
                let flatten_input = input.flatten_hw_to_1d();
                let grad_weights = grad_output.transpose().matmul(&flatten_input).sum_on_first_axis();
                grad_weights_opt = Some(grad_weights);
            });


            s.spawn(|_| {
                let grad_biases = grad_output.sum_on_first_axis();
                grad_biases_opt = Some(grad_biases);
            });

            s.spawn(|_| {
                let grad_input = grad_output.matmul(&self.weights);
                grad_input_opt = Some(grad_input);
            });
        });

        let grad_weights = grad_weights_opt.expect("grad_weights not computed");
        let grad_biases = grad_biases_opt.expect("grad_biases not computed");
        let grad_input = grad_input_opt.expect("grad_input not computed");

        self.grad_weights = Some(grad_weights.clone());
        self.grad_biases = Some(grad_biases.clone());

        grad_input

        }

    fn update_parameters(&mut self, learning_rate: T)
    {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
            let mut lr_grad_w_opt = None;
            let mut lr_grad_b_opt = None;

            scope(|s| {
                s.spawn(|_| {
                    lr_grad_w_opt = Some(grad_w.mul_scalar(&learning_rate));
                });
                s.spawn(|_| {
                    lr_grad_b_opt = Some(grad_b.mul_scalar(&learning_rate));
                });
            });
    
            let lr_grad_w = lr_grad_w_opt.expect("lr_grad_w not computed");
            let lr_grad_b = lr_grad_b_opt.expect("lr_grad_b not computed");

            let mut new_weights = None;
            let mut new_biases = None;

            scope(|s| {
                s.spawn(|_| {
                    new_weights = Some(self.weights.sub(&lr_grad_w));
                });
                s.spawn(|_| {
                    new_biases = Some(self.biases.sub(&lr_grad_b));
                });
            });
    
            self.weights = new_weights.expect("weights update failed");
            self.biases = new_biases.expect("biases update failed");

        }
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn approximate_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {

        let flatten_input = input.flatten_hw_to_1d();

        let mut weighted_sum = flatten_input.approx_matmul(&self.weights.transpose());

        let batch_size = input.shape[0];
        let output_dim = self.biases.shape[3];
        let bias_data = &self.biases.data;

        let repeated: Vec<T> = (0..batch_size)
            .into_par_iter()
            .flat_map(|_| bias_data.par_iter().cloned())
            .collect();

        let expanded_biases = PlainTensor {
            data: repeated,
            shape: vec![batch_size, 1, 1, output_dim],
        };

        weighted_sum = weighted_sum.add(&expanded_biases);
        weighted_sum

    }

    fn get_weights(&self) -> PlainTensor<T>{
        self.weights.clone()
    }

    fn get_biases(&self) -> PlainTensor<T> {
        self.biases.clone()
    }

    fn get_grad_weights(&self) -> PlainTensor<T> {
        self.grad_weights.clone().unwrap()
    }

    fn get_grad_biases(&self) -> PlainTensor<T> {
        self.grad_biases.clone().unwrap()
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }

}