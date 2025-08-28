use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

use rayon::iter::*;
use rayon::prelude::*;
use rayon::scope;

use std::panic;

pub struct PlainDenseLayer<T: PlainElement> {
    pub id: String,
    pub weights: PlainTensor<T>,
    pub biases: PlainTensor<T>,
    pub grad_weights: Option<PlainTensor<T>>,
    pub grad_biases: Option<PlainTensor<T>>
}

impl<T: PlainElement> PlainDenseLayer<T>{
    pub fn new(id: String, weights: PlainTensor<T>, biases: PlainTensor<T>) -> Self {
        Self {
            id,
            weights, // expect shape [1, 1, input_dim, output_dim]
            biases,  // expect shape [1, 1, 1, output_dim]
            grad_weights: None,
            grad_biases: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainDenseLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + Send + Sync + Clone + PlainElement + PlainValueType + Copy, 
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {

        let flatten_input = input.flatten_hw_to_1d();

        let mut weighted_sum = flatten_input.matmul(&self.weights.transpose()); 

        // Expand biases to match [batch_size, output_dim]
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
                    let grad_weights = grad_output.transpose().matmul(&flatten_input).sum_axis(0);
                    grad_weights_opt = Some(grad_weights);
                });
        
                s.spawn(|_| {
                    let grad_biases = grad_output.sum_axis(0);
                    grad_biases_opt = Some(grad_biases);
                });
        
                s.spawn(|_| {
                    let grad_input = grad_output.matmul(&self.weights);
                    grad_input_opt = Some(grad_input);
                });
            });

            // Unwrap results (these will always be Some because the spawns run synchronously)
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

            // Compute scalar multiplications in parallel
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
    
            // Subtractions can also be parallelized
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

    fn get_weights(&self) -> PlainTensor<T> 
    {
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