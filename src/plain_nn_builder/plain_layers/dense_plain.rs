use crate::plain_nn_builder::plain_layers::PlainLayer;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;

use rayon::iter::*;
use rayon::prelude::*;
use rayon::scope;

pub struct PlainDenseLayer<T: PlainElement> {
    pub id: String,
    pub weights: PlainTensor<T>,
    pub biases: PlainTensor<T>,
    pub grad_weights: Option<PlainTensor<T>>,
    pub grad_biases: Option<PlainTensor<T>>,
    pub velocity_weights: Option<PlainTensor<T>>,
    pub velocity_biases: Option<PlainTensor<T>>,
}

impl<T: PlainElement> PlainDenseLayer<T> {
    pub fn _new(id: String, weights: PlainTensor<T>, biases: PlainTensor<T>) -> Self {
        Self {
            id,
            weights,
            biases,
            grad_weights: None,
            grad_biases: None,
            velocity_weights: None,
            velocity_biases: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainDenseLayer<T>
where
    T: PlainAdd
        + PlainSub
        + PlainMul
        + PlainMulExact
        + Send
        + Sync
        + Clone
        + PlainElement
        + PlainValueType
        + Copy
        + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let out_features = self.biases.shape[3];
        let flatten_input = input.flatten_hw_to_1d();
        let mut output = flatten_input.matmul(&self.weights.transpose());

        let bias_data = &self.biases.data;

        output
            .data
            .par_chunks_exact_mut(out_features)
            .for_each(|row| {
                for i in 0..out_features {
                    row[i] = row[i].add(bias_data[i]);
                }
            });

        output
    }

    fn backward(&mut self, input: &PlainTensor<T>, grad_output: &PlainTensor<T>) -> PlainTensor<T> {
        let mut grad_weights_opt = None;
        let mut grad_biases_opt = None;
        let mut grad_input_opt = None;

        scope(|s| {
            s.spawn(|_| {
                let flatten_input = input.flatten_hw_to_1d();
                let grad_weights = grad_output
                    .transpose()
                    .matmul(&flatten_input)
                    .sum_on_first_axis();
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

    fn update_parameters(&mut self, learning_rate: T, weight_decay: T, momentum: T) {
        let zero = T::from_f32(0.0);

        if self.velocity_weights.is_none() {
            self.velocity_weights = Some(PlainTensor {
                data: vec![T::from_f32(0.0); self.weights.data.len()],
                shape: self.weights.shape.clone(),
            });
        }
        if self.velocity_biases.is_none() {
            self.velocity_biases = Some(PlainTensor {
                data: vec![T::from_f32(0.0); self.biases.data.len()],
                shape: self.biases.shape.clone(),
            });
        }

        if let (Some(grad_w), Some(grad_b), Some(vel_w), Some(vel_b)) = (
            &self.grad_weights,
            &self.grad_biases,
            &mut self.velocity_weights,
            &mut self.velocity_biases,
        ) {
            let weights = &mut self.weights;
            let biases = &mut self.biases;

            scope(|s| {
                s.spawn(|_| {
                    let g_prime = if weight_decay.to_f32() != zero.to_f32() {
                        let wd_term = weights.mul_scalar(&weight_decay);
                        grad_w.add(&wd_term)
                    } else {
                        grad_w.clone()
                    };

                    if momentum.to_f32() != zero.to_f32() {
                        let momentum_term = vel_w.mul_scalar(&momentum);
                        *vel_w = momentum_term.add(&g_prime);

                        let step = vel_w.mul_scalar(&learning_rate);
                        *weights = weights.sub(&step);
                    } else {
                        let step = g_prime.mul_scalar(&learning_rate);
                        *weights = weights.sub(&step);
                    }
                });

                s.spawn(|_| {
                    if momentum.to_f32() != zero.to_f32() {
                        let momentum_term = vel_b.mul_scalar(&momentum);
                        *vel_b = momentum_term.add(grad_b);

                        let step = vel_b.mul_scalar(&learning_rate);
                        *biases = biases.sub(&step);
                    } else {
                        let step = grad_b.mul_scalar(&learning_rate);
                        *biases = biases.sub(&step);
                    }
                });
            });
        }
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn exact_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let flatten_input = input.flatten_hw_to_1d();

        let mut weighted_sum = flatten_input.exact_matmul(&self.weights.transpose());

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

    fn get_weights(&self) -> PlainTensor<T> {
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
