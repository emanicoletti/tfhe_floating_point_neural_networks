use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

use rayon::iter::*;
use rayon::prelude::*;
use rayon::scope;

pub struct PlainConv2DLayer<T: PlainElement> {
    pub id: String,
    pub weights: PlainTensor<T>,      
    pub biases: PlainTensor<T>,        
    pub grad_weights: Option<PlainTensor<T>>,
    pub grad_biases: Option<PlainTensor<T>>,
}

impl<T: PlainElement> PlainConv2DLayer<T> {
    pub fn new(id: String, weights: PlainTensor<T>, biases: PlainTensor<T>) -> Self {
        Self {
            id,
            weights,
            biases,
            grad_weights: None,
            grad_biases: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainConv2DLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {

        let (batch_size, in_channels, in_height, in_width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) = (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);
        // Assuming stride of 2 for both height and width
        let stride = 2;
        let out_height = (in_height - kernel_height) / stride + 1;
        let out_width = (in_width - kernel_width) / stride + 1;

        let mut output = PlainTensor{
            data: vec![T::default(); batch_size * out_channels * out_height * out_width],
            shape: vec![batch_size, out_channels, out_height, out_width],
        };

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut acc: T = self.biases.data[oc];
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * stride + kh;
                                    let iw = ow * stride + kw;
                                    let input_idx = input.flatten_index(&[b, ic, ih, iw]);
                                    let weight_idx = self.weights.flatten_index(&[oc, ic, kh, kw]);
                                    let input_val = input.data[input_idx];
                                    let weight_val = self.weights.data[weight_idx];
                                    acc = acc.add(input_val.mul(weight_val));
                                }
                            }
                        }
                        let out_idx = output.flatten_index(&[b, oc, oh, ow]);
                        output.data[out_idx] = acc;
                    }
                }
            }
        }
        output
    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let (batch_size, in_channels, in_height, in_width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) = (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);
        
        let stride = 2;
        let out_height = (in_height - kernel_height) / stride + 1;
        let out_width = (in_width - kernel_width) / stride + 1;

        // 1. Initialize gradients
        let mut grad_input = PlainTensor {
            data: vec![T::default(); input.data.len()],
            shape: input.shape.clone(),
        };

        let mut grad_weights = PlainTensor {
            data: vec![T::default(); self.weights.data.len()],
            shape: self.weights.shape.clone(),
        };

        let mut grad_biases = PlainTensor {
            data: vec![T::default(); out_channels],
            shape: vec![out_channels],
        };
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let grad_out_idx = grad_output.flatten_index(&[b, oc, 0, oh * stride + ow]);
                        let dy = grad_output.data[grad_out_idx];

                        // Bias gradient
                        grad_biases.data[oc] = grad_biases.data[oc].add(dy);

                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * stride + kh;
                                    let iw = ow * stride + kw;

                                    let input_idx = input.flatten_index(&[b, ic, ih, iw]);
                                    let x_val = input.data[input_idx];

                                    let weight_idx = self.weights.flatten_index(&[oc, ic, kh, kw]);
                                    let w_val = self.weights.data[weight_idx];

                                    // Gradient w.r.t. input
                                    let dx_idx = grad_input.flatten_index(&[b, ic, ih, iw]);
                                    grad_input.data[dx_idx] = grad_input.data[dx_idx].add(dy.mul(w_val));

                                    // Gradient w.r.t. weights
                                    grad_weights.data[weight_idx] = grad_weights.data[weight_idx].add(dy.mul(x_val));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Save gradients for optimizer
        self.grad_weights = Some(grad_weights);
        self.grad_biases = Some(grad_biases);

        grad_input
    }

    fn update_parameters(&mut self, learning_rate: T)
        where
            T: Send + Sync + Copy,
        {
            if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
                // Update weights in parallel
                self.weights.data
                    .par_iter_mut()
                    .zip(grad_w.data.par_iter())
                    .for_each(|(w, &gw)| {
                        *w = w.sub(gw.mul(learning_rate.clone()));
                    });

                // Update biases in parallel
                self.biases.data
                    .par_iter_mut()
                    .zip(grad_b.data.par_iter())
                    .for_each(|(b, &gb)| {
                        *b = b.sub(gb.mul(learning_rate.clone()));
                    });
            } else {
                panic!("Missing gradients for weights or biases");
            }
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