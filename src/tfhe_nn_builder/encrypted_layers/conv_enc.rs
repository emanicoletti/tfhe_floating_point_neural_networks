use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::EncryptedContext;
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::{EncryptableValueType, EncryptedElement};
use crate::tfhe_nn_builder::encrypted_ops::{EncryptedAdd, EncryptedMul, EncryptedNegate};
use crate::tfhe_nn_builder::encrypted_layers::EncryptedLayer;

use rayon::prelude::*;

pub struct EncryptedConvLayer<T: EncryptedElement> {
    pub id: String,
    pub weights: EncryptedTensor<T>, // shape: [output_dim, input_dim, kernel_height, kernel_width]
    pub biases: EncryptedTensor<T>,  // shape: [1, 1, 1, output_dim]
    pub grad_weights: Option<EncryptedTensor<T>>,
    pub grad_biases: Option<EncryptedTensor<T>>,
    pub stride: usize,
    pub padding: usize
}

impl<T: EncryptedElement> EncryptedConvLayer<T> {
    pub fn _new(id: String, weights: EncryptedTensor<T>, biases: EncryptedTensor<T>, stride: usize, padding: usize) -> Self {
        Self {
            id,
            weights, // expect shape [output_dim, input_dim, kernel_height, kernel_width]
            biases,  // expect shape [1, 1, 1, output_dim]
            grad_weights: None,
            grad_biases: None,
            stride,
            padding
        }
    }
}

impl<K, T> EncryptedLayer<K, T> for EncryptedConvLayer<T>
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedNegate<K, T>,
    T: Clone + EncryptedElement + EncryptableValueType,
{
    fn forward(&mut self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
        let (batch_size, in_channels, in_height, in_width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) = (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);
        
        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;

        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;

        let mut padded_input = EncryptedTensor {
            data: vec![ctx.encrypted_zero.clone(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        // Copy original input into padded tensor
        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let src_idx = input.flatten_index(&[b, c, h, w]);
                        let dst_idx = padded_input.flatten_index(&[b, c, h + self.padding, w + self.padding]);
                        padded_input.data[dst_idx] = input.data[src_idx].clone();
                    }
                }
            }
        }

        let mut output = EncryptedTensor{
            data: vec![ctx.encrypted_zero.clone(); batch_size * out_channels * out_height * out_width],
            shape: vec![batch_size, out_channels, out_height, out_width],
        };

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut acc: T = self.biases.data[oc].clone();
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;

                                    let input_idx = padded_input.flatten_index(&[b, ic, ih, iw]);
                                    let weight_idx = self.weights.flatten_index(&[oc, ic, kh, kw]);
                                    
                                    let input_val = padded_input.data[input_idx].clone();
                                    let weight_val = self.weights.data[weight_idx].clone();
                                    
                                    acc = ctx.server_key.add(acc, ctx.server_key.mul(input_val, weight_val, ctx), ctx);
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
        input: &EncryptedTensor<T>,
        grad_output: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T> {
        let (batch_size, in_channels, in_height, in_width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) = (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);

        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;

        let grad_output = grad_output.unflatten_1d_to_hw(&[batch_size, out_channels, out_height, out_width], ctx);

        let mut grad_input = EncryptedTensor {
            data: vec![ctx.encrypted_zero.clone(); input.data.len()],
            shape: input.shape.clone(),
        };

        let mut grad_weights = EncryptedTensor {
            data: vec![ctx.encrypted_zero.clone(); self.weights.data.len()],
            shape: self.weights.shape.clone(),
        };

        let mut grad_biases = EncryptedTensor {
            data: vec![ctx.encrypted_zero.clone(); out_channels],
            shape: vec![out_channels],
        };

        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;

        let mut grad_input_padded = EncryptedTensor {
            data: vec![ctx.encrypted_zero.clone(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        let mut padded_input = EncryptedTensor {
            data: vec![ctx.encrypted_zero.clone(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let src = input.flatten_index(&[b, c, h, w]);
                        let dst = padded_input.flatten_index(&[b, c, h + self.padding, w + self.padding]);
                        padded_input.data[dst] = input.data[src].clone();
                    }
                }
            }
        }

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let grad_out_val = grad_output.data[grad_output.flatten_index(&[b, oc, oh, ow])].clone();

                        // Bias gradient
                        grad_biases.data[oc] = ctx.server_key.add(grad_biases.data[oc].clone(), grad_out_val.clone(), ctx);

                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;

                                    let input_idx = padded_input.flatten_index(&[b, ic, ih, iw]);
                                    let x_val = padded_input.data[input_idx].clone();

                                    let w_idx = self.weights.flatten_index(&[oc, ic, kh, kw]);
                                    let w_val = self.weights.data[w_idx].clone();

                                    grad_weights.data[w_idx] = ctx.server_key.add(
                                        grad_weights.data[w_idx].clone(),
                                        ctx.server_key.mul(
                                            grad_out_val.clone(),
                                            x_val.clone(),
                                            ctx,
                                        ),
                                        ctx,
                                    );

                                    let gip_idx = grad_input_padded.flatten_index(&[b, ic, ih, iw]);
                                    grad_input_padded.data[gip_idx] = ctx.server_key.add(
                                        grad_input_padded.data[gip_idx].clone(),
                                        ctx.server_key.mul(grad_out_val.clone(), w_val.clone(), ctx),
                                        ctx,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Remove padding from grad_input ---
        for b in 0..batch_size {
            for ic in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let padded_idx = grad_input_padded.flatten_index(&[b, ic, h + self.padding, w + self.padding]);
                        let idx = grad_input.flatten_index(&[b, ic, h, w]);
                        grad_input.data[idx] = grad_input_padded.data[padded_idx].clone();
                    }
                }
            }
        }

        // Save gradients for optimizer
        self.grad_weights = Some(grad_weights);
        self.grad_biases = Some(grad_biases);

        grad_input
    }

    fn update_parameters(
            &mut self,
            learning_rate: T,
            ctx: &EncryptedContext<K, T>,
        ) {
            if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
                self.weights.data
                    .par_iter_mut()
                    .zip(grad_w.data.par_iter())
                    .for_each(|(w, gw)| {
                        let mul = ctx.server_key.negate(
                            ctx.server_key.mul(gw.clone(), learning_rate.clone(), ctx)
                        );
                        *w = ctx.server_key.add(w.clone(), mul, ctx);
                    });

                // Update biases in parallel
                self.biases.data
                    .par_iter_mut()
                    .zip(grad_b.data.par_iter())
                    .for_each(|(b, gb)| {
                        let mul = ctx.server_key.negate(
                            ctx.server_key.mul(gb.clone(), learning_rate.clone(), ctx)
                        );
                        *b = ctx.server_key.add(b.clone(), mul, ctx);
                    });

            } else {
                panic!("Missing gradients for weights or biases");
            }
    }

    fn get_weights(&self) -> EncryptedTensor<T> {
        self.weights.clone()    
    }

    fn get_biases(&self) -> EncryptedTensor<T> {
        self.biases.clone()
    }

    fn get_grad_weights(&self) -> EncryptedTensor<T> {
        self.grad_weights.clone().unwrap()
    }

    fn get_grad_biases(&self) -> EncryptedTensor<T> {
        self.grad_biases.clone().unwrap()
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}