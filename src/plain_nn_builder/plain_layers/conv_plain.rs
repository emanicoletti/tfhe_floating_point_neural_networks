use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

use rayon::iter::*;
use rayon::prelude::*;


pub struct PlainConv2DLayer<T: PlainElement> {
    pub id: String,
    pub weights: PlainTensor<T>,      
    pub biases: PlainTensor<T>,        
    pub grad_weights: Option<PlainTensor<T>>,
    pub grad_biases: Option<PlainTensor<T>>,
    pub stride: usize,
    pub padding: usize,
}

impl<T: PlainElement> PlainConv2DLayer<T> {
    pub fn _new(id: String, weights: PlainTensor<T>, biases: PlainTensor<T>, stride: usize, padding: usize) -> Self {
        Self {
            id,
            weights,
            biases,
            grad_weights: None,
            grad_biases: None,
            stride,
            padding,
        }
    }
}

impl<T> PlainLayer<T> for PlainConv2DLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + PlainMulInf + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let (batch_size, in_channels, in_height, in_width) =
            (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) =
            (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);

        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;

        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;

        // 1. Parallel Padding
        let mut padded_data = vec![T::default(); batch_size * in_channels * padded_height * padded_width];
        let p_stride_c = padded_height * padded_width;
        let p_stride_b = in_channels * p_stride_c;
        let i_stride_c = in_height * in_width;
        let i_stride_b = in_channels * i_stride_c;

        padded_data.par_chunks_exact_mut(p_stride_b).enumerate().for_each(|(b, b_slice)| {
            for c in 0..in_channels {
                for h in 0..in_height {
                    let src_start = b * i_stride_b + c * i_stride_c + h * in_width;
                    let dst_start = c * p_stride_c + (h + self.padding) * padded_width + self.padding;
                    b_slice[dst_start..dst_start + in_width]
                        .copy_from_slice(&input.data[src_start..src_start + in_width]);
                }
            }
        });

        let padded_input = PlainTensor {
            data: padded_data,
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        // 2. Parallel Convolution
        let mut output_data = vec![T::default(); batch_size * out_channels * out_height * out_width];
        let o_stride_oc = out_height * out_width;
        let o_stride_b = out_channels * o_stride_oc;
        
        // Pre-calculate weight strides for inner loops
        let w_stride_ic = kernel_height * kernel_width;
        let w_stride_oc = in_channels * w_stride_ic;

        // Parallelize over Batch AND Output Channels
        output_data.par_chunks_exact_mut(o_stride_oc).enumerate().for_each(|(flat_idx, oc_slice)| {
            let b = flat_idx / out_channels;
            let oc = flat_idx % out_channels;
            
            let bias = self.biases.data[oc];
            let weight_base = oc * w_stride_oc;
            let input_base = b * p_stride_b;

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut acc = bias;
                    let ih_start = oh * self.stride;
                    let iw_start = ow * self.stride;

                    for ic in 0..in_channels {
                        let input_ic_base = input_base + ic * p_stride_c;
                        let weight_ic_base = weight_base + ic * w_stride_ic;

                        for kh in 0..kernel_height {
                            let input_row_idx = input_ic_base + (ih_start + kh) * padded_width + iw_start;
                            let weight_row_idx = weight_ic_base + kh * kernel_width;

                            for kw in 0..kernel_width {
                                let input_val = padded_input.data[input_row_idx + kw];
                                let weight_val = self.weights.data[weight_row_idx + kw];
                                acc = acc.add(input_val.mul(weight_val));
                            }
                        }
                    }
                    oc_slice[oh * out_width + ow] = acc;
                }
            }
        });
        PlainTensor {
            data: output_data,
            shape: vec![batch_size, out_channels, out_height, out_width],
        }
    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T> {
        let (batch_size, in_channels, in_height, in_width) =
            (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) =
            (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);

        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;
        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;

        // Strides
        let i_stride_c = in_height * in_width;
        let i_stride_b = in_channels * i_stride_c;
        let p_stride_c = padded_height * padded_width;
        let p_stride_b = in_channels * p_stride_c;
        let o_stride_oc = out_height * out_width;
        let o_stride_b = out_channels * o_stride_oc;
        let w_stride_ic = kernel_height * kernel_width;
        let w_stride_oc = in_channels * w_stride_ic;

        // 1. Parallel Padding for Forward Input (needed for dW)
        let mut padded_input_data = vec![T::default(); batch_size * p_stride_b];
        padded_input_data.par_chunks_exact_mut(p_stride_b).enumerate().for_each(|(b, b_slice)| {
            for ic in 0..in_channels {
                for h in 0..in_height {
                    let src_idx = b * i_stride_b + ic * i_stride_c + h * in_width;
                    let dst_idx = ic * p_stride_c + (h + self.padding) * padded_width + self.padding;
                    b_slice[dst_idx..dst_idx + in_width].copy_from_slice(&input.data[src_idx..src_idx + in_width]);
                }
            }
        });

        // 2. Parallel Gradient Computation
        // We parallelize by Output Channel (oc) to compute dW and dB independently per channel
        let (grad_weights_data, grad_biases_data): (Vec<Vec<T>>, Vec<T>) = (0..out_channels)
            .into_par_iter()
            .map(|oc| {
                let mut local_dw = vec![T::default(); w_stride_oc];
                let mut local_db = T::default();
                
                for b in 0..batch_size {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let go_idx = b * o_stride_b + oc * o_stride_oc + oh * out_width + ow;
                            let grad_out_val = grad_output.data[go_idx];

                            // Update bias gradient
                            local_db = local_db.add(grad_out_val);

                            for ic in 0..in_channels {
                                let ih_start = oh * self.stride;
                                let iw_start = ow * self.stride;
                                
                                for kh in 0..kernel_height {
                                    let inp_idx = b * p_stride_b + ic * p_stride_c + (ih_start + kh) * padded_width + iw_start;
                                    let w_idx_offset = ic * w_stride_ic + kh * kernel_width;
                                    
                                    for kw in 0..kernel_width {
                                        let x_val = padded_input_data[inp_idx + kw];
                                        local_dw[w_idx_offset + kw] = local_dw[w_idx_offset + kw].add(grad_out_val.mul(x_val));
                                    }
                                }
                            }
                        }
                    }
                }
                (local_dw, local_db)
            })
            .unzip();

        // 3. Parallel Input Gradient (dX)
        // We parallelize by Batch and Input Channel for dX
        let mut grad_input_padded = vec![T::default(); batch_size * p_stride_b];
        
        // Note: To avoid atomic updates, we re-structure the loop so each thread owns a chunk of dX
        grad_input_padded.par_chunks_exact_mut(p_stride_c).enumerate().for_each(|(flat_idx, ic_slice)| {
            let b = flat_idx / in_channels;
            let ic = flat_idx % in_channels;

            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let go_idx = b * o_stride_b + oc * o_stride_oc + oh * out_width + ow;
                        let grad_out_val = grad_output.data[go_idx];

                        let ih_start = oh * self.stride;
                        let iw_start = ow * self.stride;

                        for kh in 0..kernel_height {
                            let w_idx = oc * w_stride_oc + ic * w_stride_ic + kh * kernel_width;
                            let dst_h_idx = (ih_start + kh) * padded_width + iw_start;
                            
                            for kw in 0..kernel_width {
                                let w_val = self.weights.data[w_idx + kw];
                                ic_slice[dst_h_idx + kw] = ic_slice[dst_h_idx + kw].add(grad_out_val.mul(w_val));
                            }
                        }
                    }
                }
            }
        });

        // 4. Unpad grad_input
        let mut grad_input_data = vec![T::default(); input.data.len()];
        grad_input_data.par_chunks_exact_mut(i_stride_b).enumerate().for_each(|(b, b_slice)| {
            for ic in 0..in_channels {
                for h in 0..in_height {
                    let src_idx = ic * p_stride_c + (h + self.padding) * padded_width + self.padding;
                    let dst_idx = ic * i_stride_c + h * in_width;
                    b_slice[dst_idx..dst_idx + in_width].copy_from_slice(
                        &grad_input_padded[b * p_stride_b + src_idx..b * p_stride_b + src_idx + in_width]
                    );
                }
            }
        });

        // Flatten dW results back into a single vector
        let flattened_dw: Vec<T> = grad_weights_data.into_iter().flatten().collect();

        self.grad_weights = Some(PlainTensor { data: flattened_dw, shape: self.weights.shape.clone() });
        self.grad_biases = Some(PlainTensor { data: grad_biases_data, shape: vec![out_channels] });

        PlainTensor { data: grad_input_data, shape: input.shape.clone() }
    }


    fn update_parameters(&mut self, learning_rate: T)
        where
            T: Send + Sync + Copy,
    {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
            self.weights.data
                .par_iter_mut()
                .zip(grad_w.data.par_iter())
                .for_each(|(w, &gw)| {
                    *w = w.sub(gw.mul(learning_rate.clone()));
                });
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

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }
    
    fn approximate_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {

        let (batch_size, in_channels, in_height, in_width) =
            (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) =
            (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);

        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;

        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;

        let mut padded_input = PlainTensor {
            data: vec![T::default(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let src_idx = input.flatten_index(&[b, c, h, w]);
                        let dst_idx = padded_input.flatten_index(&[b, c, h + self.padding, w + self.padding]);
                        padded_input.data[dst_idx] = input.data[src_idx];
                    }
                }
            }
        }

        let mut output = PlainTensor {
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
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;

                                    let input_idx = padded_input.flatten_index(&[b, ic, ih, iw]);
                                    let weight_idx = self.weights.flatten_index(&[oc, ic, kh, kw]);

                                    let input_val = padded_input.data[input_idx];
                                    let weight_val = self.weights.data[weight_idx];

                                    acc = acc.add(input_val.mul_inf(weight_val));
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