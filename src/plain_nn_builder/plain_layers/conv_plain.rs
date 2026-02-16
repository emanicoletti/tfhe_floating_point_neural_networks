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
    pub velocity_weights: Option<PlainTensor<T>>,
    pub velocity_biases: Option<PlainTensor<T>>,
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
            velocity_weights: None,
            velocity_biases: None,
        }
    }
}

impl<T> PlainLayer<T> for PlainConv2DLayer<T>
where
    T: PlainAdd + PlainSub + PlainMul + PlainMulExact + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default,
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
        
        let padding = self.padding;
        let stride = self.stride;

        let mut padded_data = vec![T::default(); batch_size * in_channels * padded_height * padded_width];
        
        let pad_stride_c = padded_height * padded_width;
        let in_stride_c = in_height * in_width;

        padded_data.par_chunks_exact_mut(pad_stride_c)
            .zip(input.data.par_chunks_exact(in_stride_c))
            .for_each(|(pad_chan_slice, in_chan_slice)| {
                for h in 0..in_height {
                    let src_row = &in_chan_slice[h * in_width..(h + 1) * in_width];
                    // dst_offset = (row + padding) * width + padding_left
                    let dst_offset = (h + padding) * padded_width + padding;
                    
                    pad_chan_slice[dst_offset..dst_offset + in_width]
                        .copy_from_slice(src_row);
                }
            });

        let mut output_data = vec![T::default(); batch_size * out_channels * out_height * out_width];
        
        let out_stride_image = out_height * out_width;
        let pad_stride_b = in_channels * pad_stride_c;
        let weight_stride_oc = in_channels * kernel_height * kernel_width;

        output_data.par_chunks_exact_mut(out_stride_image)
            .enumerate()
            .for_each(|(global_idx, out_image_slice)| {
                let b = global_idx / out_channels;
                let oc = global_idx % out_channels;

                let bias = self.biases.data[oc];

                let weights_base_ptr = unsafe { 
                    self.weights.data.as_ptr().add(oc * weight_stride_oc) 
                };
                let input_base_ptr = unsafe { 
                    padded_data.as_ptr().add(b * pad_stride_b) 
                };

                for oh in 0..out_height {
                    let h_start = oh * stride;
                    
                    for ow in 0..out_width {
                        let w_start = ow * stride;
                        let mut acc = bias;

                        let mut weight_ptr = weights_base_ptr;

                        unsafe {
                            for ic in 0..in_channels {
                                let input_ic_ptr = input_base_ptr.add(ic * pad_stride_c);

                                for kh in 0..kernel_height {
                                    let row_offset = (h_start + kh) * padded_width + w_start;
                                    let input_row_ptr = input_ic_ptr.add(row_offset);

                                    for kw in 0..kernel_width {
                                        let val_i = *input_row_ptr.add(kw);
                                        let val_w = *weight_ptr;
                                        
                                        acc = acc.add(val_i.mul(val_w));
                                        
                                        weight_ptr = weight_ptr.add(1);
                                    }
                                }
                            }
                        }
                        
                        out_image_slice[oh * out_width + ow] = acc;
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
        let padding = self.padding;
        
        let stride = self.stride;

        let i_stride_c = in_height * in_width;
        let p_stride_c = padded_height * padded_width;
        let p_stride_b = in_channels * p_stride_c;
        let o_stride_oc = out_height * out_width;
        let o_stride_b = out_channels * o_stride_oc;
        let w_stride_ic = kernel_height * kernel_width;
        let w_stride_oc = in_channels * w_stride_ic;

        let mut padded_input_data = vec![T::default(); batch_size * p_stride_b];

        padded_input_data.par_chunks_exact_mut(p_stride_c)
            .zip(input.data.par_chunks_exact(i_stride_c))
            .for_each(|(pad_chan, in_chan)| {
                for h in 0..in_height {
                    let src_idx = h * in_width;
                    let dst_idx = (h + padding) * padded_width + padding;
                    pad_chan[dst_idx..dst_idx + in_width]
                        .copy_from_slice(&in_chan[src_idx..src_idx + in_width]);
                }
            });

        let mut grad_weights_data = vec![T::default(); out_channels * w_stride_oc];
        let mut grad_biases_data = vec![T::default(); out_channels];

        grad_weights_data.par_chunks_exact_mut(w_stride_oc)
            .zip(grad_biases_data.par_iter_mut())
            .enumerate()
            .for_each(|(oc, (dw_slice, db_val))| {
                let mut local_db = T::default();
                
                let go_oc_offset = oc * o_stride_oc;

                for b in 0..batch_size {
                    let go_batch_base = b * o_stride_b + go_oc_offset;
                    let input_batch_base = b * p_stride_b;
                    
                    let go_ptr_base = unsafe { grad_output.data.as_ptr().add(go_batch_base) };
                    let input_ptr_base = unsafe { padded_input_data.as_ptr().add(input_batch_base) };

                    for oh in 0..out_height {
                        let h_start = oh * stride;
                        for ow in 0..out_width {
                            let w_start = ow * stride;
                            
                            let grad_out_val = unsafe { *go_ptr_base.add(oh * out_width + ow) };
                            local_db = local_db.add(grad_out_val);

                            let mut dw_ptr = dw_slice.as_mut_ptr();

                            for ic in 0..in_channels {
                                let input_ic_ptr = unsafe { input_ptr_base.add(ic * p_stride_c) };
                                
                                for kh in 0..kernel_height {
                                    let in_row_ptr = unsafe { 
                                        input_ic_ptr.add((h_start + kh) * padded_width + w_start) 
                                    };

                                    unsafe {
                                        for kw in 0..kernel_width {
                                            let x_val = *in_row_ptr.add(kw);
                                            *dw_ptr = (*dw_ptr).add(grad_out_val.mul(x_val));
                                            
                                            dw_ptr = dw_ptr.add(1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                *db_val = local_db;
            });

        let mut grad_input_padded = vec![T::default(); batch_size * p_stride_b];

        grad_input_padded.par_chunks_exact_mut(p_stride_c)
            .enumerate()
            .for_each(|(flat_idx, ic_slice)| {
                let b = flat_idx / in_channels;
                let ic = flat_idx % in_channels;
                let ic_w_offset = ic * w_stride_ic; 

                let go_batch_base = unsafe { grad_output.data.as_ptr().add(b * o_stride_b) };
                
                for oc in 0..out_channels {
                    let go_oc_base = unsafe { go_batch_base.add(oc * o_stride_oc) };
                    
                    let w_base_offset = oc * w_stride_oc + ic_w_offset;
                    let w_ptr_base = unsafe { self.weights.data.as_ptr().add(w_base_offset) };

                    for oh in 0..out_height {
                        let ih_start = oh * stride;
                        for ow in 0..out_width {
                            let iw_start = ow * stride;
                            
                            let go_val = unsafe { *go_oc_base.add(oh * out_width + ow) };
                            
                            for kh in 0..kernel_height {
                                let dst_row_idx = (ih_start + kh) * padded_width + iw_start;
                                let w_row_idx = kh * kernel_width;

                                unsafe {
                                    for kw in 0..kernel_width {
                                        let w_val = *w_ptr_base.add(w_row_idx + kw);
                                        let dst_ptr = ic_slice.as_mut_ptr().add(dst_row_idx + kw);
                                        *dst_ptr = (*dst_ptr).add(go_val.mul(w_val));
                                    }
                                }
                            }
                        }
                    }
                }
            });

        let mut grad_input_data = vec![T::default(); input.data.len()];

        grad_input_data.par_chunks_exact_mut(i_stride_c)
            .zip(grad_input_padded.par_chunks_exact(p_stride_c))
            .for_each(|(dst_chan, src_chan)| {
                for h in 0..in_height {
                    let dst_start = h * in_width;
                    let src_start = (h + padding) * padded_width + padding;
                    
                    dst_chan[dst_start..dst_start + in_width]
                        .copy_from_slice(&src_chan[src_start..src_start + in_width]);
                }
            });

        self.grad_weights = Some(PlainTensor { 
            data: grad_weights_data, 
            shape: self.weights.shape.clone() 
        });
        self.grad_biases = Some(PlainTensor { 
            data: grad_biases_data, 
            shape: vec![out_channels] 
        });

        PlainTensor { 
            data: grad_input_data, 
            shape: input.shape.clone() 
        }
    }


    fn update_parameters(&mut self, learning_rate: T, weight_decay: T, momentum: T)
    where
        T: Send + Sync + Copy + PlainElement, 
    {
        let zero = T::from_f32(0.0);

        if self.velocity_weights.is_none() {
            self.velocity_weights = Some(PlainTensor{
                data: vec![T::from_f32(0.0); self.weights.data.len()],
                shape: self.weights.shape.clone(),
            });
        }
        if self.velocity_biases.is_none() {
            self.velocity_biases = Some(PlainTensor{
                data: vec![T::from_f32(0.0); self.biases.data.len()],
                shape: self.biases.shape.clone(),
            });
        }

        if let (Some(grad_w), Some(grad_b), Some(vel_w), Some(vel_b)) = (
            &self.grad_weights,
            &self.grad_biases,
            &mut self.velocity_weights,
            &mut self.velocity_biases
        ) {
            self.weights.data.par_iter_mut()
                .zip(grad_w.data.par_iter())
                .zip(vel_w.data.par_iter_mut())
                .for_each(|((w, &g), v)| {
                    let g_prime = if weight_decay.to_f32() != zero.to_f32() {
                        let wd_term = w.mul(weight_decay);
                        g.add(wd_term)
                    } else {
                        g 
                    };
                    
                    let step = if momentum.to_f32() != zero.to_f32() {
                        let v_momentum = v.mul_exact(momentum);
                        *v = v_momentum.add(g_prime);
                        v.mul(learning_rate)
                    } else {
                        *v = g_prime; 
                        g_prime.mul(learning_rate)
                    };

                    *w = w.sub(step);
                });

            self.biases.data.par_iter_mut()
                .zip(grad_b.data.par_iter())
                .zip(vel_b.data.par_iter_mut())
                .for_each(|((b, &g), v)| {
                    
                    let step = if momentum.to_f32() != zero.to_f32() {
                        let v_momentum = v.mul(momentum); 
                        *v = v_momentum.add(g);
                        v.mul(learning_rate)
                    } else {
                        *v = g;
                        g.mul(learning_rate)
                    };

                    *b = b.sub(step);
                });

        } else {
            panic!("Missing gradients or velocities for convolution layer");
        }
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }
    
    fn exact_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {

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

                                    acc = acc.add(input_val.mul_exact(weight_val));
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