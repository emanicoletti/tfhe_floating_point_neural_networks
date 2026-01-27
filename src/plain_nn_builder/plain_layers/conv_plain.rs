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
    T: PlainAdd + PlainSub + PlainMul + PlainMulInf + Send + Sync + Clone + PlainElement + PlainValueType + Copy + Default,
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let (batch_size, in_channels, in_height, in_width) =
            (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) =
            (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);

        // Output dimensions
        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;

        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;
        
        let padding = self.padding;
        let stride = self.stride;

        // --- 1. Parallel Padding ---
        // Allocation: We use default() (memset 0) which is optimized by the OS/Allocator.
        let mut padded_data = vec![T::default(); batch_size * in_channels * padded_height * padded_width];
        
        let pad_stride_c = padded_height * padded_width;
        let in_stride_c = in_height * in_width;

        // Parallel Copy: Efficient Memcpy
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

        // --- 2. Parallel Convolution ---
        let mut output_data = vec![T::default(); batch_size * out_channels * out_height * out_width];
        
        // Strides
        let out_stride_image = out_height * out_width;
        let pad_stride_b = in_channels * pad_stride_c;
        let weight_stride_oc = in_channels * kernel_height * kernel_width;

        // We process each (Batch, OutChannel) pair in parallel.
        output_data.par_chunks_exact_mut(out_stride_image)
            .enumerate()
            .for_each(|(global_idx, out_image_slice)| {
                let b = global_idx / out_channels;
                let oc = global_idx % out_channels;

                let bias = self.biases.data[oc];

                // Base pointers (Unsafe logic setup)
                // We offset into the global vectors to get the start of the data for this task.
                let weights_base_ptr = unsafe { 
                    self.weights.data.as_ptr().add(oc * weight_stride_oc) 
                };
                let input_base_ptr = unsafe { 
                    padded_data.as_ptr().add(b * pad_stride_b) 
                };

                // Iterate Output Spatial
                for oh in 0..out_height {
                    let h_start = oh * stride;
                    
                    for ow in 0..out_width {
                        let w_start = ow * stride;
                        let mut acc = bias;

                        // Pointers for the inner loop
                        // weight_ptr will walk linearly through the entire kernel volume.
                        let mut weight_ptr = weights_base_ptr;

                        unsafe {
                            // Loop over Input Channels
                            for ic in 0..in_channels {
                                // Input pointer for this channel
                                let input_ic_ptr = input_base_ptr.add(ic * pad_stride_c);

                                // Loop over Kernel Height
                                for kh in 0..kernel_height {
                                    // Calculate row start in padded input
                                    let row_offset = (h_start + kh) * padded_width + w_start;
                                    let input_row_ptr = input_ic_ptr.add(row_offset);

                                    // Loop over Kernel Width (Vectorizable Hot Loop)
                                    for kw in 0..kernel_width {
                                        // Simple pointer arithmetic: base + offset
                                        let val_i = *input_row_ptr.add(kw);
                                        let val_w = *weight_ptr;
                                        
                                        acc = acc.add(val_i.mul(val_w));
                                        
                                        // Weights are stored contiguously [ic, kh, kw], so we just increment.
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
        
        // Optimize memory allocation by pre-calculating size
        let stride = self.stride;

        // --- Strides ---
        let i_stride_c = in_height * in_width;
        let p_stride_c = padded_height * padded_width;
        let p_stride_b = in_channels * p_stride_c;
        let o_stride_oc = out_height * out_width;
        let o_stride_b = out_channels * o_stride_oc;
        let w_stride_ic = kernel_height * kernel_width;
        let w_stride_oc = in_channels * w_stride_ic;

        // --- 1. Parallel Padding (Input Reconstruction) ---
        // We need the padded input to compute dW.
        let mut padded_input_data = vec![T::default(); batch_size * p_stride_b];

        padded_input_data.par_chunks_exact_mut(p_stride_c)
            .zip(input.data.par_chunks_exact(i_stride_c))
            .for_each(|(pad_chan, in_chan)| {
                for h in 0..in_height {
                    let src_idx = h * in_width;
                    let dst_idx = (h + padding) * padded_width + padding;
                    // Intrinsic memcpy
                    pad_chan[dst_idx..dst_idx + in_width]
                        .copy_from_slice(&in_chan[src_idx..src_idx + in_width]);
                }
            });

        // --- 2. Parallel Gradient Computation (dW & dB) ---
        // We allocate the final buffers immediately to avoid intermediate collections.
        let mut grad_weights_data = vec![T::default(); out_channels * w_stride_oc];
        let mut grad_biases_data = vec![T::default(); out_channels];

        // Parallelize over Output Channels. 
        // Each thread has exclusive write access to its slice of dW and dB.
        grad_weights_data.par_chunks_exact_mut(w_stride_oc)
            .zip(grad_biases_data.par_iter_mut())
            .enumerate()
            .for_each(|(oc, (dw_slice, db_val))| {
                let mut local_db = T::default();
                
                // Pointers to avoid repeated base calculations
                // grad_out base for this channel (across all batches)
                // We will hop by o_stride_b to get the next batch
                let go_oc_offset = oc * o_stride_oc;

                // Optimization: Iterate Batch -> Spatial -> InputChannel
                for b in 0..batch_size {
                    let go_batch_base = b * o_stride_b + go_oc_offset;
                    let input_batch_base = b * p_stride_b;
                    
                    // Access GradOutput and Input using pointers
                    let go_ptr_base = unsafe { grad_output.data.as_ptr().add(go_batch_base) };
                    let input_ptr_base = unsafe { padded_input_data.as_ptr().add(input_batch_base) };

                    for oh in 0..out_height {
                        let h_start = oh * stride;
                        for ow in 0..out_width {
                            let w_start = ow * stride;
                            
                            // 1. Load Gradient Output
                            let grad_out_val = unsafe { *go_ptr_base.add(oh * out_width + ow) };
                            local_db = local_db.add(grad_out_val);

                            // 2. Accumulate into dW
                            // We iterate kernel logic here.
                            // We use a raw pointer for the weights slice to avoid bounds checks in the deep loop.
                            let mut dw_ptr = dw_slice.as_mut_ptr();

                            for ic in 0..in_channels {
                                let input_ic_ptr = unsafe { input_ptr_base.add(ic * p_stride_c) };
                                
                                for kh in 0..kernel_height {
                                    // Pre-calculate row pointer for input
                                    let in_row_ptr = unsafe { 
                                        input_ic_ptr.add((h_start + kh) * padded_width + w_start) 
                                    };

                                    // Hot Loop: Kernel Width
                                    unsafe {
                                        for kw in 0..kernel_width {
                                            let x_val = *in_row_ptr.add(kw);
                                            // dw += grad_out * x
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

        // --- 3. Parallel Input Gradient (dX) ---
        // Logic: "Scatter" or Transposed Convolution.
        // We parallelize by (Batch x InputChannel). This ensures every thread has 
        // EXCLUSIVE access to a slice of `grad_input_padded`, so no atomic adds are needed.
        let mut grad_input_padded = vec![T::default(); batch_size * p_stride_b];

        grad_input_padded.par_chunks_exact_mut(p_stride_c)
            .enumerate()
            .for_each(|(flat_idx, ic_slice)| {
                let b = flat_idx / in_channels;
                let ic = flat_idx % in_channels;
                let ic_w_offset = ic * w_stride_ic; // Weight offset for this Input Channel

                // Pre-calculate base pointers
                let go_batch_base = unsafe { grad_output.data.as_ptr().add(b * o_stride_b) };
                
                // Iterate over all Output Channels
                for oc in 0..out_channels {
                    let go_oc_base = unsafe { go_batch_base.add(oc * o_stride_oc) };
                    
                    // Weights for this (OC, IC) pair
                    let w_base_offset = oc * w_stride_oc + ic_w_offset;
                    let w_ptr_base = unsafe { self.weights.data.as_ptr().add(w_base_offset) };

                    for oh in 0..out_height {
                        let ih_start = oh * stride;
                        for ow in 0..out_width {
                            let iw_start = ow * stride;
                            
                            // Load grad_output(b, oc, oh, ow)
                            let go_val = unsafe { *go_oc_base.add(oh * out_width + ow) };
                            
                            // Inner Convolution Loop (Transposed)
                            for kh in 0..kernel_height {
                                let dst_row_idx = (ih_start + kh) * padded_width + iw_start;
                                let w_row_idx = kh * kernel_width;

                                // HOT LOOP: Kernel Width
                                // We add (grad_out * weight) to the input buffer
                                unsafe {
                                    for kw in 0..kernel_width {
                                        let w_val = *w_ptr_base.add(w_row_idx + kw);
                                        // ic_slice[dst] += go_val * w_val
                                        let dst_ptr = ic_slice.as_mut_ptr().add(dst_row_idx + kw);
                                        *dst_ptr = (*dst_ptr).add(go_val.mul(w_val));
                                    }
                                }
                            }
                        }
                    }
                }
            });

        // --- 4. Unpad (Copy padded dX to final dX) ---
        let mut grad_input_data = vec![T::default(); input.data.len()];

        // Parallel unpadding
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

        // Update Self
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
        T: Send + Sync + Copy + PlainElement, // Assumo PlainElement supporti le op matematiche
    {
        // 1. Inizializzazione Lazy delle Velocity
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

        // 2. Estrazione sicura
        if let (Some(grad_w), Some(grad_b), Some(vel_w), Some(vel_b)) = (
            &self.grad_weights,
            &self.grad_biases,
            &mut self.velocity_weights,
            &mut self.velocity_biases
        ) {
            // --- AGGIORNAMENTO PESI (Parallelo su ogni singolo peso) ---
            // Iteriamo su 3 vettori insieme: Weights, Grads, Velocities
            self.weights.data.par_iter_mut()
                .zip(grad_w.data.par_iter())
                .zip(vel_w.data.par_iter_mut())
                .for_each(|((w, &g), v)| {
                    // Formula:
                    // 1. Weight Decay: g' = g + (wd * w)
                    // 2. Momentum:     v = (mu * v) + g'
                    // 3. Update:       w = w - (lr * v)
                    
                    // Nota: Uso metodi simili a quelli che hai postato (.mul, .add, .sub)
                    // Se T supporta +, -, *, usa quelli per leggibilità.
                    
                    let wd_term = w.mul(weight_decay);
                    let g_prime = g.add(wd_term);
                    
                    let v_momentum = v.mul_inf(momentum);
                    *v = v_momentum.add(g_prime); // Aggiorna stato velocity
                    
                    let step = v.mul(learning_rate);
                    *w = w.sub(step);
                });

            // --- AGGIORNAMENTO BIAS (Parallelo) ---
            // Solitamente NO Weight Decay sui bias
            self.biases.data.par_iter_mut()
                .zip(grad_b.data.par_iter())
                .zip(vel_b.data.par_iter_mut())
                .for_each(|((b, &g), v)| {
                    // Formula Bias:
                    // 1. Momentum: v = (mu * v) + g
                    // 2. Update:   b = b - (lr * v)
                    
                    let v_momentum = v.mul(momentum);
                    *v = v_momentum.add(g); // Aggiorna stato velocity
                    
                    let step = v.mul(learning_rate);
                    *b = b.sub(step);
                });

        } else {
            panic!("Missing gradients or velocities for convolution layer");
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