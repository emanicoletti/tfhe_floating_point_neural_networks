use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;
use rayon::iter::*;


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

        let mut padded_input = PlainTensor {
            data: vec![T::default(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        // Copy original input into padded tensor
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
        let (batch_size, in_channels, in_height, in_width) =
            (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, kernel_height, kernel_width) =
            (self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]);

        let out_height = (in_height + 2 * self.padding - kernel_height) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_width) / self.stride + 1;

        let grad_output = grad_output.unflatten_1d_to_hw(&[batch_size, out_channels, out_height, out_width]);

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

        let padded_height = in_height + 2 * self.padding;
        let padded_width = in_width + 2 * self.padding;

        let mut grad_input_padded = PlainTensor {
            data: vec![T::default(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        let mut padded_input = PlainTensor {
            data: vec![T::default(); batch_size * in_channels * padded_height * padded_width],
            shape: vec![batch_size, in_channels, padded_height, padded_width],
        };

        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let src = input.flatten_index(&[b, c, h, w]);
                        let dst = padded_input.flatten_index(&[b, c, h + self.padding, w + self.padding]);
                        padded_input.data[dst] = input.data[src];
                    }
                }
            }
        }

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let grad_out_val = grad_output.data[grad_output.flatten_index(&[b, oc, oh, ow])];

                        grad_biases.data[oc] = grad_biases.data[oc].add(grad_out_val);

                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * self.stride + kh; 
                                    let iw = ow * self.stride + kw; 

                                    let inp_idx = padded_input.flatten_index(&[b, ic, ih, iw]);
                                    let x_val = padded_input.data[inp_idx];

                                    let w_idx = self.weights.flatten_index(&[oc, ic, kh, kw]);
                                    let w_val = self.weights.data[w_idx];

                                    grad_weights.data[w_idx] =
                                        grad_weights.data[w_idx].add(grad_out_val.mul(x_val));

                                    let gip_idx = grad_input_padded.flatten_index(&[b, ic, ih, iw]);
                                    grad_input_padded.data[gip_idx] =
                                        grad_input_padded.data[gip_idx].add(grad_out_val.mul(w_val));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        for b in 0..batch_size {
            for ic in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let padded_idx = grad_input_padded.flatten_index(&[b, ic, h + self.padding, w + self.padding]);
                        let idx = grad_input.flatten_index(&[b, ic, h, w]);
                        grad_input.data[idx] = grad_input_padded.data[padded_idx];
                    }
                }
            }
        }

        self.grad_weights = Some(grad_weights);
        self.grad_biases = Some(grad_biases);

        grad_input
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