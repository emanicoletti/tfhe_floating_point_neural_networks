use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

pub struct PlainMaxPoolingLayer<T: PlainElement> {
    _input: PlainTensor<T>,
    _input_dim: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    id: String,
}

impl<T: PlainElement> PlainMaxPoolingLayer<T>{
    pub fn new(
        id: String,
        input_dim: Vec<usize>,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self {
            _input: PlainTensor::new(vec![], input_dim.clone()),
            id,
            _input_dim: input_dim,
            kernel_size,
            stride
        }
    }
}

impl<T> PlainLayer<T> for PlainMaxPoolingLayer<T>
where 
    T: PlainElement + Copy + Ord, 
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        let kernel = self.kernel_size;
        let stride = self.stride;
        let shape = input.shape.clone();
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

        let out_height = (height - kernel) / stride + 1;
        let out_width = (width - kernel) / stride + 1;
        let mut output = Vec::with_capacity(batch * channels * out_height * out_width);

        for n in 0..batch {
            for c in 0..channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let mut patch = Vec::with_capacity(kernel * kernel);
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;
                                let val = input.get(&[n, c, ih, iw]).clone();
                                patch.push(val);
                            }
                        }

                        let patch_tensor = PlainTensor {
                            data: patch,
                            shape: vec![kernel * kernel],
                        };

                        let max_val = patch_tensor.max();
                        output.push(max_val);
                    }
                }
            }
        }

        PlainTensor {
            data: output,
            shape: vec![batch, channels, out_height, out_width],
        }
    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T>
    where
        T: Default,
    {
        let kernel = self.kernel_size;
        let stride = self.stride;
        let (batch, channels, height, width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);

        let out_h = (height - kernel) / stride + 1;
        let out_w = (width - kernel) / stride + 1;

        let grad_output = grad_output.unflatten_1d_to_hw(&[batch, channels, out_h, out_w]);

        let mut grad_input = PlainTensor::new(
            vec![T::default(); batch * channels * height * width],
            vec![batch, channels, height, width],
        );

        for n in 0..batch {
            for c in 0..channels {
                for h in 0..out_h {
                    for w in 0..out_w {
                        let mut max_val = T::default();
                        let mut max_idx = (0, 0);
                        let mut first = true;

                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;
                                let val = input.get(&[n, c, ih, iw]).clone();

                                if first || val > max_val {
                                    max_val = val;
                                    max_idx = (ih, iw);
                                    first = false;
                                }
                            }
                        }

                        let grad = grad_output.get(&[n, c, h, w]).clone();
                        let flat_index = ((n * channels + c) * height + max_idx.0) * width + max_idx.1;
                        grad_input.data[flat_index] = grad;
                    }
                }
            }
        }

        grad_input
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn approximate_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn update_parameters(
        &mut self,
        _learning_rate: T
    ) {
        // No parameters to update in max pooling
    }

    fn get_weights(&self) -> PlainTensor<T> {
        // No weights in max pooling
        PlainTensor {
            data: vec![],
            shape: vec![],
        }
    }

    fn get_biases(&self) -> PlainTensor<T> {
        // No biases in max pooling
        PlainTensor {
            data: vec![],
            shape: vec![],
        }
    }

    fn get_grad_weights(&self) -> PlainTensor<T> {
        // No gradients for weights in max pooling
        PlainTensor {
            data: vec![],
            shape: vec![],
        }
    }

    fn get_grad_biases(&self) -> PlainTensor<T> {
        // No gradients for biases in max pooling
        PlainTensor {
            data: vec![],
            shape: vec![],
        }
    }   

    fn get_id(&self) -> String {
        self.id.clone()
    }

}

