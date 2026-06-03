use crate::plain_nn_builder::plain_layers::PlainLayer;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;

pub struct PlainAvgPoolingLayer<T: PlainElement> {
    _input: PlainTensor<T>,
    _input_dim: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    id: String,
}

impl<T: PlainElement> PlainAvgPoolingLayer<T> {
    pub fn new(id: String, input_dim: Vec<usize>, kernel_size: usize, stride: usize) -> Self {
        Self {
            _input: PlainTensor::new(vec![], input_dim.clone()),
            id,
            _input_dim: input_dim,
            kernel_size,
            stride,
        }
    }
}

impl<T> PlainLayer<T> for PlainAvgPoolingLayer<T>
where
    T: PlainElement + Copy + Ord + PlainAdd + PlainDiv + Default + PlainValueType,
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
                        let mut sum = T::default();
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;
                                let val = input.get(&[n, c, ih, iw]).clone();
                                sum = sum.add(val);
                            }
                        }
                        let avg = sum.div(T::from_f32((kernel * kernel) as f32));
                        output.push(avg);
                    }
                }
            }
        }

        PlainTensor {
            data: output,
            shape: vec![batch, channels, out_height, out_width],
        }
    }

    fn backward(&mut self, _input: &PlainTensor<T>, grad_output: &PlainTensor<T>) -> PlainTensor<T>
    where
        T: Default + Copy + PlainAdd + PlainDiv,
        // T needs to support division, addition, and conversion from integer for the area
    {
        let kernel = self.kernel_size;
        let stride = self.stride;
        let (batch, channels, height, width) = (
            _input.shape[0],
            _input.shape[1],
            _input.shape[2],
            _input.shape[3],
        );

        let out_h = (height - kernel) / stride + 1;
        let out_w = (width - kernel) / stride + 1;

        let grad_output = grad_output.unflatten_1d_to_hw(&[batch, channels, out_h, out_w]);

        // Initialize grad_input with zeros
        let mut grad_input = PlainTensor::new(
            vec![T::default(); batch * channels * height * width],
            vec![batch, channels, height, width],
        );

        // Calculate the area to normalize the gradient
        // We cast kernel size to T (assuming T is f32/f64)
        let area = T::from_f32((kernel * kernel) as f32);

        for n in 0..batch {
            for c in 0..channels {
                for h in 0..out_h {
                    for w in 0..out_w {
                        // 1. Get the gradient from the next layer
                        let grad = grad_output.get(&[n, c, h, w]).clone();

                        // 2. Divide it by the window size (distribute equally)
                        let distributed_grad = grad.div(area);

                        // 3. Apply this distributed gradient to the whole window
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;

                                // Calculate flat index for the input gradient tensor
                                let flat_index = ((n * channels + c) * height + ih) * width + iw;

                                // IMPORTANT: Use += because windows might overlap
                                grad_input.data[flat_index] =
                                    grad_input.data[flat_index].add(distributed_grad);
                            }
                        }
                    }
                }
            }
        }

        grad_input
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn exact_inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn update_parameters(&mut self, _learning_rate: T, _weight_decay: T, _momentum: T) {
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
