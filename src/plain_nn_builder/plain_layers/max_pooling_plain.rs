use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;

pub struct PlainMaxPoolingLayer<T: PlainElement> {
    input: PlainTensor<T>,
    input_dim: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    id: String,
}

impl<T: PlainElement> PlainMaxPoolingLayer<T>{
    pub fn new(
        id: String,
        input_dim: Vec<usize>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            input: PlainTensor::new(vec![], input_dim.clone()),
            id,
            input_dim,
            kernel_size,
            stride,
            padding,
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
        let shape = &input.shape;
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    
        let mut grad_input = PlainTensor::new(
            vec![T::default(); batch * channels * height * width],
            vec![batch, channels, height, width],
        );
    
        let out_shape = [
            grad_output.shape[0],              // batch size
            grad_output.shape[1],              // channels
            grad_output.shape[3] / kernel,     // output height after pooling
            grad_output.shape[3] / kernel,     // output width after pooling
          ]; // [B, C, OH, OW]
        
        
        for n in 0..batch {
            for c in 0..channels {
                for h in 0..out_shape[2] {
                    for w in 0..out_shape[3] {
                        let mut patch = Vec::new();
                        let mut indices = Vec::new();
    
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;
                                let val = input.get(&[n, c, ih, iw]).clone();
                                patch.push(val);
                                indices.push((ih, iw));
                            }
                        }
    
                        let patch_tensor = PlainTensor {
                            data: patch.clone(),
                            shape: vec![kernel * kernel],
                        };
                        let max_val = patch_tensor.max();
                        let reshaped_grad_output = PlainTensor{
                            data: grad_output.data.clone(),
                            shape: out_shape.to_vec()
                        };
                        let grad = reshaped_grad_output.get(&[n, c, h, w]).clone();
                        for (idx, (ih, iw)) in indices.into_iter().enumerate() {
                            let val = patch[idx];
                            let flat_index = ((n * channels + c) * height + ih) * width + iw;
                        
                            grad_input.data[flat_index] = if val == max_val {
                                grad
                            } else {
                                T::default()
                            };
                        }
                    }
                }
            }
        }
        grad_input
    }

    fn update_parameters(
        &mut self,
        learning_rate: T
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

