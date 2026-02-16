use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::EncryptedContext;
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::{EncryptableValueType, EncryptedElement};
use crate::tfhe_nn_builder::encrypted_ops::{EncryptedAdd, EncryptedMul, EncryptedNegate, EncryptedMax, EncryptedGradIfEqual};
use crate::tfhe_nn_builder::encrypted_layers::EncryptedLayer;

pub struct EncryptedMaxPoolingLayer<T: EncryptedElement> {
    _input: EncryptedTensor<T>,
    _input_dim: Vec<usize>, 
    kernel_size: usize,
    stride: usize,
    id: String,
}

impl<T: EncryptedElement> EncryptedMaxPoolingLayer<T>{
    pub fn new(
        id: String,
        input_dim: Vec<usize>,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self {
            _input: EncryptedTensor::new(vec![], input_dim.clone()),
            id,
            _input_dim: input_dim,
            kernel_size,
            stride,
        }
    }
}

impl<K, T> EncryptedLayer<K, T> for EncryptedMaxPoolingLayer<T>
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedNegate<K, T> + EncryptedMax<K, T> + EncryptedGradIfEqual<K, T>,
    T: Clone + EncryptedElement + EncryptableValueType<>,
{
    fn forward(&mut self, input: &EncryptedTensor<T>, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T> {
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

                        let patch_tensor = EncryptedTensor {
                            data: patch,
                            shape: vec![kernel * kernel],
                        };

                        let max_val = patch_tensor.max(ctx);
                        output.push(max_val);
                    }
                }
            }
        }

        EncryptedTensor {
            data: output,
            shape: vec![batch, channels, out_height, out_width],
        }
    }

fn backward(
    &mut self,
    input: &EncryptedTensor<T>,
    grad_output: &EncryptedTensor<T>,
    ctx: &EncryptedContext<K, T>,
) -> EncryptedTensor<T> {
    let kernel = self.kernel_size;
    let stride = self.stride;
    let shape = &input.shape;
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let mut grad_input = EncryptedTensor::new(
        vec![ctx.encrypted_zero.clone(); batch * channels * height * width],
        vec![batch, channels, height, width],
    );

    let out_shape = [
        grad_output.shape[0],              
        grad_output.shape[1],              
        grad_output.shape[3] / kernel,     
        grad_output.shape[3] / kernel,     
      ]; 
    
    
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

                    let patch_tensor = EncryptedTensor {
                        data: patch.clone(),
                        shape: vec![kernel * kernel],
                    };
                    let max_val = patch_tensor.max(ctx);
                    let reshaped_grad_output = EncryptedTensor{
                        data: grad_output.data.clone(),
                        shape: out_shape.to_vec()
                    };
                    let grad = reshaped_grad_output.get(&[n, c, h, w]).clone();
                    for (idx, (ih, iw)) in indices.into_iter().enumerate() {
                        let val = &patch[idx];
                        let masked_grad = ctx.server_key.grad_if_equal(val.clone(), max_val.clone(), ctx.encrypted_zero.clone(), grad.clone(), ctx);
                        let flat_index = ((n * channels + c) * height + ih) * width + iw;
                        grad_input.data[flat_index] = masked_grad;
                    }
                }
            }
        }
    }
    grad_input
}

    fn update_parameters(
        &mut self,
        _learning_rate: T,
        _ctx: &EncryptedContext<K, T>,
    ) {
        // No parameters to update in max pooling
    }

    fn get_weights(&self) -> EncryptedTensor<T> {
        // No weights in max pooling
        EncryptedTensor {
            data: vec![],
            shape: vec![],
        }
    }

    fn get_biases(&self) -> EncryptedTensor<T> {
        // No biases in max pooling
        EncryptedTensor {
            data: vec![],
            shape: vec![],
        }
    }

    fn get_grad_weights(&self) -> EncryptedTensor<T> {
        // No gradients for weights in max pooling
        EncryptedTensor {
            data: vec![],
            shape: vec![],
        }
    }

    fn get_grad_biases(&self) -> EncryptedTensor<T> {
        // No gradients for biases in max pooling
        EncryptedTensor {
            data: vec![],
            shape: vec![],
        }
    }   

    fn get_id(&self) -> String {
        self.id.clone()
    }
}

