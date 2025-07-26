use crate::encrypted_ops::{ops, EncryptedNegate};
use crate::encrypted_utils::encrypted_types::EncryptedElement;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_ops::{EncryptedAdd, EncryptedMul, EncryptedTanh, EncryptedMax};

use crate::encrypted_utils::encrypted_types::EncryptableValueType;

use rayon::prelude::*;

use half::f16;

use std::time::Instant;


#[derive(Clone)]
pub struct EncryptedTensor<T: EncryptedElement> {
    pub data: Vec<T>,
    pub shape: Vec<usize>, // [batch, channels, height, width]
}

impl<T: EncryptedElement> EncryptedTensor<T> {
    /// Generate a new EncryptedTensor with validated shape
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        //assert_eq!(data.len(), expected_size, "Data length does not match shape dimensions");
        Self { data, shape }
    }

    /// Returns the flat index from multi-dimensional indices
    fn flatten_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Dimension mismatch in indexing");
        
        let mut index = 0;
        let mut stride = 1;   
        for i in (0..self.shape.len()).rev() {
            let dim_size = self.shape[i];
            let idx = indices[i];
    
            // This assertion fails when index is invalid
            assert!(
                idx < dim_size,
                "Index {} out of bounds for dimension {} (size {})",
                idx,
                i,
                dim_size
            );
    
            index += idx * stride;
            stride *= dim_size;
        }
        index
    }

     /// Get a reference to an element
     pub fn get(&self, indices: &[usize]) -> &T {
        let idx = self.flatten_index(indices);
        &self.data[idx]
    }

    pub fn get_tensor(&self) -> EncryptedTensor<T> {
        EncryptedTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
    pub fn matmul<K>(
        &self,
        other: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedMul<K, T> + EncryptedAdd<K, T>,
        T: EncryptableValueType<Plain = u32> + Clone,
    {
        assert_eq!(self.shape.len(), 4, "Left tensor must be 4D");
        assert_eq!(other.shape.len(), 4, "Right tensor must be 4D");
    
        let [batch, channel, h1, w1] = self.shape[..] else {
            panic!("Left tensor shape must be [B, C, H1, W1]");
        };
    
        let [b2, c2, h2, w2] = other.shape[..] else {
            panic!("Right tensor shape must be [B, C, H2, W2]");
        };
    
        assert_eq!(channel, c2, "Channel dimensions must match");
        assert_eq!(w1, h2, "Inner dimensions must match for matmul");
    
        let result_shape = vec![batch, channel, h1, w2];
    
        let result_data = (0..batch * channel * h1 * w2)
            .into_par_iter()
            .map(|flat_index| {
                let b = (flat_index / (channel * h1 * w2)) % batch;
                let mut b_self = b;
                let mut b_other = b;
                if other.shape[0] == 1 {
                    b_other = 0;
                }
                if self.shape[0] == 1 {
                    b_self = 0;
                }
                let c = (flat_index / (h1 * w2)) % channel;
                let i = (flat_index / w2) % h1;
                let j = flat_index % w2;
    
                // Collect all multiplications first
                let mut products = Vec::with_capacity(w1);
                for k in 0..w1 {
                    let a_val = self.get(&[b_self, c, i, k]).clone();
                    let b_val = other.get(&[b_other, c, k, j]).clone();
                    let prod = ctx.server_key.mul(a_val, b_val, ctx);
                    products.push(prod);
                }
    
                // Tree-reduced addition of the products
                while products.len() > 1 {
                    let mut next = Vec::with_capacity((products.len() + 1) / 2);
                    for pair in products.chunks(2) {
                        if pair.len() == 2 {
                            next.push(ctx.server_key.add(pair[0].clone(), pair[1].clone(), ctx));
                        } else {
                            next.push(pair[0].clone());
                        }
                    }
                    products = next;
                }
    
                products.pop().unwrap_or_else(|| ctx.encrypted_zero.clone())
            })
            .collect();
    
        EncryptedTensor::new(result_data, result_shape)
    }

    /// Element-wise addition of two tensors with the same shape.
    pub fn add<K>(
        &self,
        other: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedAdd<K, T>,
    {
        assert_eq!(self.shape, other.shape, "Shape mismatch for add");
    
        let data: Vec<T> = self.data.par_iter()
        .zip(other.data.par_iter())
        .map(|(a, b)| ctx.server_key.add(a.clone(), b.clone(), ctx))
        .collect();
            
        EncryptedTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub<K>(
        &self,
        other: &EncryptedTensor<T>,
        ctx: &EncryptedContext<K, T>,
    ) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedNegate<K, T>,
        T: Clone + Send + Sync,
    {
        assert_eq!(self.shape.len(), other.shape.len(), "Tensors must have same rank");

        let shape = self.shape.clone();
    
        let result_data = (0..self.data.len())
            .into_par_iter()
            .map(|flat_index| {
                let mut remainder = flat_index;
                let mut idx = vec![0; shape.len()];
                for i in (0..shape.len()).rev() {
                    idx[i] = remainder % shape[i];
                    remainder /= shape[i];
                }
    
                let idx_other: Vec<usize> = idx
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if other.shape[i] == 1 { 0 } else { v })
                    .collect();
    
                let a = self.get(&idx).clone();
                let b = other.get(&idx_other).clone();
                ctx.server_key.add(a, ctx.server_key.negate(b), ctx)
            })
            .collect();
    
        EncryptedTensor::new(result_data, shape)
    }

    pub fn transpose(&self) -> EncryptedTensor<T> {
        assert_eq!(self.shape.len(), 4, "Transpose supports 4D tensors only");
        let batches = self.shape[0];
        let rows = self.shape[2];
        let cols = self.shape[3];

        let mut transposed_data = Vec::with_capacity(self.data.len());

        for b in 0..batches {
            for col in 0..cols {
                for row in 0..rows {
                    transposed_data.push(self.get(&[b, 0, row, col]).clone());
                }
            }
        }
        EncryptedTensor::new(transposed_data, vec![self.shape[0], self.shape[1], cols, rows])
    }

    pub fn sum_axis<K>(&self, axis: usize, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedAdd<K, T>,
        T: Send + Sync + Clone,
    {
        assert_eq!(self.shape.len(), 4, "sum_axis supports 4D tensors only");

        let (batch, channel, height, width) = (
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        );

        // Sum over batch dimension
        let result_data: Vec<T> = (0..channel * height * width)
            .into_par_iter()
            .map(|index| {
                let c = (index / (height * width)) % channel;
                let h = (index / width) % height;
                let w = index % width;

                // Gather values to sum
                let mut values: Vec<T> = (0..batch)
                    .map(|b| self.get(&[b, c, h, w]).clone())
                    .collect();

                // Fused tree reduction
                while values.len() > 1 {
                    values = values
                        .chunks(2)
                        .map(|chunk| {
                            if chunk.len() == 2 {
                                ctx.server_key.add(chunk[0].clone(), chunk[1].clone(), ctx)
                            } else {
                                chunk[0].clone()
                            }
                        })
                        .collect();
                }

                values.into_iter().next().unwrap()
            })
            .collect();

        EncryptedTensor::new(result_data, vec![1, channel, height, width])
    }

    pub fn mul_scalar<K>(&self, scalar: &T, ctx: &EncryptedContext<K, T>) -> Self
    where
        K: ServerKeyTrait + EncryptedMul<K, T>,
    {
        let data = self.data.par_iter().map(|x| ctx.server_key.mul(x.clone(), scalar.clone(), ctx)).collect();

        EncryptedTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn tanh<K>(
        &self,
        ctx: &EncryptedContext<K, T>,
    ) -> (EncryptedTensor<T>, EncryptedTensor<T>)
    where
        K: ServerKeyTrait + EncryptedTanh<K, T> + Sync,
        T: EncryptableValueType + Send + Sync,
    {
        let (result_data, derivatives): (Vec<_>, Vec<_>) = self
            .data
            .iter()
            .map(|value| {
                ctx.server_key.tanh(value.clone(), ctx)
            })
            .unzip();
    
        (
            EncryptedTensor::new(result_data, self.shape.clone()),
            EncryptedTensor::new(derivatives, self.shape.clone()),
        )
    }

    
    pub fn max<K>(&self, ctx: &EncryptedContext<K, T>) -> T
    where
        K: ServerKeyTrait + EncryptedMax<K, T> + Sync,
        T: EncryptableValueType + Send + Sync,
    {
        self.data.iter()
            .cloned()
            .reduce(|a, b| ctx.server_key.max(a, b, ctx))
            .expect("Empty tensor has no max")
    }

    pub fn flatten_hw_to_1d(&self) -> EncryptedTensor<T> {
        assert_eq!(self.shape.len(), 4, "Tensor must be 4D [B, C, H, W]");
        let [batch, channel, height, width] = self.shape[..] else {
            panic!("Invalid shape length");
        };

        let mut result_data = Vec::with_capacity(batch * channel * height * width);

        for b in 0..batch {
            for c in 0..channel {
                let mut flattened = Vec::with_capacity(height * width);
                for h in 0..height {
                    for w in 0..width {
                        let val = self.get(&[b, c, h, w]).clone();
                        flattened.push(val);
                    }
                }
                result_data.extend(flattened);
            }
        }

        EncryptedTensor {
            data: result_data,
            shape: vec![batch, channel, 1, height * width],
        }
    }
    
}



