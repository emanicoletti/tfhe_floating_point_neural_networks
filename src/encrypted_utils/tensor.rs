use crate::encrypted_ops::ops;
use crate::encrypted_utils::encrypted_types::EncryptedElement;
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_context::EncryptedContext;
use crate::encrypted_ops::{EncryptedAdd, EncryptedMul};

#[derive(Clone)]
pub struct EncryptedTensor<T: EncryptedElement> {
    pub data: Vec<T>,
    pub shape: Vec<usize>, // [batch, channels, height, width]
}

impl<T: EncryptedElement> EncryptedTensor<T> {
    /// Generate a new EncryptedTensor with validated shape
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(data.len(), expected_size, "Data length does not match shape dimensions");
        Self { data, shape }
    }

    /// Returns the flat index from multi-dimensional indices
    fn flatten_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Dimension mismatch in indexing");
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim_size) in self.shape.iter().rev().enumerate() {
            let idx = indices[self.shape.len() - 1 - i];
            assert!(idx < dim_size, "Index out of bounds");
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
    {
        assert_eq!(self.shape.len(), 2, "Left tensor must be 2D");
        assert_eq!(other.shape.len(), 2, "Right tensor must be 2D");
        let (m, n1) = (self.shape[0], self.shape[1]);
        let (n2, p) = (other.shape[0], other.shape[1]);
        assert_eq!(n1, n2, "Inner dimensions must match");
    
        let mut result_data = Vec::with_capacity(m * p);
        for i in 0..m {
            for j in 0..p {
                let mut sum = ctx.encrypted_zero.clone();
    
                for k in 0..n1 {
                    let a_val = self.get(&[i, k]).clone();
                    let b_val = other.get(&[k, j]).clone();
                    let prod = ctx.server_key.mul(a_val, b_val, ctx);
                    sum = ctx.server_key.add(sum, prod, ctx);
                }
    
                result_data.push(sum);
            }
        }
    
        EncryptedTensor::new(result_data, vec![m, p])
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
    
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| ctx.server_key.add(a.clone(), b.clone(), ctx))
            .collect();
    
        EncryptedTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn transpose(&self) -> EncryptedTensor<T> {
        assert_eq!(self.shape.len(), 2, "Transpose supports 2D tensors only");
        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut transposed_data = Vec::with_capacity(self.data.len());

        for col in 0..cols {
            for row in 0..rows {
                transposed_data.push(self.get(&[row, col]).clone());
            }
        }

        EncryptedTensor::new(transposed_data, vec![cols, rows])
    }

    pub fn sum_axis<K>(&self, axis: usize, ctx: &EncryptedContext<K, T>) -> EncryptedTensor<T>
    where
        K: ServerKeyTrait + EncryptedAdd<K, T>,
    {
        assert_eq!(self.shape.len(), 2, "sum_axis supports 2D tensors only");
        let (dim0, dim1) = (self.shape[0], self.shape[1]);
        assert!(axis == 0 || axis == 1, "Only axis 0 or 1 supported");

        match axis {
            0 => {
                // Sum over rows (batch), result shape: [features]
                let mut result_data = Vec::with_capacity(dim1);

                for col in 0..dim1 {
                    let mut sum = ctx.encrypted_zero.clone();
                    for row in 0..dim0 {
                        let val = self.get(&[row, col]).clone();
                        sum = ctx.server_key.add(sum, val, ctx);
                    }
                    result_data.push(sum);
                }

                EncryptedTensor::new(result_data, vec![dim1]) // 1D tensor for biases
            }
            1 => {
                // Sum over columns, result shape: [batch]
                let mut result_data = Vec::with_capacity(dim0);

                for row in 0..dim0 {
                    let mut sum = ctx.encrypted_zero.clone();
                    for col in 0..dim1 {
                        let val = self.get(&[row, col]).clone();
                        sum = ctx.server_key.add(sum, val, ctx);
                    }
                    result_data.push(sum);
                }

                EncryptedTensor::new(result_data, vec![dim0]) // 1D tensor
            }
            _ => unreachable!(),
        }
    }


}



