use crate::plain_nn_builder::plain_utils::{PlainElement, PlainValueType};
use crate::plain_nn_builder::plain_ops::*;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IndexedParallelIterator;
use std::cmp::max;

#[derive(Clone)]
pub struct PlainTensor<T: PlainElement> {
    pub data: Vec<T>,
    pub shape: Vec<usize> //[batch, channels, height, width]
}

impl<T: PlainElement> PlainTensor<T> {

    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Returns the flat index from multi-dimensional indices
    pub fn flatten_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Dimension mismatch in indexing");
        
        let mut index = 0;
        let mut stride = 1;   
        for i in (0..self.shape.len()).rev() {
            let dim_size = self.shape[i];
            let idx = indices[i];
            
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

    #[allow(dead_code)]
    pub fn get_tensor(&self) -> PlainTensor<T> {
        PlainTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }

    pub fn matmul(
        &self,
        other: &PlainTensor<T>,
    )-> PlainTensor<T>
    where 
        T: PlainAdd + PlainMul + PlainValueType + Copy,
    {
        assert_eq!(self.shape.len(), 4, "Left tensor must be 4D");
        assert_eq!(other.shape.len(), 4, "Right tensor must be 4D");
    
        let [batch, channel, h1, w1] = self.shape[..] else {
            panic!("Left tensor shape must be [B, C, H1, W1]");
        };
    
        let [_, c2, h2, w2] = other.shape[..] else {
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
                    let prod = a_val.clone().mul(b_val.clone());
                    products.push(prod);
                }
    
                // Tree-reduced addition of the products
                while products.len() > 1 {
                    let mut next = Vec::with_capacity((products.len() + 1) / 2);
                    for pair in products.chunks(2) {
                        if pair.len() == 2 {
                            next.push(pair[0].clone().add(pair[1].clone()));
                        } else {
                            next.push(pair[0].clone());
                        }
                    }
                    products = next;
                }
    
                products.pop().unwrap()
            })
            .collect();
    
        PlainTensor::new(result_data, result_shape)
    }

    pub fn approx_matmul(
        &self,
        other: &PlainTensor<T>,
    )-> PlainTensor<T>
    where 
        T: PlainAdd + PlainMulInf + PlainValueType + Copy,
    {
        assert_eq!(self.shape.len(), 4, "Left tensor must be 4D");
        assert_eq!(other.shape.len(), 4, "Right tensor must be 4D");
    
        let [batch, channel, h1, w1] = self.shape[..] else {
            panic!("Left tensor shape must be [B, C, H1, W1]");
        };
    
        let [_, c2, h2, w2] = other.shape[..] else {
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
                    let prod = a_val.clone().mul_inf(b_val.clone());
                    products.push(prod);
                }
    
                // Tree-reduced addition of the products
                while products.len() > 1 {
                    let mut next = Vec::with_capacity((products.len() + 1) / 2);
                    for pair in products.chunks(2) {
                        if pair.len() == 2 {
                            next.push(pair[0].clone().add(pair[1].clone()));
                        } else {
                            next.push(pair[0].clone());
                        }
                    }
                    products = next;
                }
    
                products.pop().unwrap()
            })
            .collect();
    
        PlainTensor::new(result_data, result_shape)
    }


    /// Element-wise addition of two tensors with the same shape.
    pub fn add(
        &self,
        other: &PlainTensor<T>,
    ) -> PlainTensor<T>
    where
        T: PlainAdd,
    {
        assert_eq!(self.shape, other.shape, "Shape mismatch for add");
    
        
        let data: Vec<T> = self
        .data
        .par_iter()
        .zip(other.data.par_iter()) 
        .map(|(a, b)| a.clone().add(b.clone())) 
        .collect();
            
        PlainTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(
        &self,
        other: &PlainTensor<T>,
    ) -> PlainTensor<T>
    where
        T: PlainSub,
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
                a.sub(b)
            })
            .collect();
    
        PlainTensor::new(result_data, shape)
    }

    pub fn transpose(&self) -> PlainTensor<T> {
        assert_eq!(self.shape.len(), 4, "Transpose supports 4D tensors only");
        let batches = self.shape[0];
        let channels = self.shape[1];
        let rows = self.shape[2];
        let cols = self.shape[3];

        let mut transposed_data = Vec::with_capacity(self.data.len());

        for b in 0..batches {
            for c in 0..channels {
                for col in 0..cols {
                    for row in 0..rows {
                        transposed_data.push(self.get(&[b, c, row, col]).clone());
                    }
                }
            }
        }
        PlainTensor::new(transposed_data, vec![self.shape[0], self.shape[1], cols, rows])
    }

    pub fn sum_on_first_axis(&self) -> PlainTensor<T>
    where
        T: PlainAdd,
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
                                chunk[0].clone().add(chunk[1].clone())
                            } else {
                                chunk[0].clone()
                            }
                        })
                        .collect();
                }

                values.into_iter().next().unwrap()
            })
            .collect();

        PlainTensor::new(result_data, vec![1, channel, height, width])
    }

    pub fn mul_scalar(&self, scalar: &T) -> Self
    where
       T: PlainMul
    {
        let data = self.data.par_iter().map(|x| x.clone().mul(scalar.clone())).collect();

        PlainTensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn tanh(
        &self,
    ) -> (PlainTensor<T>, PlainTensor<T>)
    where
        T: PlainTanh
    {
        let (result_data, derivatives): (Vec<_>, Vec<_>) = self
            .data
            .iter()
            .map(|value| {
                value.clone().tanh()
            })
            .unzip();
    
        (
            PlainTensor::new(result_data, self.shape.clone()),
            PlainTensor::new(derivatives, self.shape.clone()),
        )
    }

    pub fn relu(
        &self,
    ) -> (PlainTensor<T>, PlainTensor<T>)
    where
        T: PlainReLU
    {
        let (result_data, derivatives): (Vec<_>, Vec<_>) = self
            .data
            .iter()
            .map(|value| {
                value.clone().relu()
            })
            .unzip();
    
        (
            PlainTensor::new(result_data, self.shape.clone()),
            PlainTensor::new(derivatives, self.shape.clone()),
        )
    }

    pub fn max(&self) -> T
    where
        T: Copy + Ord, 
    {
        self.data
            .iter()
            .copied() 
            .reduce(max)
            .expect("Empty tensor has no max")
    }

    pub fn flatten_hw_to_1d(&self) -> PlainTensor<T> {
         assert_eq!(self.shape.len(), 4, "Tensor must be 4D [B, C, H, W]");
        let [batch, channel, height, width] = self.shape[..] else {
            panic!("Invalid shape length");
        };

        let mut result_data = Vec::with_capacity(batch * channel * height * width);

        for b in 0..batch {
            for c in 0..channel {
                for h in 0..height {
                    for w in 0..width {
                        let val = self.get(&[b, c, h, w]).clone();
                        result_data.push(val);
                    }
                }
            }
        }

        PlainTensor {
            data: result_data,
            shape: vec![batch, 1, 1, channel * height * width],
        }
    }

    pub fn unflatten_1d_to_hw(&self, original_shape: &[usize; 4]) -> PlainTensor<T> where T: Default {
        let [batch, channel, height, width] = *original_shape;

        assert_eq!(self.shape.len(), 4, "Flattened tensor must be 4D [B,1,1,C*H*W]");
        assert_eq!(
            self.shape[0], batch,
            "Batch size must match the original shape"
        );

        if self.shape[3] != (channel * height * width) {
            return self.clone();
        }

        let mut result_data = vec![T::default(); batch * channel * height * width];

        for b in 0..batch {
            for c in 0..channel {
                for h in 0..height {
                    for w in 0..width {
                        let flat_idx = c * height * width + h * width + w;
                        let val = self.get(&[b, 0, 0, flat_idx]).clone();
                        let dst_idx = b * channel * height * width + c * height * width + h * width + w;
                        result_data[dst_idx] = val;
                    }
                }
            }
        }

        PlainTensor {
            data: result_data,
            shape: vec![batch, channel, height, width],
        }
    }

    pub fn print_tensor(&self) where T: PlainValueType + Copy {
        let prediction = self.clone();
        let size = prediction.shape[0];
        let channel = prediction.shape[1];
        let rows = prediction.shape[2];
        let cols = prediction.shape[3];
        let flat = &prediction.data;

        if flat.len() != size * channel * rows * cols {
            println!("Shape mismatch: expected {} elements, got {}", size * channel * rows * cols, flat.len());
            return;
        }

        for b in 0..size {
            println!("\n=== Batch {} ===", b);
            for c in 0..channel {
                println!("--- Channel {} ---", c);
                for i in 0..rows {
                    print!("[");
                    for j in 0..cols {
                        let index = b * channel * rows * cols
                            + c * rows * cols
                            + i * cols
                            + j;
                        print!("{:<6.3} ", flat[index].to_f32());
                    }
                    println!("]");
                }
            }
        }
    }


}