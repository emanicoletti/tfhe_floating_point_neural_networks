use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::PlainLayer;
use crate::plain_nn_builder::plain_ops::*;

use rayon::iter::*;

pub struct PlainTanhActivation<T: PlainElement> {
    pub id: String,
    pub derivatives: PlainTensor<T>, // same shape as input, stores derivative per element
    pub ranges: Vec<(T, T, T, T, T)>,    // piecewise segments: (min, max, a, b, derivative)
}

impl<T: PlainElement> PlainTanhActivation<T> {
    pub fn new(id: String, derivatives: PlainTensor<T>, ranges: Vec<(T, T, T, T, T)>) -> Self {
        Self {
            id,
            derivatives,
            ranges
        }
    }
}

impl<T> PlainLayer<T> for PlainTanhActivation<T>
where
    T: PlainTanh + PlainMul + Send + Sync + Clone + PlainElement, 
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> 
    {
        let (activations, derivatives ) = input.tanh();
        self.derivatives = derivatives;
        activations
    }

    fn backward(
        &mut self,
        input: &PlainTensor<T>,
        grad_output: &PlainTensor<T>,
    ) -> PlainTensor<T> 
    {
        // grad_input = grad_output * derivative
        let grad_input_data: Vec<T> = grad_output
        .data
        .par_iter()
        .zip(self.derivatives.data.par_iter())
        .map(|(g, d)| g.clone().mul(d.clone()))
        .collect();
    
        PlainTensor::new(grad_input_data, grad_output.shape.clone())
    }

    fn update_parameters(
        &mut self,
        learning_rate: T,
    ) {
    // No parameters to update in tanh activation
    }

    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        self.forward(input)
    }

    fn get_biases(&self) -> PlainTensor<T> {
        // No biases in tanh activation
        self.derivatives.clone()
    }

    fn get_grad_biases(&self) -> PlainTensor<T> {
        // No gradients for biases in tanh activation
        self.derivatives.clone()
    }   

    fn get_grad_weights(&self) -> PlainTensor<T> {
        // No gradients for weights in tanh activation
        self.derivatives.clone()
    }

    fn get_weights(&self) -> PlainTensor<T> {
        // No weights in tanh activation
        self.derivatives.clone()
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }   


}